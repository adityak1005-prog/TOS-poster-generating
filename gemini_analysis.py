"""
Stage 2: analysis + character match + prompt construction, collapsed into a
single multimodal call to Gemini.

Old pipeline (6 stages): MediaPipe pose -> MediaPipe face -> numpy color
quantization -> rule-based scorer -> occasional LLM tie-break -> separate
prompt-building function.

New pipeline (this file is the whole of stage 2): send the captured photo
straight to Gemini with one prompt that asks it to (a) look at the person,
(b) pick the best-matching character from a fixed list, (c) write a caption,
and (d) write a ready-to-use image-editing prompt -- all in one structured
JSON response. No local model, no local heuristics.

Model: gemini-2.5-flash-lite (falls back to gemini-2.0-flash via env var).
Both are free-tier eligible and fast enough for a single-image, low-token
JSON response to comfortably fit inside the booth's latency budget.

SDK: `google-genai` (the current SDK -- `import google.generativeai` is the
deprecated predecessor and is not used here).
"""

import os
import json
from google import genai
from google.genai import types
from pydantic import BaseModel

MODEL_ID = os.environ.get("GEMINI_MODEL","gemini-2.0-flash")

_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Fixed roster the model must choose from. Each entry carries a longer,
# hand-written *text* style anchor -- not reference images. Reference images
# of these characters are copyrighted studio/franchise material (Marvel, DC,
# Peaky Blinders, Naruto, Mattel), so instead of sourcing and re-transmitting
# actual character artwork to a third-party API, the style is described in
# prose. Gemini already knows what these characters look like from training,
# and gpt-image-1 (stage 3) only needs a text prompt anyway -- so no reference
# image is lost by doing it this way, and it sidesteps any IP/licensing
# question around storing or shipping character images with the app.
#
# Descriptions are intentionally long and specific (build, outfit/colors,
# expression, iconic prop, backdrop/lighting) -- more concrete visual anchors
# here means a more accurate match against the photo, since this is the only
# grounding Gemini gets for each character.
CHARACTER_STYLE_GUIDE = {
    "Batman": (
        "A brooding, powerfully built vigilante in a matte-black armored "
        "bat-suit with a flowing cape and a cowl with pointed ears framing a "
        "stern, unsmiling jaw. Utility belt with gadget pouches, muscular "
        "silhouette, gauntlets with blade-like fins. Shot in moody "
        "Gotham-noir lighting -- deep shadows, cold blue-grey highlights, "
        "rain-slicked rooftops or a dark alley backdrop. Expression is "
        "intense, guarded, rarely smiling."
    ),
    "Iron Man": (
        "A confident, athletic figure in sleek red-and-gold powered armor -- "
        "articulated metal plating, a glowing blue arc reactor at the center "
        "of the chest, repulsor palms raised in a ready stance. Faceplate is "
        "polished and reflective with narrow glowing eye-slits, or open to "
        "reveal a sharp goatee and smirk underneath. Backdrop is bright, "
        "high-tech -- lab light, sky, or explosion glow. Pose reads as "
        "heroic and self-assured, chin slightly raised."
    ),
    "Joker": (
        "A lean, theatrical figure in a rumpled purple tailcoat suit over a "
        "green vest, with slicked-back or unkempt green-dyed hair. "
        "Chalk-white face paint, a smeared red smile carved wide across the "
        "mouth, dark heavy eye makeup. Pose is loose and expressive -- a "
        "wide grin, jazz hands, or leaning in menacingly. Backdrop is "
        "chaotic neon-lit city grime, graffiti, or flickering carnival "
        "light."
    ),
    "Spider-Man": (
        "An athletic, lithe figure in a skin-tight red-and-blue spandex "
        "suit with bold black web patterning and large white reflective eye "
        "lenses. Pose is dynamic and acrobatic -- mid-swing, crouched on a "
        "ledge, or a classic web-shooting hand gesture. Backdrop is a dense "
        "city skyline, skyscrapers and web-lines cutting across a dusk sky. "
        "Overall feel is energetic, youthful, and playful."
    ),
    "Thomas Shelby": (
        "A sharply dressed, composed figure in a fitted 1920s tweed "
        "three-piece suit with a flat cap pulled low, hair slicked back and "
        "undercut at the sides. Expression is unreadable, jaw set, eyes "
        "narrowed with quiet menace, often holding or about to light a "
        "cigarette. Backdrop is muted sepia-toned Birmingham streets, "
        "factory smoke, or a dim pub interior. Mood is restrained power, "
        "cold calculation, old-world formality."
    ),
    "Naruto": (
        "An energetic, wiry young ninja in an orange-and-black jumpsuit, a "
        "blue forehead protector with a metal leaf-village emblem, and "
        "spiky sun-bleached blonde hair. Whisker-like markings on each "
        "cheek, wide grinning expression, often mid-jump or making a "
        "hand-sign with one fist raised. Backdrop is a dynamic anime-style "
        "burst of energy, orange and blue chakra swirls, motion lines. Mood "
        "is upbeat, determined, full of youthful energy."
    ),
    "Black Widow": (
        "A lean, athletic figure in a fitted black tactical bodysuit with "
        "subtle utility straps and holsters, in a confident, combat-ready "
        "stance -- often crouched low or mid-motion. Wavy red hair, sharp "
        "focused eyes, minimal but striking makeup, a calm and controlled "
        "expression that reads as dangerous competence rather than "
        "aggression. Backdrop is cool steel-blue or shadowed tactical "
        "environments -- a dim corridor, rain, or muted urban night. Mood "
        "is composed, precise, quietly formidable."
    ),
    "Barbie": (
        "A poised, glamorous figure dressed in vibrant hot-pink -- a fitted "
        "retro-style dress or gingham outfit, matching accessories, and "
        "perfectly styled voluminous blonde hair. Pose is cheerful and "
        "confident, a bright open smile, hand on hip or a playful wave, "
        "evoking classic plastic-doll perfection. Backdrop is a saturated "
        "pastel-and-pink dreamland -- a plastic dollhouse aesthetic, sunny "
        "sky, or a convertible car scene. Mood is upbeat, glossy, and full "
        "of exaggerated glamour."
    ),
}

CHARACTERS = list(CHARACTER_STYLE_GUIDE.keys())


class DetectedTraits(BaseModel):
    outfit_color: str
    pose_description: str
    glasses: bool


class BoothMatch(BaseModel):
    character: str
    reasoning: str
    caption: str
    diffusion_prompt: str
    detected_traits: DetectedTraits


def _build_instruction() -> str:
    roster_lines = "\n".join(
        f'- {name}: {style}' for name, style in CHARACTER_STYLE_GUIDE.items()
    )
    return f"""You are the analysis brain for a college-fest photo booth.
Look at the attached photo of a person and do all of the following in one pass:

1. Pick the single best-matching character for this person from this fixed
   list only (do not invent a character not on this list):
{roster_lines}

Base the match on whatever is visually striking in the photo -- pose, outfit
color, expression, accessories (e.g. glasses) -- combined with the mood/style
of each character above.

2. Write "reasoning": one or two short sentences, written directly to the
   person ("you"), explaining *why* they were matched to this character --
   ground it in specific things you actually saw (e.g. "Your crossed arms and
   steady gaze read as calm authority, and the dark jacket matches Batman's
   brooding palette."). Concrete and specific, not generic.

3. Write "caption": a short, witty, one-line caption (8-12 words) about the
   match. Fun and a bit cheeky, never mean-spirited.

4. Write "diffusion_prompt": a single ready-to-use prompt for an image-editing
   model that will restyle THIS photo. Keep it TIGHT -- about 50-70 words,
   one flowing sentence, no filler adjectives repeated for effect. Pack in
   only concrete, high-signal visual anchors: (a) the person keeps their own
   pose/framing, (b) 2-4 specific visual details pulled from the chosen
   character's style cues above (exact colors/props/hair, not vague mood
   words), (c) one lighting/backdrop cue, (d) "cinematic movie-poster
   quality, college fest promotional poster style" as a closing style tag.
   A short, dense prompt lets the image model converge faster with the same
   fidelity -- do not pad it with restated synonyms.

5. Fill "detected_traits": your own quick read of the photo --
   - outfit_color: the single dominant clothing color you see, one word
     (e.g. "red", "blue", "black", "gray")
   - pose_description: a short phrase describing their pose/stance
     (e.g. "arms crossed", "peace sign", "neutral standing pose")
   - glasses: true or false, whether they're wearing glasses/sunglasses

Respond with JSON matching the required schema only."""


def analyze_and_match(image_bytes: bytes, mime_type: str = "image/jpeg") -> dict:
    """
    Stage 2 entry point. Takes the captured photo's raw bytes, returns:
    {
      "character": "...",
      "caption": "...",
      "diffusion_prompt": "...",
      "detected_traits": {"outfit_color": "...", "pose_description": "...", "glasses": bool}
    }
    """
    response = _client.models.generate_content(
        model=MODEL_ID,
        contents=[
            _build_instruction(),
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            # CRITICAL FIX: We removed response_schema=BoothMatch to prevent 
            # the SDK from crashing internally while trying to parse the output.
            temperature=0.8, 
        ),
    )

    # Safely extract the text, falling back to dictionary traversal 
    # if the SDK's .text property still bugs out.
    try:
        raw_text = response.text
    except AttributeError:
        response_dict = response.model_dump() if hasattr(response, "model_dump") else dict(response)
        candidates = response_dict.get("candidates", [])
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        raw_text = parts[0].get("text", "") if parts else "{}"

    # Parse the JSON string into a dictionary manually
    result = json.loads(raw_text)

    # Clamp off-roster results
    if result.get("character") not in CHARACTERS:
        result["character"] = CHARACTERS[0]

    # Defensive default in case the model ever omits the field
    result.setdefault("reasoning", "")

    return result