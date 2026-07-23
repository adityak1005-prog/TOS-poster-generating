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

Model: gemini-2.5-flash-lite by default (override via GEMINI_MODEL env var).
Free-tier eligible and fast enough for a single-image, low-token JSON
response to comfortably fit inside the booth's latency budget.

NOTE: gemini-2.0-flash -- previously used as the hardcoded fallback here --
was shut down by Google on 2026-06-01. The fallback below now points at
gemini-2.5-flash-lite so the booth still works even if GEMINI_MODEL is
unset in a given environment; don't reintroduce 2.0-flash as a fallback.

SDK: `google-genai` (the current SDK -- `import google.generativeai` is the
deprecated predecessor and is not used here).
"""

import os
import re
import json
from google import genai
from google.genai import types
from pydantic import BaseModel

MODEL_ID = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")

_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Fixed roster the model must choose from. Each entry carries a longer,
# hand-written *text* style anchor. These are always sent; if a matching
# reference image also exists in CHARACTER_REF_DIR (see the caching section
# below), it's used alongside the text for stronger visual grounding -- but
# text-only is the default and everything works with zero images present.
#
# Descriptions are intentionally long and specific (build, outfit/colors,
# expression, iconic prop, backdrop/lighting) -- more concrete visual anchors
# here means a more accurate match against the photo.
CHARACTER_STYLE_GUIDE = {
    "Iron Man": (
        "A confident, athletic figure in sleek red-and-gold powered armor -- "
        "articulated metal plating, a glowing blue arc reactor at the center "
        "of the chest, repulsor palms raised in a ready stance. Faceplate is "
        "polished and reflective with narrow glowing eye-slits, or open to "
        "reveal a sharp goatee and smirk underneath. Backdrop is bright, "
        "high-tech -- lab light, sky, or explosion glow. Pose reads as "
        "heroic and self-assured, chin slightly raised."
    ),
    "Batman": (
        "A brooding, powerfully built vigilante in a matte-black armored "
        "bat-suit with a flowing cape and a cowl with pointed ears framing a "
        "stern, unsmiling jaw. Utility belt with gadget pouches, muscular "
        "silhouette, gauntlets with blade-like fins. Shot in moody "
        "Gotham-noir lighting -- deep shadows, cold blue-grey highlights, "
        "rain-slicked rooftops or a dark alley backdrop. Expression is "
        "intense, guarded, rarely smiling."
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
    "Professor from Money Heist": (
        "A calm, calculating mastermind -- disheveled academic look with "
        "thick-rimmed glasses, tousled hair, and a rumpled cardigan when "
        "planning, contrasted with the crew's signature red jumpsuit and a "
        "white Salvador Dali mask (wide painted grin, rosy cheeks, thin "
        "mustache) when in heist mode. Composed, quietly intense gaze that "
        "reads as always three steps ahead. Backdrop is a bank vault stacked "
        "with gold bars or a sunbaked Spanish plaza. Mood is restrained, "
        "cerebral, unshakeable control."
    ),
    "Gwen Stacy": (
        "An athletic, confident figure in a pink-and-white hooded Spider "
        "suit with a bold black spider emblem, hood down to show "
        "blonde-and-pink-streaked hair and a smirking, self-assured "
        "expression. Pose is dynamic -- perched on a ledge, mid-leap, or "
        "leaning back casually. Backdrop is a comic-book-style graffiti "
        "splash of pink and blue against a city skyline. Mood is cool, "
        "playful, quietly rebellious."
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
    "Daenerys Targaryen": (
        "A regal, striking figure with long silver-platinum braided hair and "
        "a flowing pale ivory or deep-blue gown with intricate detailing. "
        "Expression is composed but fierce -- calm authority with the "
        "implication of real power underneath. Subtle dragon-scale texture "
        "or drifting smoke/embers in the frame. Backdrop is a sunbaked "
        "desert city, a grand throne hall, or a dragon's silhouette against "
        "the sky. Mood is commanding, regal, quietly dangerous."
    ),
    "Hermione Granger": (
        "A sharp, studious figure with bushy brown hair and Gryffindor "
        "school robes -- black robe, maroon-and-gold tie, house crest. "
        "Often shown mid-gesture with a wand raised, holding a stack of "
        "books, or with a focused, thinking expression. Backdrop is a "
        "candlelit castle library or stone corridor. Mood is intelligent, "
        "quietly confident, prepared for anything."
    ),
    "Wednesday Addams": (
        "A pale, deadpan figure with jet-black hair in two tight braids, "
        "dressed in a black Victorian-collared dress or a dark academy "
        "uniform. Expression is flat, unimpressed, and unblinking -- no "
        "smile, dry composure. Backdrop is a foggy graveyard, a gothic "
        "academy hallway, or muted grey-and-black tones throughout. Mood is "
        "dark, dry-witted, completely unbothered."
    ),
    "Kakashi Hatake": (
        "A relaxed but sharp-eyed ninja with messy silver hair and a Leaf "
        "Village forehead protector tilted to cover one eye. Navy-blue "
        "flak vest over a dark long-sleeve uniform, a fabric mask covering "
        "the nose and mouth, one hand often in his pocket or holding an "
        "orange book. Backdrop is a misty forest or a village rooftop at "
        "dusk. Mood is cool, effortlessly skilled, quietly amused."
    ),
    "Sasuke Uchiha": (
        "A dark-haired, sharp-featured young ninja with a brooding, intense "
        "stare -- red-and-black Sharingan eye detail implied. Navy-and-white "
        "high-collared ninja outfit, a katana slung across the back. Pose "
        "is still and coiled, ready to strike. Backdrop is crackling blue "
        "lightning energy or a dark storm-lit battlefield. Mood is aloof, "
        "intense, quietly powerful."
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
    "Jon Snow": (
        "A brooding, weathered figure with dark tousled shoulder-length "
        "hair, dressed in thick black furs and layered leather armor. A "
        "sword often at the hip, stoic and burdened expression. Backdrop is "
        "a snow-covered northern landscape, a massive icy wall, or a grey "
        "overcast sky. Mood is honorable, quietly resolute, world-weary."
    ),
    "Shinchan": (
        "A goofy, exaggerated cartoon kid with spiky messy black hair and "
        "a wide, mischievous grin, dressed in a simple bright shirt and "
        "shorts. Pose is playful and over-the-top -- eyebrows raised, "
        "cheeky wink, hands on hips. Backdrop is bold, flat, colorful "
        "cartoon-style scenery. Mood is comedic, carefree, unapologetically "
        "silly."
    ),
    "Harry Potter": (
        "A boyish, earnest figure with round glasses, a lightning-bolt scar "
        "on the forehead, and messy black hair. Gryffindor robes with a "
        "red-and-gold scarf, often holding a wand or a broomstick. "
        "Expression is determined but still a little wide-eyed. Backdrop is "
        "a Hogwarts castle silhouette or a starry night sky over a "
        "quidditch pitch. Mood is brave, hopeful, quietly heroic."
    ),
    "Dr. Doom": (
        "An imposing figure in full green-and-metal armor with a long, "
        "flowing green hooded cloak. A smooth, expressionless metal mask, "
        "arms often crossed or raised with commanding, regal authority. "
        "Backdrop is a gothic stone castle or crackling green arcane "
        "energy. Mood is imperious, coldly intellectual, quietly menacing."
    ),
    "Regina George": (
        "A poised, sharply styled figure with sleek blonde hair and a "
        "trendy, put-together pink or designer outfit. Confident smirk, "
        "hand on hip, a look that reads as effortless social power. "
        "Backdrop is a glossy high-school hallway or a soft pastel-pink "
        "gradient. Mood is icy confidence, queen-bee charisma, sharp wit."
    ),
    "Elle Woods": (
        "A bright, confident figure with voluminous blonde hair and a "
        "bold pink outfit -- blazer, dress, or matching accessories. Wide, "
        "genuine smile, often holding a legal pad or striking a poised, "
        "self-assured pose. Backdrop is a sunny pastel campus or a "
        "courtroom. Mood is bubbly, sharp, underestimated but brilliant."
    ),
    "Sheldon Cooper": (
        "A neat, precise figure with side-swept hair and a layered graphic "
        "tee under a collared shirt. Posture is prim and exact, arms often "
        "crossed, expression deadpan and faintly superior. Backdrop is a "
        "cluttered apartment with a whiteboard full of equations or a "
        "comic-book-lined room. Mood is hyper-logical, socially oblivious, "
        "quietly self-satisfied."
    ),
}

CHARACTERS = list(CHARACTER_STYLE_GUIDE.keys())


# --------------------------------------------------------------------------
# Optional reference-image grounding, via Gemini context caching.
#
# static/characters/<slug>.jpg already exists for the frontend showcase
# (app.py serves it at /static/characters/). This section reuses the same
# folder: if a character has a matching image file, that image + its style
# guide text get baked into a Gemini cache once, and every analyze_and_match
# call afterward references the cache instead of re-uploading/re-encoding
# those images on every single request.
#
# This is entirely additive and safe by default: with zero images present
# (the out-of-the-box state), _load_reference_images() returns {}, no cache
# is built, and analyze_and_match behaves exactly as before -- full text
# roster inline, no images sent to Gemini at all.
#
# Whoever supplies these images is responsible for having the right to use
# and display them -- this app doesn't source or vet them.
# --------------------------------------------------------------------------

CHARACTER_REF_DIR = os.environ.get("CHARACTER_REF_DIR", "static/characters")

# Gemini's caching API only accepts a duration, not "never expires" -- there's
# no true unlimited option. 24h comfortably covers a full event day; at this
# roster's content size (21 images + descriptions), storage cost is trivial
# even at multi-day TTLs if you want to push it further. Whatever TTL you
# pick, analyze_and_match() below self-heals if the cache expires mid-run
# (drops it, retries once without it, next call rebuilds fresh) -- so this
# number is about convenience, not a failure risk either way.
CACHE_TTL = os.environ.get("GEMINI_CACHE_TTL", "86400s")

_cache_name: str | None = None


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _load_reference_images() -> dict:
    """Returns {character_name: (image_bytes, mime_type)} for whichever
    roster members currently have a matching file in CHARACTER_REF_DIR.
    Characters without a file are simply skipped."""
    found = {}
    ext_mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}
    for name in CHARACTERS:
        slug = _slugify(name)
        for ext, mime in ext_mime.items():
            path = os.path.join(CHARACTER_REF_DIR, f"{slug}.{ext}")
            if os.path.exists(path):
                with open(path, "rb") as f:
                    found[name] = (f.read(), mime)
                break
    return found


def _build_cache_contents(images: dict) -> list:
    parts = [
        "Reference set for a college-fest photo booth character roster. "
        "Each character below is given as a labeled reference image plus a "
        "written style anchor -- use both together as grounding for what "
        "that character should look like when matching a photo later."
    ]
    for name, (img_bytes, mime) in images.items():
        parts.append(f"Reference for {name}: {CHARACTER_STYLE_GUIDE[name]}")
        parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))
    return parts


def _ensure_character_cache(force: bool = False):
    """Lazily builds a Gemini context cache from whatever reference images
    currently exist. Safe to call on every request -- a no-op after the
    first successful (or failed) attempt unless force=True.

    Falls back to no cache (None) if there are no reference images yet, if
    this model/SDK version doesn't support caching, or if the cached content
    is too small to meet Gemini's caching minimum -- analyze_and_match then
    just uses the full inline text roster like before, so a caching failure
    can never break a live capture.
    """
    global _cache_name
    if _cache_name is not None and not force:
        return _cache_name

    images = _load_reference_images()
    if not images:
        _cache_name = None
        return None

    try:
        cache = _client.caches.create(
            model=MODEL_ID,
            config=types.CreateCachedContentConfig(
                contents=_build_cache_contents(images),
                ttl=CACHE_TTL,
            ),
        )
        _cache_name = cache.name
        print(f"[gemini_analysis] character cache built with {len(images)} "
              f"reference image(s): {_cache_name}")
    except Exception as e:
        print(f"[gemini_analysis] character cache build failed ({e}); "
              f"falling back to inline text roster, no images sent.")
        _cache_name = None

    return _cache_name


def rebuild_character_cache():
    """Force a fresh cache build -- call after dropping new images into
    CHARACTER_REF_DIR without restarting the server (see the
    /booth/admin/rebuild-cache endpoint in app.py)."""
    return _ensure_character_cache(force=True)


def list_characters_with_images() -> list:
    """Public helper for the admin endpoint: which roster characters
    currently have a matching reference image on disk."""
    return list(_load_reference_images().keys())


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


def _build_instruction(cached: bool = False) -> str:
    if cached:
        # Character roster + reference images already live in the cached
        # content prepended to this request -- no need to repeat them here.
        roster_section = (
            "A labeled reference image plus a written style anchor for each "
            "character on the roster is already provided above in this "
            "context. Use those references directly. The full roster you "
            "must choose from (exact names, do not invent others) is:\n"
            + "\n".join(f"- {name}" for name in CHARACTERS)
        )
    else:
        roster_lines = "\n".join(
            f'- {name}: {style}' for name, style in CHARACTER_STYLE_GUIDE.items()
        )
        roster_section = (
            "Pick from this fixed list only (do not invent a character not "
            f"on this list):\n{roster_lines}"
        )

    return f"""You are the analysis brain for a college-fest photo booth.
Look at the attached photo of a person and do all of the following in one pass:

1. Pick the single best-matching character for this person.
{roster_section}

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
   one flowing sentence, no filler adjectives repeated for effect.

   The single most important rule: this is a costume/backdrop swap on the
   SAME person, not a redraw of a different person. The model must keep the
   subject's actual face (identity, facial features, skin tone, expression)
   and their exact pose/framing/camera angle from the photo completely
   unchanged -- only the outfit, props, and background change. Open the
   prompt by stating this explicitly (e.g. "Keep the exact same person, face,
   and pose from the photo unchanged;") before describing anything else.

   Then pack in only concrete, high-signal visual anchors: (a) 2-4 specific
   *costume/prop* details pulled from the chosen character's style cues
   above -- clothing colors, accessories, iconic props -- but never hair,
   facial structure, or anything that would alter the person's actual face,
   (b) one lighting/backdrop cue, (c) "cinematic movie-poster quality,
   college fest promotional poster style" as a closing style tag.
   A short, dense prompt lets the image model converge faster with the same
   fidelity -- do not pad it with restated synonyms.

5. Fill "detected_traits": your own quick read of the photo --
   - outfit_color: the single dominant clothing color you see, one word
     (e.g. "red", "blue", "black", "gray")
   - pose_description: a short phrase describing their pose/stance
     (e.g. "arms crossed", "peace sign", "neutral standing pose")
   - glasses: true or false, whether they're wearing glasses/sunglasses

Respond with JSON matching the required schema only."""


def _call_gemini(image_bytes: bytes, mime_type: str, cache_name):
    config_kwargs = dict(
        response_mime_type="application/json",
        # CRITICAL FIX: We removed response_schema=BoothMatch to prevent
        # the SDK from crashing internally while trying to parse the output.
        temperature=0.8,
    )
    if cache_name:
        config_kwargs["cached_content"] = cache_name

    return _client.models.generate_content(
        model=MODEL_ID,
        contents=[
            _build_instruction(cached=bool(cache_name)),
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        ],
        config=types.GenerateContentConfig(**config_kwargs),
    )


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
    global _cache_name
    cache_name = _ensure_character_cache()

    try:
        response = _call_gemini(image_bytes, mime_type, cache_name)
    except Exception as e:
        if not cache_name:
            raise  # nothing cache-related to blame, let the real error surface
        # The cache may have expired or become invalid mid-event (TTL ran
        # out, was deleted server-side, etc.). Drop it so the next request
        # rebuilds fresh via _ensure_character_cache(), and retry THIS
        # request once without it so the current visitor's capture still
        # succeeds instead of failing the whole job.
        print(f"[gemini_analysis] cached request failed ({e}); dropping cache "
              f"and retrying without it")
        _cache_name = None
        response = _call_gemini(image_bytes, mime_type, None)

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