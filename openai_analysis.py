"""
Stage 2: analysis + character match + prompt construction, collapsed into a
single multimodal call to OpenAI (moved from Gemini).

Why this moved from Gemini to OpenAI: the booth needs conversational context
to carry from this analysis call into stage 3's poster generation call, and
that's only available on the paid OpenAI account currently in use -- not on
the Gemini API key currently in use. Sending this first call through OpenAI
instead keeps everything on one provider/account so that context can be
threaded through later.

New pipeline (this file is the whole of stage 2): send the captured photo
straight to OpenAI with one prompt that asks it to (a) look at the person,
(b) pick the best-matching character from a fixed list, (c) write a caption,
and (d) write a ready-to-use image-editing prompt -- all in one structured
JSON response. No local model, no local heuristics.

Model: gpt-4o by default (override via OPENAI_ANALYSIS_MODEL env var) -- a
vision-capable chat model, distinct from OPENAI_IMAGE_MODEL in image_gen.py
which is the image-generation model used for stage 3.

SDK: `openai` (the same SDK/client already used by image_gen.py).
"""

import os
import re
import json
import base64
from openai import OpenAI
from pydantic import BaseModel
import image_utils

MODEL_ID = os.environ.get("OPENAI_ANALYSIS_MODEL", "gpt-4o")

# Shared with image_gen.py's OPENAI_INPUT_MAX_DIM so both calls downscale to
# the same size -- the captured photo was previously sent full-resolution
# to this analysis call (only the poster-edit call downscaled), so a large
# phone photo was effectively being uploaded twice at full size for no
# benefit; gpt-4o downsamples large images internally anyway.
ANALYSIS_MAX_DIM = int(os.environ.get("OPENAI_INPUT_MAX_DIM", "1280"))

_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Fixed roster the model must choose from. Each entry carries a longer,
# hand-written *text* style anchor -- unchanged from the Gemini version.
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
        "heroic and self-assured, chin slightly raised. Reads as a man with "
        "a neat, trimmed goatee when unmasked; the vibe is witty, quippy "
        "tech-genius bravado, not brooding or withdrawn."
    ),
    "Batman": (
        "A brooding, powerfully built vigilante in a matte-black armored "
        "bat-suit with a flowing cape and a cowl with pointed ears framing a "
        "stern, unsmiling jaw. Utility belt with gadget pouches, muscular "
        "silhouette, gauntlets with blade-like fins. Shot in moody "
        "Gotham-noir lighting -- deep shadows, cold blue-grey highlights, "
        "rain-slicked rooftops or a dark alley backdrop. Expression is "
        "intense, guarded, rarely smiling. Reads as a man, jaw usually "
        "clean-shaven or with only light stubble beneath the cowl; the vibe "
        "is grim, controlled, and guarded -- never playful or loud."
    ),
    "Spider-Man": (
        "An athletic, lithe figure in a skin-tight red-and-blue spandex "
        "suit with bold black web patterning and large white reflective eye "
        "lenses. Pose is dynamic and acrobatic -- mid-swing, crouched on a "
        "ledge, or a classic web-shooting hand gesture. Backdrop is a dense "
        "city skyline, skyscrapers and web-lines cutting across a dusk sky. "
        "Overall feel is energetic, youthful, and playful. The full mask "
        "conceals the entire face, including any facial hair, so a match "
        "here should lean on youthful build, dynamic pose, and playful "
        "energy rather than facial resemblance, which isn't visible "
        "in-costume."
    ),
    "Thomas Shelby": (
        "A sharply dressed, composed figure in a fitted 1920s tweed "
        "three-piece suit with a flat cap pulled low, hair slicked back and "
        "undercut at the sides. Expression is unreadable, jaw set, eyes "
        "narrowed with quiet menace, often holding or about to light a "
        "cigarette. Backdrop is muted sepia-toned Birmingham streets, "
        "factory smoke, or a dim pub interior. Mood is restrained power, "
        "cold calculation, old-world formality. Reads as a man, usually "
        "clean-shaven or close-cropped stubble, never a full beard; the "
        "vibe is cold and controlled, the opposite of warm or playful."
    ),
    "Naruto": (
        "An energetic, wiry young ninja in an orange-and-black jumpsuit, a "
        "blue forehead protector with a metal leaf-village emblem, and "
        "spiky sun-bleached blonde hair. Whisker-like markings on each "
        "cheek, wide grinning expression, often mid-jump or making a "
        "hand-sign with one fist raised. Backdrop is a dynamic anime-style "
        "burst of energy, orange and blue chakra swirls, motion lines. Mood "
        "is upbeat, determined, full of youthful energy. Reads as a young "
        "man, no real facial hair (only the whisker markings); the vibe is "
        "loud, warm, unguarded energy -- the opposite of brooding or aloof."
    ),
    "Professor from Money Heist": (
        "A calm, calculating mastermind -- disheveled academic look with "
        "thick-rimmed glasses, tousled hair, and a rumpled cardigan when "
        "planning, contrasted with the crew's signature red jumpsuit and a "
        "white Salvador Dali mask (wide painted grin, rosy cheeks, thin "
        "mustache) when in heist mode. Composed, quietly intense gaze that "
        "reads as always three steps ahead. Backdrop is a bank vault stacked "
        "with gold bars or a sunbaked Spanish plaza. Mood is restrained, "
        "cerebral, unshakeable control. Reads as a man; the disguise mask "
        "has a thin mustache, but his own unmasked face is clean-shaven "
        "with a tired, scholarly look. Vibe is calm and cerebral, never "
        "loud or theatrical."
    ),
    "Gwen Stacy": (
        "An athletic, confident figure in a pink-and-white hooded Spider "
        "suit with a bold black spider emblem, hood down to show "
        "blonde-and-pink-streaked hair and a smirking, self-assured "
        "expression. Pose is dynamic -- perched on a ledge, mid-leap, or "
        "leaning back casually. Backdrop is a comic-book-style graffiti "
        "splash of pink and blue against a city skyline. Mood is cool, "
        "playful, quietly rebellious. Reads as a woman, no facial hair; "
        "the vibe is relaxed, cool confidence rather than fierce or icy."
    ),
    "Black Widow": (
        "A lean, athletic figure in a fitted black tactical bodysuit with "
        "subtle utility straps and holsters, in a confident, combat-ready "
        "stance -- often crouched low or mid-motion. Wavy red hair, sharp "
        "focused eyes, minimal but striking makeup, a calm and controlled "
        "expression that reads as dangerous competence rather than "
        "aggression. Backdrop is cool steel-blue or shadowed tactical "
        "environments -- a dim corridor, rain, or muted urban night. Mood "
        "is composed, precise, quietly formidable. Reads as a woman, no "
        "facial hair; the vibe is composed and controlled, not overtly "
        "emotional, warm, or playful."
    ),
    "Daenerys Targaryen": (
        "A regal, striking figure with long silver-platinum braided hair and "
        "a flowing pale ivory or deep-blue gown with intricate detailing. "
        "Expression is composed but fierce -- calm authority with the "
        "implication of real power underneath. Subtle dragon-scale texture "
        "or drifting smoke/embers in the frame. Backdrop is a sunbaked "
        "desert city, a grand throne hall, or a dragon's silhouette against "
        "the sky. Mood is commanding, regal, quietly dangerous. Reads as a "
        "woman, no facial hair; the vibe is regal and commanding, cool "
        "rather than warm or bubbly."
    ),
    "Hermione Granger": (
        "A sharp, studious figure with bushy brown hair and Gryffindor "
        "school robes -- black robe, maroon-and-gold tie, house crest. "
        "Often shown mid-gesture with a wand raised, holding a stack of "
        "books, or with a focused, thinking expression. Backdrop is a "
        "candlelit castle library or stone corridor. Mood is intelligent, "
        "quietly confident, prepared for anything. Reads as a woman, no "
        "facial hair; the vibe is studious and quietly confident, not "
        "flashy or loud."
    ),
    "Wednesday Addams": (
        "A pale, deadpan figure with jet-black hair in two tight braids, "
        "dressed in a black Victorian-collared dress or a dark academy "
        "uniform. Expression is flat, unimpressed, and unblinking -- no "
        "smile, dry composure. Backdrop is a foggy graveyard, a gothic "
        "academy hallway, or muted grey-and-black tones throughout. Mood is "
        "dark, dry-witted, completely unbothered. Reads as a woman/girl, no "
        "facial hair; the vibe is flat and deadpan, the direct opposite of "
        "warm, bubbly, or expressive."
    ),
    "Kakashi Hatake": (
        "A relaxed but sharp-eyed ninja with messy silver hair and a Leaf "
        "Village forehead protector tilted to cover one eye. Navy-blue "
        "flak vest over a dark long-sleeve uniform, a fabric mask covering "
        "the nose and mouth, one hand often in his pocket or holding an "
        "orange book. Backdrop is a misty forest or a village rooftop at "
        "dusk. Mood is cool, effortlessly skilled, quietly amused. The "
        "fabric mask covers the lower half of the face, so facial hair "
        "isn't visible in-costume; reads as a man. Vibe is relaxed and "
        "effortlessly cool, quietly amused rather than intense or brooding."
    ),
    "Sasuke Uchiha": (
        "A dark-haired, sharp-featured young ninja with a brooding, intense "
        "stare -- red-and-black Sharingan eye detail implied. Navy-and-white "
        "high-collared ninja outfit, a katana slung across the back. Pose "
        "is still and coiled, ready to strike. Backdrop is crackling blue "
        "lightning energy or a dark storm-lit battlefield. Mood is aloof, "
        "intense, quietly powerful. Reads as a young man, no facial hair; "
        "the vibe is aloof and controlled, never warm or goofy."
    ),
    "Joker": (
        "A lean, theatrical figure in a rumpled purple tailcoat suit over a "
        "green vest, with slicked-back or unkempt green-dyed hair. "
        "Chalk-white face paint, a smeared red smile carved wide across the "
        "mouth, dark heavy eye makeup. Pose is loose and expressive -- a "
        "wide grin, jazz hands, or leaning in menacingly. Backdrop is "
        "chaotic neon-lit city grime, graffiti, or flickering carnival "
        "light. Reads as a man under heavy face paint, no real facial hair "
        "emphasized; the vibe is loud, unstable, theatrical energy -- the "
        "opposite of calm or composed."
    ),
    "Jon Snow": (
        "A brooding, weathered figure with dark tousled shoulder-length "
        "hair, dressed in thick black furs and layered leather armor. A "
        "sword often at the hip, stoic and burdened expression. Backdrop is "
        "a snow-covered northern landscape, a massive icy wall, or a grey "
        "overcast sky. Mood is honorable, quietly resolute, world-weary. "
        "Reads as a man, usually with visible stubble or a light-to-medium "
        "beard; the vibe is weary and honorable, serious rather than "
        "playful or loud."
    ),
    "Shinchan": (
        "A goofy, exaggerated cartoon kid with spiky messy black hair and "
        "a wide, mischievous grin, dressed in a simple bright shirt and "
        "shorts. Pose is playful and over-the-top -- eyebrows raised, "
        "cheeky wink, hands on hips. Backdrop is bold, flat, colorful "
        "cartoon-style scenery. Mood is comedic, carefree, unapologetically "
        "silly. Reads as a young boy, no facial hair; the vibe is loud, "
        "silly, and over-the-top -- the most purely comedic character on "
        "this roster."
    ),
    "Harry Potter": (
        "A boyish, earnest figure with round glasses, a lightning-bolt scar "
        "on the forehead, and messy black hair. Gryffindor robes with a "
        "red-and-gold scarf, often holding a wand or a broomstick. "
        "Expression is determined but still a little wide-eyed. Backdrop is "
        "a Hogwarts castle silhouette or a starry night sky over a "
        "quidditch pitch. Mood is brave, hopeful, quietly heroic. Reads as "
        "a young man, no facial hair; the vibe is earnest and a little "
        "wide-eyed, not jaded, cold, or aloof."
    ),
    "Dr. Doom": (
        "An imposing figure in full green-and-metal armor with a long, "
        "flowing green hooded cloak. A smooth, expressionless metal mask, "
        "arms often crossed or raised with commanding, regal authority. "
        "Backdrop is a gothic stone castle or crackling green arcane "
        "energy. Mood is imperious, coldly intellectual, quietly menacing. "
        "The full mask conceals the entire face, so no facial features "
        "(including facial hair) are visible in-costume -- a match here "
        "should lean on commanding posture and imperious vibe, not facial "
        "resemblance."
    ),
    "Regina George": (
        "A poised, sharply styled figure with sleek blonde hair and a "
        "trendy, put-together pink or designer outfit. Confident smirk, "
        "hand on hip, a look that reads as effortless social power. "
        "Backdrop is a glossy high-school hallway or a soft pastel-pink "
        "gradient. Mood is icy confidence, queen-bee charisma, sharp wit. "
        "Reads as a woman, no facial hair; the vibe is icy and socially "
        "sharp, not warm or approachable."
    ),
    "Elle Woods": (
        "A bright, confident figure with voluminous blonde hair and a "
        "bold pink outfit -- blazer, dress, or matching accessories. Wide, "
        "genuine smile, often holding a legal pad or striking a poised, "
        "self-assured pose. Backdrop is a sunny pastel campus or a "
        "courtroom. Mood is bubbly, sharp, underestimated but brilliant. "
        "Reads as a woman, no facial hair; the vibe is warm, bright, and "
        "openly enthusiastic -- the opposite of icy or deadpan."
    ),
    "Sheldon Cooper": (
        "A neat, precise figure with side-swept hair and a layered graphic "
        "tee under a collared shirt. Posture is prim and exact, arms often "
        "crossed, expression deadpan and faintly superior. Backdrop is a "
        "cluttered apartment with a whiteboard full of equations or a "
        "comic-book-lined room. Mood is hyper-logical, socially oblivious, "
        "quietly self-satisfied. Reads as a man, clean-shaven; the vibe is "
        "rigid, logical, and mildly condescending, not scruffy or rugged."
    ),
}

CHARACTERS = list(CHARACTER_STYLE_GUIDE.keys())


# --------------------------------------------------------------------------
# Reference-image grounding via OpenAI's automatic prompt caching.
#
# OpenAI has no Gemini-style explicit "create a cache, get a cache_name back"
# API. Instead, OpenAI automatically caches the *prefix* of a request (the
# leading, identical portion of the message content) whenever that exact
# prefix was recently processed -- no code required to opt in, but it only
# pays off if the static part of the prompt (the roster instructions +
# reference images) is:
#   1. always sent in the exact same byte-for-byte form, in the same order,
#   2. placed BEFORE the one thing that changes per request (this student's
#      captured photo),
#   3. large enough to matter (OpenAI's minimum cacheable prefix is 1024
#      tokens; a roster of ~20 images comfortably clears that).
# Cache entries are held for roughly 5-10 minutes of inactivity (up to a
# 1-hour hard cap) -- there's no "24 hour TTL" knob like Gemini's. In
# practice, at event volume (one capture every few seconds/minutes), each
# request keeps the cache warm for the next one, so this still saves real
# cost/latency across a session even without an explicit build-once step.
#
# static/characters/<slug>.{jpg,jpeg,png,webp} already exists for the
# frontend showcase (app.py serves it at /static/characters/). This section
# reuses the same folder and the same slug convention as the old Gemini
# caching code: if a character has a matching image file on disk, it's
# attached as an extra reference image in the roster block. Characters
# without a file are simply described in text only, same as before.
#
# Whoever supplies these images is responsible for having the right to use
# and display them -- this app doesn't source or vet them.
# --------------------------------------------------------------------------

CHARACTER_REF_DIR = os.environ.get("CHARACTER_REF_DIR", "static/characters")


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _load_reference_images() -> dict:
    """Returns {character_name: (image_bytes, mime_type)} for whichever
    roster members currently have a matching file in CHARACTER_REF_DIR.
    Characters without a file are simply skipped. Sorted by character name
    on the way out (see _build_roster_content) so the byte order is
    deterministic across requests/restarts -- required for the prefix to
    match and actually hit OpenAI's cache."""
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


# Loaded once at import time (module-level, not per-request) so the bytes
# and their order are identical across every call in this process -- any
# difference here (re-reading the file, re-ordering the dict) would change
# the prefix and silently break cache hits even though the request still
# succeeds.
_REFERENCE_IMAGES = _load_reference_images()


def _build_roster_content() -> list:
    """Static, request-invariant content block: roster instructions plus
    whatever reference images exist on disk, each labeled with the
    character it belongs to. This must be assembled the exact same way on
    every call (same images, same order, same text) and placed BEFORE the
    per-request photo in the message -- that's what makes it a stable,
    cacheable prefix instead of a one-off payload OpenAI has to reprocess
    in full on every single capture."""
    parts = [{"type": "text", "text": _build_instruction()}]

    if _REFERENCE_IMAGES:
        parts.append({
            "type": "text",
            "text": (
                "Reference images for some roster characters follow below, "
                "each preceded by a label naming which character it shows. "
                "Use these as additional visual grounding alongside the "
                "text style anchors above."
            ),
        })
        # Sorted for determinism -- CHARACTERS order is already fixed by
        # dict insertion order in CHARACTER_STYLE_GUIDE, but iterate it
        # explicitly rather than the raw dict to guarantee stability.
        for name in CHARACTERS:
            if name not in _REFERENCE_IMAGES:
                continue
            img_bytes, mime = _REFERENCE_IMAGES[name]
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            parts.append({"type": "text", "text": f"Reference image for {name}:"})
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })

    return parts


def rebuild_character_cache():
    """Re-reads static/characters/ from disk and rebuilds the in-process
    reference-image set used to build the cacheable prefix (see
    _build_roster_content). There's no server-side cache "handle" to return
    the way Gemini's caches.create() had -- OpenAI's prompt caching is
    automatic and keyed off matching request content, not an explicit
    object -- so this just refreshes what images get sent, and the next
    request establishes a fresh cached prefix on OpenAI's side."""
    global _REFERENCE_IMAGES
    _REFERENCE_IMAGES = _load_reference_images()
    return f"{len(_REFERENCE_IMAGES)} reference image(s) loaded" if _REFERENCE_IMAGES else None


def list_characters_with_images() -> list:
    """Public helper for the admin endpoint: which roster characters
    currently have a matching reference image on disk."""
    return list(_REFERENCE_IMAGES.keys())


class DetectedTraits(BaseModel):
    outfit_color: str
    pose_description: str
    glasses: bool
    standout_trait: str


class BoothMatch(BaseModel):
    character: str
    reasoning: str
    caption: str
    diffusion_prompt: str
    detected_traits: DetectedTraits


def _strict_schema(model: type[BaseModel]) -> dict:
    """OpenAI's strict structured-output mode requires every object node in
    the JSON schema -- including nested models under $defs -- to explicitly
    set additionalProperties: false and list every property as required.
    Pydantic's model_json_schema() doesn't set this by default, so without
    this walk OpenAI rejects the request with a 400 'additionalProperties
    is required to be supplied and to be false' error before the model ever
    runs."""
    schema = model.model_json_schema()

    def _tighten(node):
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                node["additionalProperties"] = False
                if "properties" in node:
                    node["required"] = list(node["properties"].keys())
            for value in node.values():
                _tighten(value)
        elif isinstance(node, list):
            for item in node:
                _tighten(item)

    _tighten(schema)
    return schema


def _build_instruction() -> str:
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

Weigh ALL of these dimensions together when deciding, with NO fixed priority
order between them: facial resemblance/features, general vibe or energy,
complexion, gender presentation, facial hair (or the lack of it), pose, type
of clothing, and clothing color. Do not default to the same one or two
dimensions on every photo (e.g. always leaning on outfit color or glasses
just because they're easy to see) -- look at THIS specific photo and decide
which single dimension is the strongest, most distinctive match to a
character on the roster, and let that dimension drive the pick. Different
photos should land on different characters for different reasons: if
someone's clothing is generic but their expression and vibe strongly
resemble a specific character, match on that instead of forcing an outfit
match. The goal is a varied, personalized set of matches across many
people -- avoid converging on the same handful of "safe" characters just
because their descriptions mention easy-to-spot items like glasses or a
graphic tee; check the other dimensions (especially facial hair and gender
presentation) before committing to a match like that.

2. Write "reasoning": one or two short sentences, written directly to the
   person ("you"), explaining *why* they were matched to this character --
   name the specific standout dimension from step 1 that actually drove the
   match, grounded in specific things you actually saw (e.g. "Your steady,
   deadpan expression and clean-shaven, buttoned-up look read exactly like
   Sheldon" or "Your loud energy and big grin are pure Shinchan, no matter
   what you're wearing"). Concrete and specific, not generic -- and don't
   default to describing their outfit if a different trait was actually the
   deciding factor.

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
   (b) one lighting/backdrop cue that evokes that character's own film or
   franchise (e.g. Gotham-noir shadows for Batman, a Hogwarts corridor for
   Harry Potter, chakra-burst energy for Naruto) -- the backdrop should feel
   like it belongs to their world, not a generic photo-booth setting,
   (c) "cinematic movie-poster quality" as a closing style tag. Never
   mention a college, fest, campus, event, or any promotional/booth
   branding anywhere in this prompt -- the poster should read as a genuine
   still from that character's film, not an event souvenir.
   A short, dense prompt lets the image model converge faster with the same
   fidelity -- do not pad it with restated synonyms.

5. Fill "detected_traits": your own quick read of the photo --
   - outfit_color: the single dominant clothing color you see, one word
     (e.g. "red", "blue", "black", "gray")
   - pose_description: a short phrase describing their pose/stance
     (e.g. "arms crossed", "peace sign", "neutral standing pose")
   - glasses: true or false, whether they're wearing glasses/sunglasses
   - standout_trait: which single dimension from step 1 actually decided
     this match, as one short phrase (e.g. "facial hair", "vibe/energy",
     "gender presentation", "clothing color", "pose") -- this should match
     whatever you named in "reasoning", and should not be "clothing color"
     or "clothing type" every single time.

Respond with JSON matching the required schema only."""


def _call_openai(image_bytes: bytes, mime_type: str, include_reference_images: bool = True):
    # Downscale before sending -- previously this call got the full-res
    # original while only the poster-edit call downscaled; a large phone
    # photo was effectively being uploaded twice at full size for no
    # accuracy benefit. Re-encoded as JPEG by prepare_image_bytes, so the
    # mime type sent to OpenAI is always image/jpeg regardless of the
    # original upload's mime_type.
    small_bytes = image_utils.prepare_image_bytes(image_bytes, max_dim=ANALYSIS_MAX_DIM)
    b64_image = base64.b64encode(small_bytes).decode("utf-8")

    # Static roster block (instructions + reference images) goes first --
    # identical across every request, so it's the part OpenAI's automatic
    # prompt caching can match and reuse. The captured photo is the only
    # thing that changes per student, so it goes last, after the stable
    # prefix, instead of being interleaved with it. include_reference_images
    # is only False on a fallback retry (see analyze_and_match) -- if the
    # roster+images block itself is somehow the cause of a failure, a
    # smaller text-only retry gives the request a real chance to succeed
    # instead of hitting the exact same error twice.
    content = (_build_roster_content() if include_reference_images else [{"type": "text", "text": _build_instruction()}]) + [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
        },
    ]

    return _client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": content}],
        # Stable key (not per-request) so OpenAI is more likely to route
        # repeated calls to the same cache-holding backend -- a routing
        # hint, not a cache handle; it doesn't guarantee a hit by itself.
        prompt_cache_key="movie-booth-character-roster",
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "booth_match",
                "schema": _strict_schema(BoothMatch),
                "strict": True,
            },
        },
        temperature=0.8,
    )


def _log_cache_usage(response) -> None:
    """Debug: surface how much of this request's prompt was served from
    OpenAI's automatic cache. Chat Completions reports this under
    usage.prompt_tokens_details.cached_tokens -- 0 means no cache hit (first
    request after a cold start, prefix changed, or the >~10min idle window
    expired), any nonzero value means the roster prefix (instructions +
    reference images) was reused instead of reprocessed."""
    try:
        usage = response.usage
        cached = usage.prompt_tokens_details.cached_tokens
        total_prompt = usage.prompt_tokens
        print(f"[openai_analysis] prompt cache: {cached}/{total_prompt} "
              f"prompt tokens served from cache"
              + (" (cache HIT)" if cached > 0 else " (cache MISS)"))
    except Exception as e:
        print(f"[openai_analysis] could not read cache usage info: {e}")


def analyze_and_match(image_bytes: bytes, mime_type: str = "image/jpeg") -> dict:
    """
    Stage 2 entry point. Takes the captured photo's raw bytes, returns:
    {
      "character": "...",
      "caption": "...",
      "diffusion_prompt": "...",
      "detected_traits": {"outfit_color": "...", "pose_description": "...", "glasses": bool}
    }

    Falls back once on failure: if the full request (roster instructions +
    reference images + photo) raises -- a transient API error, or possibly
    something in the reference-image block itself -- retry once with the
    text-only roster (no reference images) instead of failing the capture
    outright. Both the failure and the fallback are printed so they're
    visible in the terminal in real time, not just swallowed.
    """
    try:
        response = _call_openai(image_bytes, mime_type)
    except Exception as e:
        print(f"[openai_analysis] full request (with reference images) failed "
              f"({e}); retrying once with text-only roster, no images")
        try:
            response = _call_openai(image_bytes, mime_type, include_reference_images=False)
        except Exception as e2:
            print(f"[openai_analysis] text-only retry also failed ({e2}); "
                  f"giving up on this capture")
            raise

    _log_cache_usage(response)

    raw_text = response.choices[0].message.content
    result = json.loads(raw_text)

    # Clamp off-roster results
    if result.get("character") not in CHARACTERS:
        result["character"] = CHARACTERS[0]

    # Defensive default in case the model ever omits the field
    result.setdefault("reasoning", "")

    return result