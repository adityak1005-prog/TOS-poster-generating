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

Model: gpt-5-mini by default (override via OPENAI_ANALYSIS_MODEL env var) --
OpenAI's cheapest vision-capable chat model as of mid-2026, chosen for
latency: this call was taking ~7s on gpt-4o, and gpt-5-mini is meaningfully
faster/cheaper while still vision-capable and still covered by OpenAI's
automatic prompt caching (all GPT-5-series models support it). Swap back to
gpt-4o via the env var if match/reasoning quality ever seems to suffer --
gpt-5-mini is a real speed/cost trade, not a strict upgrade.

SDK: `openai` (the same SDK/client already used by image_gen.py).
"""

import os
import re
import json
import base64
from openai import OpenAI
from pydantic import BaseModel
import image_utils

MODEL_ID = os.environ.get("OPENAI_ANALYSIS_MODEL", "gpt-5-mini")

# gpt-5-mini is a reasoning model -- it spends time on internal "thinking"
# tokens before producing the visible JSON answer, and that thinking time is
# what was actually driving the 35+s analysis latency reported live (not the
# vision/JSON-generation work itself). reasoning_effort controls how much of
# that invisible thinking it does: minimal/low/medium/high. This task is a
# single-pass classification + short text generation against a fixed
# schema -- it doesn't need multi-step deliberation, so "minimal" is the
# right default. Override via OPENAI_REASONING_EFFORT if match quality ever
# seems to need more deliberation.
REASONING_EFFORT = os.environ.get("OPENAI_REASONING_EFFORT", "minimal")

# Shared with image_gen.py's OPENAI_INPUT_MAX_DIM so both calls downscale to
# the same size -- the captured photo was previously sent full-resolution
# to this analysis call (only the poster-edit call downscaled), so a large
# phone photo was effectively being uploaded twice at full size for no
# benefit; gpt-4o downsamples large images internally anyway.
ANALYSIS_MAX_DIM = int(os.environ.get("OPENAI_INPUT_MAX_DIM", "1280"))

# Much smaller than ANALYSIS_MAX_DIM on purpose. The 21 character reference
# images are sent inline on EVERY analysis call whenever OpenAI's automatic
# cache doesn't hit (the cache window is only ~5-10min, so at a booth with
# gaps between visitors most calls miss it) -- they were previously read
# straight off disk at whatever resolution the source file happened to be,
# which for 21 images stacked together is very likely the single biggest
# contributor to slow, non-cached analysis calls (much bigger than the
# reasoning-effort overhead). Fine detail isn't needed here, these just need
# to be recognizable for visual grounding, so downscale hard.
REFERENCE_IMAGE_MAX_DIM = int(os.environ.get("OPENAI_REFERENCE_IMAGE_MAX_DIM", "400"))

# OpenAI's automatic cache normally evicts a prefix after ~5-10min idle and
# guarantees eviction within 1hr regardless. This opt-in flag asks OpenAI to
# hold onto it for up to 24hrs instead -- documented for the GPT-5.1/5.4/5.5
# family, NOT confirmed for plain gpt-5-mini. Set to "" to disable trying it
# outright. If the API rejects it, _extended_retention_supported below flips
# to False after the first failure and every request after that stops
# sending it -- watch the terminal for a line starting
# "[openai_analysis] EXTENDED CACHE RETENTION NOT SUPPORTED" to know for
# certain whether this is actually doing anything on your account/model.
EXTENDED_CACHE_RETENTION = os.environ.get("OPENAI_EXTENDED_CACHE_RETENTION", "24h")
_extended_retention_supported = True

_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Fixed roster the model must choose from. Each entry carries a longer,
# hand-written *text* style anchor -- unchanged from the Gemini version.
#
# Descriptions are intentionally long and specific (build, outfit/colors,
# expression, iconic prop, backdrop/lighting) -- more concrete visual anchors
# here means a more accurate match against the photo.
# Each entry below is split into two clearly labeled halves with very
# different jobs:
#
# MATCH ON -- traits an ORDINARY phone photo could actually contain:
# expression/mood, pose/posture, gender presentation, facial hair (or lack
# of it), approximate hair length/color IF plausible, build, and energy
# level. This is what step 1 of the instruction below uses to decide who
# someone resembles. Nothing in here requires an actual costume.
#
# POSTER LOOK -- the character's iconic costume/props/backdrop. Nobody at
# the booth is wearing this; it exists purely for step 4 to pull
# image-editing details from once a character has already been chosen.
# Never used for matching.
#
# This split exists because most of the roster's old descriptions were
# written costume-first -- e.g. Daenerys's "silver-platinum braided hair",
# Joker's "chalk-white face paint" -- which are things a real photo can
# essentially never have. A handful of characters with "everyday" traits
# (glasses, dark clothing, a calm expression) were soaking up most matches
# by default, simply because they were the only ones an ordinary photo
# COULD match. Every character below now has its own realistic, reachable
# trigger, specifically so the roster doesn't collapse onto a few "safe"
# names.
CHARACTER_STYLE_GUIDE = {
    "Iron Man": (
        "MATCH ON: visible facial hair in a goatee or defined stubble "
        "shape, paired with a closed-mouth smirk pulling to ONE side. "
        "Male-presenting. Facial hair is the deciding cue -- a smirk "
        "without any beard/stubble points to Sheldon or Thomas Shelby "
        "instead. POSTER LOOK: sleek red-and-gold powered armor, glowing "
        "blue arc reactor at the chest, faceplate open to reveal the "
        "goatee underneath, repulsor palms raised, bright high-tech "
        "backdrop (lab light, sky, explosion glow)."
    ),
    "Batman": (
        "MATCH ON: a broad, square-ish jaw held with a completely flat, "
        "unsmiling mouth -- neutral, not a smirk, not a frown. "
        "Male-presenting, clean-shaven. This is the 'plain serious face, "
        "no beard, no smirk' bucket -- if there's a beard, that's Jon "
        "Snow; if there's a one-sided smirk, that's Iron Man or Sheldon. "
        "POSTER LOOK: matte-black armored bat-suit, flowing cape, cowl "
        "with pointed ears, utility belt, moody Gotham-noir lighting "
        "(deep shadows, cold blue-grey highlights, rain-slicked "
        "rooftops)."
    ),
    "Spider-Man": (
        "MATCH ON: match this purely from the face, not a costume or "
        "pose -- nobody will be dressed as or posing like Spider-Man. "
        "Look for a youthful, boyish face with a lopsided/crooked grin "
        "(closed or slightly open, higher on one side than the other) "
        "and bright, slightly squinted, mischievous eyes. Male-presenting, "
        "no facial hair, leaner/younger-looking build. Distinct from Iron "
        "Man (no facial hair here) and from Naruto (grin is lopsided/sly "
        "here, not a big even open-mouth grin). POSTER LOOK: skin-tight "
        "red-and-blue spandex suit, black web patterning, large white "
        "reflective eye lenses, city skyline backdrop with web-lines."
    ),
    "Thomas Shelby": (
        "MATCH ON: a lean face with high, sharp cheekbones, held in a "
        "genuinely still, barely-moving closed mouth -- calmer and more "
        "composed than Batman's flat-but-ordinary neutral. Male-presenting, "
        "clean-shaven, neat hair. The lean/sharp cheekbone shape is what "
        "separates this from Batman's broader square jaw. POSTER LOOK: "
        "fitted 1920s tweed three-piece suit, flat cap pulled low, "
        "slicked-back undercut hair, muted sepia Birmingham streets or a "
        "dim pub."
    ),
    "Naruto": (
        "MATCH ON: a round, full-cheeked young face with a wide, open, "
        "genuinely happy grin showing the top teeth, eyebrows relaxed "
        "(not raised, not low/thick). Male-presenting, young, no facial "
        "hair. Against Joker: eyebrows stay neutral, not arched dramatically "
        "high. Against Shinchan: eyebrows are relaxed, not thick and low. "
        "POSTER LOOK: orange-and-black jumpsuit, blue forehead protector, "
        "spiky blonde hair, whisker cheek markings, anime-style "
        "chakra-burst backdrop."
    ),
    "Professor from Money Heist": (
        "MATCH ON: a calm, composed, steady gaze held with a closed, "
        "still mouth, on a face that reads as noticeably more mature or "
        "older-looking than a typical college-age student on its own "
        "merits (not by comparing to any other photo) -- a light beard or "
        "mustache is a supporting cue if present. Male-presenting. This "
        "will be one of the rarer matches given a young crowd, and that's "
        "expected -- do not force it onto young, smooth-faced people just "
        "because they're calm and wearing glasses; that's Sheldon's "
        "territory instead. POSTER LOOK: default to the iconic heist-mode "
        "costume -- bright red jumpsuit plus a white Salvador Dali mask "
        "(painted grin, rosy cheeks, thin mustache) -- a bank vault or "
        "sunbaked Spanish plaza backdrop. Only use the alternate academic "
        "look (thick-rimmed glasses, tousled hair, rumpled cardigan) if "
        "the person's own glasses/cardigan were themselves the standout "
        "match reason."
    ),
    "Gwen Stacy": (
        "MATCH ON: one eyebrow cocked distinctly higher than the other, "
        "paired with a one-sided smirk and a loose, casual head tilt. "
        "Female-presenting. The head tilt is what separates this from "
        "Regina George, whose smirk is even on both sides with the head "
        "held straight. POSTER LOOK: pink-and-white hooded Spider suit "
        "with a black spider emblem, hood down to show "
        "blonde-and-pink-streaked hair, comic-book graffiti backdrop."
    ),
    "Black Widow": (
        "MATCH ON: a completely flat, unsmiling mouth held with a "
        "straight, level head (no tilt at all), on sharp/angular "
        "cheekbones. Female-presenting. The no-tilt straight posture is "
        "what separates this from Gwen Stacy's tilted head. POSTER LOOK: "
        "fitted black tactical bodysuit, utility straps, wavy red hair, "
        "cool steel-blue or shadowed tactical backdrop."
    ),
    "Daenerys Targaryen": (
        "MATCH ON: softly rounded cheeks (not sharp/angular) on a wide "
        "forehead, with a calm expression and chin held level or very "
        "slightly raised. Female-presenting; light/blonde hair is a bonus "
        "cue if visible. The soft roundness separates this from Black "
        "Widow's sharper cheekbones. POSTER LOOK: long silver-platinum "
        "braided hair, flowing pale ivory or deep-blue gown, sunbaked "
        "desert city or grand throne-hall backdrop."
    ),
    "Hermione Granger": (
        "MATCH ON: visibly furrowed or knit-together eyebrows (active "
        "concentration) paired with wide, alert eyes looking directly "
        "forward. Female-presenting; glasses are a bonus cue if present. "
        "The furrowed brow is what separates this from Wednesday's "
        "totally flat/passive brow. POSTER LOOK: bushy brown hair, black "
        "Gryffindor robes with a maroon-and-gold tie, candlelit castle "
        "library backdrop."
    ),
    "Wednesday Addams": (
        "MATCH ON: eyebrows completely flat and passive (not furrowed "
        "like Hermione), combined with a mouth that has zero curve at "
        "all and eyes that read as checked-out/unbothered rather than "
        "alert. Female-presenting, dark or neutral hair. Reads as more "
        "disengaged than Black Widow's composed flatness. POSTER LOOK: "
        "jet-black hair in two tight braids, black Victorian-collared "
        "dress, foggy graveyard or gothic-academy backdrop."
    ),
    "Kakashi Hatake": (
        "MATCH ON: naturally heavy, half-lidded or sleepy-looking eyes "
        "combined with a loose, closed-mouth ease -- reads as genuinely "
        "laid-back rather than merely calm. Male-presenting, "
        "tousled/messy hair if visible. This should be a distinct "
        "'sleepy-eyed' look, not just any relaxed face -- if the eyes are "
        "fully open and alert, this isn't the match; consider Thomas "
        "Shelby or Batman instead. POSTER LOOK: messy silver hair, Leaf "
        "Village forehead protector tilted over one eye, fabric mask over "
        "nose and mouth, navy flak vest, misty forest or rooftop-at-dusk "
        "backdrop."
    ),
    "Sasuke Uchiha": (
        "MATCH ON: a young, narrow/angular face (not broad like Batman) "
        "with thin, straight eyebrows and a hard, direct stare, mouth "
        "flat. Male-presenting, young, no facial hair, dark hair if "
        "visible. The narrow face shape is the deciding cue against "
        "Batman's broader jaw. POSTER LOOK: navy-and-white high-collared "
        "ninja outfit, katana on the back, Sharingan eye detail, crackling "
        "blue lightning backdrop."
    ),
    "Joker": (
        "MATCH ON: a wide, open-mouthed grin/laugh combined with "
        "eyebrows raised dramatically high and wide, energetic eyes -- "
        "reads as manic/theatrical rather than simply happy. "
        "Male-presenting. The high-arched raised eyebrows are what "
        "separate this from Naruto (neutral brows) and Shinchan (low, "
        "thick brows). POSTER LOOK: rumpled purple tailcoat over a green "
        "vest, chalk-white face paint, smeared red smile, chaotic "
        "neon-lit city backdrop."
    ),
    "Jon Snow": (
        "MATCH ON: visible beard or heavy stubble covering the jaw, on a "
        "long face with a weary or serious expression. Male-presenting. "
        "The beard is close to a requirement -- do not pick this "
        "clean-shaven, that's Batman's or Sasuke's territory. POSTER "
        "LOOK: dark tousled shoulder-length hair, thick black furs and "
        "leather armor, snow-covered northern landscape or icy-wall "
        "backdrop."
    ),
    "Shinchan": (
        "MATCH ON: thick, low-set eyebrows drawn down close to the eyes, "
        "combined with a wide, open-mouthed laugh -- reads as goofy/silly "
        "rather than warm. Distinct from Naruto (neutral brows, warmer "
        "grin) and Joker (high raised brows, manic energy) by the low, "
        "thick brow position specifically. POSTER LOOK: spiky messy "
        "black hair, simple bright shirt and shorts, bold flat colorful "
        "cartoon-style backdrop."
    ),
    "Harry Potter": (
        "MATCH ON: round/circular-shaped glasses specifically (not "
        "rectangular), combined with messy, unstyled hair and a "
        "wide-eyed, slightly startled or eager look. Male-presenting, "
        "young. Round glasses shape plus messy hair is what separates "
        "this from Sheldon (rectangular glasses, neat hair). If hair is "
        "neat/styled, this is not the match even with round glasses. "
        "POSTER LOOK: round glasses, lightning-bolt scar, messy black "
        "hair, Gryffindor robes with a red-and-gold scarf, Hogwarts "
        "castle or starry quidditch-pitch backdrop."
    ),
    "Dr. Doom": (
        "MATCH ON: match from the face, not a pose -- a strong, broad jaw "
        "with narrowed, hard-set eyes and mouth corners turned slightly "
        "down, reading as stern/superior rather than merely neutral. "
        "Works for either gender presentation. The slight downturn and "
        "narrowed glare separate this from Batman's totally flat, "
        "unreadable neutral. POSTER LOOK: full green-and-metal armor, "
        "smooth expressionless mask, flowing green hooded cloak, gothic "
        "stone castle or arcane-green-energy backdrop."
    ),
    "Regina George": (
        "MATCH ON: sleek, smoothed-down hair with a closed-mouth smirk "
        "that is even on BOTH sides (not one-sided), chin held level or "
        "slightly down. Female-presenting. The even two-sided smirk with "
        "no head tilt separates this from Gwen Stacy's one-sided "
        "smirk-plus-tilt. POSTER LOOK: sleek blonde hair, trendy pink or "
        "designer outfit, glossy high-school-hallway or pastel-pink "
        "backdrop."
    ),
    "Elle Woods": (
        "MATCH ON: a full, open-mouthed bright smile on a soft, rounded "
        "face (not sharp cheekbones), with an upbeat, bouncy energy. "
        "Female-presenting. The open full smile plus soft round face "
        "separates this from Regina's closed even smirk. POSTER LOOK: "
        "voluminous blonde hair, bold pink outfit (blazer, dress, or "
        "accessories), sunny pastel-campus or courtroom backdrop."
    ),
    "Sheldon Cooper": (
        "MATCH ON: rectangular or thick-rimmed glasses (not round) with "
        "neat, styled hair, and a thin, prim closed-mouth line where one "
        "corner sits very slightly higher than the other. Male-presenting, "
        "clean-shaven. The rectangular glasses plus neat hair is the "
        "deciding combo against Harry Potter's round glasses and messy "
        "hair -- if hair is messy, go there instead. POSTER LOOK: "
        "side-swept hair, layered graphic tee under a collared shirt, "
        "cluttered apartment with an equation-covered whiteboard "
        "backdrop."
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
    match and actually hit OpenAI's cache.

    Each image is downscaled to REFERENCE_IMAGE_MAX_DIM and re-encoded as
    JPEG via image_utils -- these files were previously read raw off disk at
    whatever size they happened to be saved at (often full-resolution
    showcase/poster art), and with 21 of them stacked into every non-cached
    analysis call, that was very likely the biggest single latency cost in
    the whole pipeline. Downscaling happens once here at load/rebuild time,
    not per-request, so it costs nothing on the hot path."""
    found = {}
    ext_mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}
    for name in CHARACTERS:
        slug = _slugify(name)
        for ext, mime in ext_mime.items():
            path = os.path.join(CHARACTER_REF_DIR, f"{slug}.{ext}")
            if os.path.exists(path):
                with open(path, "rb") as f:
                    raw_bytes = f.read()
                small_bytes = image_utils.prepare_image_bytes(raw_bytes, max_dim=REFERENCE_IMAGE_MAX_DIM)
                found[name] = (small_bytes, "image/jpeg")
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

Each roster entry below has two parts: "MATCH ON" (realistic cues an
ordinary, un-costumed phone photo can actually show -- expression, pose,
gender presentation, facial hair, build, energy) and "POSTER LOOK" (the
character's iconic costume, purely for step 4, never a matching cue). Base
this decision ONLY on the "MATCH ON" text -- ignore "POSTER LOOK" details
entirely when deciding who someone resembles, since nobody at the booth is
actually wearing a costume.

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
presentation) before committing to a match like that. Several roster entries
(Professor from Money Heist, Sheldon Cooper, Harry Potter) explicitly warn
that glasses plus a neutral/calm expression is common and weak evidence on
its own -- respect those warnings rather than defaulting to those names.

If more than one character is a genuinely close, plausible match, do not
always resolve the tie toward the most famous or most "obvious" name --
treat close ties as a real choice and let a less-expected but still
well-supported character win sometimes, so the roster doesn't collapse onto
the same 4-5 famous defaults across many different people.

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

   Then pack in only concrete, high-signal visual anchors: (a) the character's
   FULL iconic costume, pulled from the chosen character's "POSTER LOOK" text
   above (never from "MATCH ON") -- describe it as a complete outfit swap
   (e.g. for the Professor: the entire red jumpsuit AND the white Dali mask,
   not just "a red accent"; for Iron Man: the full red-and-gold armor, not
   just a chest piece). The person's actual clothing should be entirely
   replaced by the costume, not layered with or peeking through it. List the
   2-4 most defining pieces/colors of that full costume so the model has
   enough to fully cover the subject, but never touch hair, facial structure,
   or anything that would alter the person's actual face,
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


def _call_openai(
    image_bytes: bytes,
    mime_type: str,
    include_reference_images: bool = True,
    try_extended_retention: bool = True,
):
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

    kwargs = dict(
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
        # No temperature override -- gpt-5-mini (a reasoning model) only
        # supports its default value of 1 and errors on any other value
        # (confirmed live: "Unsupported value: 'temperature' does not
        # support 0.95 with this model"). Match variety for close ties is
        # instead handled entirely via the tie-break prompt language in
        # _build_instruction() rather than a sampling-temperature knob.
        reasoning_effort=REASONING_EFFORT,
    )

    global _extended_retention_supported
    if try_extended_retention and _extended_retention_supported and EXTENDED_CACHE_RETENTION:
        kwargs["prompt_cache_retention"] = EXTENDED_CACHE_RETENTION

    return _client.chat.completions.create(**kwargs)


def _log_cache_usage(response) -> None:
    """Debug: surface how much of this request's prompt was served from
    OpenAI's automatic cache. Chat Completions reports this under
    usage.prompt_tokens_details.cached_tokens -- 0 means no cache hit (first
    request after a cold start, prefix changed, or the >~10min idle window
    expired), any nonzero value means the roster prefix (instructions +
    reference images) was reused instead of reprocessed."""
    retention_status = (
        f"extended retention: {EXTENDED_CACHE_RETENTION}"
        if _extended_retention_supported and EXTENDED_CACHE_RETENTION
        else "extended retention: OFF (not supported or disabled -- using OpenAI's default ~5-10min window)"
    )
    try:
        usage = response.usage
        cached = usage.prompt_tokens_details.cached_tokens
        total_prompt = usage.prompt_tokens
        print(f"[openai_analysis] prompt cache: {cached}/{total_prompt} "
              f"prompt tokens served from cache"
              + (" (cache HIT)" if cached > 0 else " (cache MISS)")
              + f" | {retention_status}")
    except Exception as e:
        print(f"[openai_analysis] could not read cache usage info: {e} | {retention_status}")


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

    Extended (24h) cache retention is attempted first if enabled; if this
    specific request is what's rejected, we've almost certainly found that
    the account/model doesn't support prompt_cache_retention. That's logged
    loudly (search terminal for "EXTENDED CACHE RETENTION NOT SUPPORTED")
    and the global flag flips off, so this exact failure only happens once
    per process -- every later request goes straight back to OpenAI's
    default caching window instead of paying for a doomed retry each time.
    """
    global _extended_retention_supported
    try:
        response = _call_openai(image_bytes, mime_type)
    except Exception as e:
        if _extended_retention_supported and "prompt_cache_retention" in str(e).lower():
            _extended_retention_supported = False
            print(f"[openai_analysis] EXTENDED CACHE RETENTION NOT SUPPORTED on "
                  f"{MODEL_ID} ({e}); disabling it for the rest of this process and "
                  f"falling back to OpenAI's default ~5-10min cache window")
            try:
                response = _call_openai(image_bytes, mime_type)
            except Exception as e2:
                print(f"[openai_analysis] retry without extended retention also "
                      f"failed ({e2}); retrying once with text-only roster, no images")
                try:
                    response = _call_openai(image_bytes, mime_type, include_reference_images=False)
                except Exception as e3:
                    print(f"[openai_analysis] text-only retry also failed ({e3}); "
                          f"giving up on this capture")
                    raise
        else:
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