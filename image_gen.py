"""
Stage 3: poster generation via OpenAI's gpt-image-1, using the
`images.edit` endpoint so the original photo is used as the input image
(image-to-image styling) rather than generating purely from a text prompt.

Why images.edit and not images.generate: `generate` only takes a text
prompt and produces *a* person in the character's style -- not *this*
person. `edit` takes the captured photo plus the prompt and asks the model
to transform that specific image, which keeps the subject's likeness/pose
recognizable, matching what the booth promises ("styled as a character, in
the person's own pose"). This mirrors the old ControlNet+img2img design in
spirit, just via OpenAI's editing endpoint instead of a pose-conditioned
diffusion model.

This call blocks for the duration of generation. app.py runs it via
asyncio.to_thread and wraps it in a hard timeout so one slow request can't
stall the booth -- unchanged from the previous architecture.
"""

import os
import requests
import base64
from openai import OpenAI
import image_utils

# gpt-image-2 (released April 2026) is OpenAI's current state-of-the-art
# image model and replaces gpt-image-1 as the default here. It uses the
# same images.edit endpoint/parameters, so this is a drop-in swap -- no
# other code changes needed. Two things worth knowing: it's rated higher
# quality but only "medium" speed (not confirmed faster than gpt-image-1,
# possibly slower), and it isn't available on OpenAI's Free usage tier --
# override via OPENAI_IMAGE_MODEL=gpt-image-1 if your account can't use it
# yet or you want the old speed/quality trade-off back.
IMAGE_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-2")
POSTER_SIZE = os.environ.get("OPENAI_IMAGE_SIZE", "1024x1536")  # portrait poster

# "auto" lets OpenAI pick (effectively the slowest/highest tier). "low" is
# the fastest/cheapest tier on gpt-image-2 (~3-8s vs ~20-40s for "medium"),
# with quality reported as meaningfully better than previous-generation
# models' "low" tier -- worth it for a live booth where throughput matters
# more than maximum polish. Override with OPENAI_IMAGE_QUALITY=medium/high
# if you want to trade speed for quality.
IMAGE_QUALITY = os.environ.get("OPENAI_IMAGE_QUALITY", "low")

# The captured photo (especially a phone upload) can be several thousand
# pixels on a side. None of that extra resolution improves the edit result --
# gpt-image-1 downsamples internally anyway -- it just means more bytes to
# upload and more pixels for the model to process before it starts
# generating. Downscaling to a sane input size before the edit call cuts
# that overhead without visibly affecting the output, since the poster is
# a full regeneration, not a resize of the original.
MAX_INPUT_DIM = int(os.environ.get("OPENAI_INPUT_MAX_DIM", "1280"))

# gpt-image-1 defaults to PNG if you don't ask for anything else -- PNG is
# lossless and noticeably bigger than a compressed JPEG for a photo-style
# poster, which means a slower Supabase upload and slower QR/email delivery
# downstream for no visual benefit here (compose.py re-encodes to JPEG
# anyway for the final composite, so PNG's lossless precision was being
# thrown away one step later regardless). Only "jpeg"/"webp" support the
# compression param; "png" ignores it.
OUTPUT_FORMAT = os.environ.get("OPENAI_OUTPUT_FORMAT", "jpeg")
OUTPUT_COMPRESSION = int(os.environ.get("OPENAI_OUTPUT_COMPRESSION", "85"))

REQUEST_TIMEOUT_S = 15  # for the plain HTTP calls this module might still make

_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# Fallback used if the configured model/quality combination fails outright
# -- e.g. the account doesn't have gpt-image-2 access yet, or a transient
# API error. "low" quality on gpt-image-1 is the cheapest/fastest/most
# widely-available combination, so it's the last thing tried before giving
# up entirely rather than failing the whole capture.
FALLBACK_MODEL = os.environ.get("OPENAI_IMAGE_MODEL_FALLBACK", "gpt-image-1")
FALLBACK_QUALITY = "low"


def _edit_once(image_bytes: bytes, diffusion_prompt: str, model: str, quality: str) -> bytes:
    """One attempt at the images.edit call with a given model/quality. A
    fresh file-like object is built per attempt since a BytesIO already
    consumed by one upload attempt can't be safely reused for a retry."""
    input_file = image_utils.prepare_image_file(image_bytes, max_dim=MAX_INPUT_DIM)

    edit_kwargs = dict(
        model=model,
        image=input_file,
        prompt=diffusion_prompt,
        size=POSTER_SIZE,
        quality=quality,
        output_format=OUTPUT_FORMAT,
        n=1,
    )
    if OUTPUT_FORMAT in ("jpeg", "webp"):
        edit_kwargs["output_compression"] = OUTPUT_COMPRESSION

    result = _client.images.edit(**edit_kwargs)
    image_entry = result.data[0]

    # gpt-image-1/2 return base64-encoded image data (b64_json) by default;
    # handle both that and a hosted url just in case a future response
    # shape switches to url-based delivery.
    if getattr(image_entry, "b64_json", None):
        return base64.b64decode(image_entry.b64_json)

    if getattr(image_entry, "url", None):
        resp = requests.get(image_entry.url, timeout=REQUEST_TIMEOUT_S)
        resp.raise_for_status()
        return resp.content

    raise RuntimeError(f"{model} edit response contained no image data")


def generate_poster(person_image_bytes: bytes, diffusion_prompt: str) -> bytes:
    """
    Submit the captured photo + the analysis stage's diffusion_prompt to
    the images.edit endpoint and return the raw poster image bytes.

    Tries the configured IMAGE_MODEL/IMAGE_QUALITY first. If that raises
    (model unavailable on this account, transient API error, timeout, etc.)
    it falls back once to FALLBACK_MODEL/FALLBACK_QUALITY (gpt-image-1 at
    "low" by default) rather than failing the whole capture outright. Every
    fallback/failure is printed so it's visible in the terminal during a
    live event, not just swallowed.
    """
    try:
        return _edit_once(person_image_bytes, diffusion_prompt, IMAGE_MODEL, IMAGE_QUALITY)
    except Exception as e:
        if (IMAGE_MODEL, IMAGE_QUALITY) == (FALLBACK_MODEL, FALLBACK_QUALITY):
            raise  # already the fallback combination, nothing left to fall back to
        print(f"[image_gen] {IMAGE_MODEL}/{IMAGE_QUALITY} edit failed ({e}); "
              f"retrying once with {FALLBACK_MODEL}/{FALLBACK_QUALITY}")
        try:
            return _edit_once(person_image_bytes, diffusion_prompt, FALLBACK_MODEL, FALLBACK_QUALITY)
        except Exception as e2:
            print(f"[image_gen] fallback {FALLBACK_MODEL}/{FALLBACK_QUALITY} edit also "
                  f"failed ({e2}); giving up on this capture")
            raise
