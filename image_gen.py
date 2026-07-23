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
import io
import requests
import base64
from openai import OpenAI
from PIL import Image

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

# "auto" lets OpenAI pick (effectively the slowest/highest tier for
# gpt-image-1). Explicitly requesting "medium" cuts generation time a lot
# with only a modest fidelity trade-off -- this is the single biggest lever
# on the ~1min generation time. Override with OPENAI_IMAGE_QUALITY=high/low
# if you want to trade speed for quality either direction.
IMAGE_QUALITY = os.environ.get("OPENAI_IMAGE_QUALITY", "medium")

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


def _prepare_input_image(image_bytes: bytes) -> io.BytesIO:
    """Downscale + re-encode the captured photo before it's sent to OpenAI.
    Only affects what gpt-image-1 sees -- the original bytes passed into
    generate_poster() are untouched and still used as-is for the
    split-screen compose step and storage upload."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = img.size
    scale = MAX_INPUT_DIM / max(width, height)
    if scale < 1:
        img = img.resize((round(width * scale), round(height * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    buf.name = "capture.jpg"  # SDK uses this to infer the mime type
    return buf


def generate_poster(person_image_bytes: bytes, diffusion_prompt: str) -> bytes:
    """
    Submit the captured photo + Gemini's diffusion_prompt to gpt-image-1's
    edit endpoint and return the raw poster image bytes.

    gpt-image-1 requires the input image to be a png/webp/jpg file object,
    not raw bytes directly -- the photo is downscaled/re-encoded in-memory
    (see _prepare_input_image) and wrapped in a file-like object with a
    name so the SDK can infer content type, no temp file needed on disk.
    """
    input_file = _prepare_input_image(person_image_bytes)

    edit_kwargs = dict(
        model=IMAGE_MODEL,
        image=input_file,
        prompt=diffusion_prompt,
        size=POSTER_SIZE,
        quality=IMAGE_QUALITY,
        output_format=OUTPUT_FORMAT,
        n=1,
    )
    if OUTPUT_FORMAT in ("jpeg", "webp"):
        edit_kwargs["output_compression"] = OUTPUT_COMPRESSION

    result = _client.images.edit(**edit_kwargs)

    image_entry = result.data[0]

    # gpt-image-1 returns base64-encoded image data (b64_json) by default;
    # handle both that and a hosted url just in case a future response
    # shape switches to url-based delivery.
    if getattr(image_entry, "b64_json", None):
        return base64.b64decode(image_entry.b64_json)

    if getattr(image_entry, "url", None):
        resp = requests.get(image_entry.url, timeout=REQUEST_TIMEOUT_S)
        resp.raise_for_status()
        return resp.content

    raise RuntimeError("gpt-image-1 edit response contained no image data")
