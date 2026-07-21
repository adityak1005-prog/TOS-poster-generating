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

IMAGE_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1")
POSTER_SIZE = os.environ.get("OPENAI_IMAGE_SIZE", "1024x1536")  # portrait poster
REQUEST_TIMEOUT_S = 15  # for the plain HTTP calls this module might still make

_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def generate_poster(person_image_bytes: bytes, diffusion_prompt: str) -> bytes:
    """
    Submit the captured photo + Gemini's diffusion_prompt to gpt-image-1's
    edit endpoint and return the raw poster image bytes.

    gpt-image-1 requires the input image to be a png/webp/jpg file object,
    not raw bytes directly -- wrap the bytes in an in-memory file-like
    object with a name so the SDK can infer content type, no temp file
    needed on disk.
    """
    input_file = io.BytesIO(person_image_bytes)
    input_file.name = "capture.jpg"  # SDK uses this to infer the mime type

    result = _client.images.edit(
        model=IMAGE_MODEL,
        image=input_file,
        prompt=diffusion_prompt,
        size=POSTER_SIZE,
        n=1,
    )

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
