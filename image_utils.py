"""
Shared helper for downscaling the captured photo before it's sent to any
external API.

Why this exists as its own module: previously only image_gen.py downscaled
the photo (via its own private _prepare_input_image) before the OpenAI
poster-edit call -- the analysis call in openai_analysis.py was still
sending the full-resolution original on every single capture. Since
analysis now also runs through OpenAI (see openai_analysis.py), that photo
was effectively being uploaded twice at full size: once for analysis, once
for poster generation. A phone photo can be several thousand pixels on a
side and several MB -- none of that helps either model, both of which
downsample internally anyway. This module gives both call sites one shared,
consistent downscale so that token/bandwidth cost isn't paid twice for the
same waste.
"""

import io
from PIL import Image, ImageOps

DEFAULT_MAX_DIM = 1280
DEFAULT_JPEG_QUALITY = 90


def prepare_image_bytes(image_bytes: bytes, max_dim: int = DEFAULT_MAX_DIM,
                         jpeg_quality: int = DEFAULT_JPEG_QUALITY) -> bytes:
    """Downscales (if needed) and re-encodes as JPEG, returning raw bytes.

    ImageOps.exif_transpose() bakes in the EXIF orientation tag before
    resizing -- phone photos are very often stored "sideways" with an
    orientation tag that viewers apply automatically, and a naive
    Image.open()+resize() silently drops that tag, which is how you get a
    poster generated from a rotated photo. Applying it here means every
    downstream consumer (analysis call, poster-edit call) sees the photo
    right-side-up without needing to handle orientation itself.

    Doesn't touch the original bytes passed in by the caller -- callers
    that also need the untouched original (e.g. compose.py's left panel,
    the final Supabase upload) should keep using the raw bytes from the
    capture, not this function's output.
    """
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")

    width, height = img.size
    scale = max_dim / max(width, height)
    if scale < 1:
        img = img.resize((round(width * scale), round(height * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality)
    return buf.getvalue()


def prepare_image_file(image_bytes: bytes, max_dim: int = DEFAULT_MAX_DIM,
                        jpeg_quality: int = DEFAULT_JPEG_QUALITY) -> io.BytesIO:
    """Same as prepare_image_bytes, but wrapped as a named file-like object
    -- what the OpenAI SDK needs to infer content type for images.edit."""
    data = prepare_image_bytes(image_bytes, max_dim=max_dim, jpeg_quality=jpeg_quality)
    buf = io.BytesIO(data)
    buf.seek(0)
    buf.name = "capture.jpg"
    return buf
