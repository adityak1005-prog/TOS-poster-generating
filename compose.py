"""
Stage 4: compose the split-screen image and generate the shareable QR code.

Left panel: the original captured photo, with a text overlay of the traits
Gemini reported (outfit color, pose description, glasses) -- no skeleton
dots, since there are no local pose keypoints in this architecture anymore.
Right panel: the generated poster from stage 3.
"""

import io
import qrcode
from PIL import Image, ImageDraw

CANVAS_H = 1400
PANEL_W = 900


def _draw_overlay(original_image: Image.Image, traits: dict) -> Image.Image:
    """Left panel: original photo with a semi-transparent label bar
    showing what Gemini detected."""
    img = original_image.convert("RGB").resize((PANEL_W, CANVAS_H), Image.LANCZOS)

    # Draw a translucent bar behind the text so it stays readable over any
    # photo background, then the trait text on top of it.
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    bar_height = 90
    draw.rectangle([0, 0, PANEL_W, bar_height], fill=(0, 0, 0, 140))

    label = f"{traits.get('outfit_color', 'unknown')} outfit  |  {traits.get('pose_description', 'neutral pose')}"
    if traits.get("glasses"):
        label += "  |  glasses"

    draw.text((20, 30), label, fill=(255, 255, 255, 255))

    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def compose_split_screen(original_image_bytes: bytes, poster_bytes: bytes, traits: dict, match: dict) -> bytes:
    original = Image.open(io.BytesIO(original_image_bytes))
    left = _draw_overlay(original, traits)
    right = Image.open(io.BytesIO(poster_bytes)).convert("RGB").resize((PANEL_W, CANVAS_H), Image.LANCZOS)

    canvas = Image.new("RGB", (PANEL_W * 2 + 20, CANVAS_H + 100), "black")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (PANEL_W + 20, 0))

    draw = ImageDraw.Draw(canvas)
    caption = f"You matched: {match['character']}  --  {match.get('caption', '')}"
    draw.text((30, CANVAS_H + 30), caption, fill="white")

    out = io.BytesIO()
    canvas.save(out, format="JPEG", quality=92)
    return out.getvalue()


def make_qr_for_url(url: str) -> bytes:
    """QR pointing at the hosted final image so people scan and save/share it."""
    qr = qrcode.QRCode(box_size=10, border=2)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()
