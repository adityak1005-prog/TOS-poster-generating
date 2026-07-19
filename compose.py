"""
Stage 6: compose the split-screen image and generate the shareable QR code.

"""

import io
import qrcode
from PIL import Image, ImageDraw, ImageFont

CANVAS_H = 1400
PANEL_W = 900
SKELETON_COLOR = (255, 210, 60)
SWATCH_SIZE = 60


def _draw_overlay(original_np, traits: dict) -> Image.Image:
    """Left panel: original photo annotated with the detected skeleton + a
    color swatch + trait labels, so people can see what the CV stage saw."""
    img = Image.fromarray(original_np).convert("RGB")
    img = img.resize((PANEL_W, CANVAS_H), Image.LANCZOS)
    scale_x, scale_y = PANEL_W / original_np.shape[1], CANVAS_H / original_np.shape[0]
    draw = ImageDraw.Draw(img)

    # skeleton dots (full connective drawing omitted here for brevity --
    # draw lines between the MediaPipe POSE_CONNECTIONS pairs for a real skeleton)
    if traits.get("pose_keypoints"):
        for x, y, visibility in traits["pose_keypoints"]:
            if visibility < 0.5:
                continue
            px, py = x * scale_x, y * scale_y
            draw.ellipse([px - 5, py - 5, px + 5, py + 5], fill=SKELETON_COLOR)

    # color swatch + label bar
    swatch_rgb = traits.get("outfit_color_rgb", (128, 128, 128))
    draw.rectangle([20, 20, 20 + SWATCH_SIZE, 20 + SWATCH_SIZE], fill=swatch_rgb, outline="white", width=3)
    label = f"{traits['outfit_color']}  |  {traits['pose'].replace('_', ' ')}"
    if traits.get("glasses"):
        label += "  |  glasses"
    draw.text((20 + SWATCH_SIZE + 15, 30), label, fill="white")

    return img


def compose_split_screen(original_np, poster_bytes: bytes, traits: dict, match: dict) -> bytes:
    left = _draw_overlay(original_np, traits)
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
