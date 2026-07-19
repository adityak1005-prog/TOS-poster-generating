"""
Stage 4 + 5: prompt construction and poster generation.

"""

import os
import time
import base64
import requests

API_KEY = os.environ["DIFFUSION_API_KEY"]
BASE_URL = os.environ.get("DIFFUSION_API_BASE", "https://api.example-fast-diffusion.com/v1")
MODEL_ID = os.environ.get("DIFFUSION_MODEL_ID", "fast-turbo-controlnet")

POLL_INTERVAL_S = 0.5
POLL_TIMEOUT_S = 20  # hard ceiling so one slow job can't blow the 30s budget


def build_prompt(traits: dict, match: dict) -> str:
    """Merge extracted traits + matched character into a single diffusion prompt."""
    pose_phrase = match.get("pose_phrase") or traits["pose"].replace("_", " ")
    return (
        f"cinematic movie poster, subject styled as {match['character']}, "
        f"{traits['outfit_color']} costume accent, {pose_phrase} pose, "
        f"dramatic studio lighting, bold poster typography, high detail, photorealistic, "
        f"college fest promotional poster style"
    )


def generate_poster(person_image_np, prompt: str, pose_keypoints) -> bytes:
    """
    Submit an image-to-image + pose-conditioned generation job.
    Returns the raw image bytes of the generated poster.
    """
    from PIL import Image
    import io

    buf = io.BytesIO()
    Image.fromarray(person_image_np).save(buf, format="JPEG", quality=90)
    image_b64 = base64.b64encode(buf.getvalue()).decode()

    payload = {
        "model": MODEL_ID,
        "prompt": prompt,
        "image": image_b64,          # img2img reference for identity
        "control_type": "pose",       # ControlNet-style pose conditioning
        "control_points": pose_keypoints,
        "num_inference_steps": 4,     # turbo/schnell-class models: 1-8 steps
        "guidance_scale": 1.5,
        "strength": 0.65,             # how much to preserve vs. the source photo
    }
    headers = {"Authorization": f"Bearer {API_KEY}"}

    submit = requests.post(f"{BASE_URL}/generate", json=payload, headers=headers, timeout=10)
    submit.raise_for_status()
    job = submit.json()

    if job.get("status") == "completed":
        return requests.get(job["output_url"], timeout=10).content

    job_id = job["job_id"]
    waited = 0.0
    while waited < POLL_TIMEOUT_S:
        time.sleep(POLL_INTERVAL_S)
        waited += POLL_INTERVAL_S
        status = requests.get(f"{BASE_URL}/jobs/{job_id}", headers=headers, timeout=10).json()
        if status["status"] == "completed":
            return requests.get(status["output_url"], timeout=10).content
        if status["status"] == "failed":
            raise RuntimeError(f"Poster generation failed: {status.get('error')}")

    raise TimeoutError("Poster generation exceeded time budget -- fall back to a retry or a cached template")
