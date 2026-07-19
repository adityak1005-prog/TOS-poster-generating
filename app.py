"""
Orchestrator: ties all six stages into one async job.

Run with: uvicorn app:app --host 0.0.0.0 --port 8000
"""

import io
import time
import uuid
import asyncio
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

import cv_analysis
import character_match
import image_gen
import compose
import storage  # your cloud storage upload helper -- S3 / GCS / Cloudinary etc.

app = FastAPI()
JOBS: dict[str, dict] = {}  # swap for Redis if running multiple booth machines


async def run_pipeline(job_id: str, image_bytes: bytes):
    timings = {}
    t_start = time.time()

    def mark(stage: str, t0: float):
        timings[stage] = round(time.time() - t0, 2)

    try:
        JOBS[job_id]["status"] = "processing"

        # Stage 1 already happened client-side (the capture). We just decode.
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(img)

        # Stage 2: CV trait extraction -- CPU-bound, run in a thread so the
        # event loop stays free to accept other people's requests.
        t0 = time.time()
        traits = await asyncio.to_thread(cv_analysis.analyze, image_np)
        mark("cv_analysis", t0)

        # Stage 3: character matching -- usually instant, occasionally one LLM call
        t0 = time.time()
        match = await asyncio.to_thread(character_match.match_character, traits)
        mark("character_match", t0)

        # Stage 4: prompt construction -- pure string building, negligible time
        t0 = time.time()
        prompt = image_gen.build_prompt(traits, match)
        mark("prompt_build", t0)

        # Stage 5: poster generation -- the network-bound, GPU-bound stage.
        # This is a blocking requests call under the hood; run it in a thread
        # too so it doesn't block other jobs' CPU-bound stages.
        t0 = time.time()
        poster_bytes = await asyncio.to_thread(
            image_gen.generate_poster, image_np, prompt, traits.get("pose_keypoints")
        )
        mark("poster_generation", t0)

        # Stage 6: compose + upload + QR
        t0 = time.time()
        final_bytes = compose.compose_split_screen(image_np, poster_bytes, traits, match)
        image_url = await asyncio.to_thread(storage.upload, final_bytes, f"{job_id}.jpg")
        qr_bytes = compose.make_qr_for_url(image_url)
        qr_url = await asyncio.to_thread(storage.upload, qr_bytes, f"{job_id}_qr.png")
        mark("compose_and_share", t0)

        timings["total"] = round(time.time() - t_start, 2)

        JOBS[job_id].update({
            "status": "done",
            "image_url": image_url,
            "qr_url": qr_url,
            "traits": traits,
            "match": match,
            "timings": timings,
        })

    except Exception as e:
        JOBS[job_id].update({"status": "failed", "error": str(e), "timings": timings})


@app.post("/booth/capture")
async def capture(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    image_bytes = await file.read()
    JOBS[job_id] = {"status": "queued"}
    asyncio.create_task(run_pipeline(job_id, image_bytes))
    return {"job_id": job_id}


@app.get("/booth/status/{job_id}")
async def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "unknown job_id"})
    return job
