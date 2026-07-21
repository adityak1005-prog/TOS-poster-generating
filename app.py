"""
Orchestrator: ties all four stages into one async job.

Run with: uvicorn app:app --host 0.0.0.0 --port 8000
"""

import os
import time
import uuid
import asyncio
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import traceback
load_dotenv()

import gemini_analysis
import image_gen
import compose
import storage  # Supabase Storage upload helper

app = FastAPI()

# Serves character showcase images if you drop real files into
# static/characters/<slug>.jpg later (see index.html for the naming
# convention). The folder doesn't need to exist yet -- StaticFiles just
# 404s per-file until you add one, the frontend already falls back to a
# generated placeholder card when that happens.
os.makedirs("static/characters", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def serve_frontend():
    """Serves the index.html page at the root URL."""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>index.html not found!</h1><p>Please create index.html in the same directory as app.py.</p>", status_code=404)


@app.get("/booth/characters")
async def characters():
    """Returns the fixed character roster so the frontend showcase always
    matches whatever is actually in gemini_analysis.py -- add a character
    there and it shows up here automatically, no frontend edit needed."""
    return {"characters": gemini_analysis.CHARACTERS}


JOBS: dict[str, dict] = {}  # swap for Redis if running multiple booth machines

POSTER_TIMEOUT_S = 20  # hard budget for the OpenAI image edit call


async def run_pipeline(job_id: str, image_bytes: bytes):
    timings = {}
    t_start = time.time()

    def mark(stage: str, t0: float):
        timings[stage] = round(time.time() - t0, 2)

    try:
        JOBS[job_id]["status"] = "processing"

        # Stage 1 already happened client-side (the capture). image_bytes
        # is the raw photo as uploaded -- no local decode/CV needed here.

        # Stage 2: single multimodal Gemini call -- replaces MediaPipe trait
        # extraction, the rule-based character scorer, the LLM tie-break,
        # and the prompt-builder all at once. Network-bound, so run in a
        # thread like every other blocking stage.
        t0 = time.time()
        match = await asyncio.to_thread(gemini_analysis.analyze_and_match, image_bytes)
        mark("gemini_analysis", t0)

        # Update job dict immediately so the status endpoint exposes the character choice
        JOBS[job_id].update({
            "character": match["character"],
            "reasoning": match.get("reasoning", ""),
            "caption": match["caption"],
            "traits": match["detected_traits"],
            "stage": "generating_poster",
        })
    # Stage 3: poster generation via OpenAI gpt-image-1 (images.edit,
        # image-to-image). Timeout removed to allow for slower generations.
        t0 = time.time()
        poster_bytes = await asyncio.to_thread(
            image_gen.generate_poster, image_bytes, match["diffusion_prompt"]
        )
        mark("poster_generation", t0)

        # Stage 4: compose + upload + QR
        t0 = time.time()
        final_bytes = compose.compose_split_screen(
            image_bytes, poster_bytes, match["detected_traits"], match
        )
        image_url = await asyncio.to_thread(storage.upload, final_bytes, f"{job_id}.jpg")
        qr_bytes = compose.make_qr_for_url(image_url)
        qr_url = await asyncio.to_thread(storage.upload, qr_bytes, f"{job_id}_qr.png")
        mark("compose_and_share", t0)

        timings["total"] = round(time.time() - t_start, 2)

        JOBS[job_id].update({
            "status": "done",
            "image_url": image_url,
            "qr_url": qr_url,
            "character": match["character"],
            "reasoning": match.get("reasoning", ""),
            "caption": match["caption"],
            "traits": match["detected_traits"],
            "timings": timings,
        })

    # ... [inside run_pipeline] ...
    except Exception as e:
        print(f"\n--- JOB {job_id} FAILED ---")
        traceback.print_exc()  # This prints the exact file and line number to your terminal!
        print("---------------------------\n")
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


# --------------------------------------------------------------------------
# TEMPORARY DEV/TESTING ENDPOINT -- remove before the event.
#
# Runs Gemini analysis only (character match + reasoning + caption + the
# diffusion_prompt that would be sent to OpenAI) and returns it directly.
# Deliberately does NOT call image_gen, compose, or storage, so you can
# sanity-check matches and prompt quality without burning gpt-image-1
# credits/quota. Synchronous (no job queue) since this stage alone is fast.
# --------------------------------------------------------------------------
@app.post("/booth/test-analyze")
async def test_analyze(file: UploadFile = File(...)):
    image_bytes = await file.read()
    t0 = time.time()
    try:
        match = await asyncio.to_thread(gemini_analysis.analyze_and_match, image_bytes)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
    match["elapsed_s"] = round(time.time() - t0, 2)
    return match
