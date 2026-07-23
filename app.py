"""
Orchestrator: ties all four stages into one async job.

Run with: uvicorn app:app --host 0.0.0.0 --port 8000
"""

import os
import time
import uuid
import asyncio
import datetime
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import traceback
load_dotenv()

import gemini_analysis
import image_gen
import compose
import storage  # Supabase Storage upload/list/delete helpers
import emailer  # optional Gmail SMTP send of the finished poster

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

# --------------------------------------------------------------------------
# Storage cleanup: opt-in (off by default). Turn on with
# SUPABASE_CLEANUP_ENABLED=true if you ever need to keep the bucket bounded
# for a bigger event -- every uploaded file (posters + QR codes) then gets
# deleted once it's older than SUPABASE_CLEANUP_AGE_HOURS, checked on a
# timer in the background. POST /booth/admin/cleanup for an on-demand sweep
# is always available regardless of this flag, so you can still force one
# manually (e.g. end of event) without turning on the automatic timer.
# --------------------------------------------------------------------------
CLEANUP_ENABLED = os.environ.get("SUPABASE_CLEANUP_ENABLED", "false").strip().lower() == "true"
CLEANUP_MAX_AGE_HOURS = float(os.environ.get("SUPABASE_CLEANUP_AGE_HOURS", "6"))
CLEANUP_INTERVAL_S = int(os.environ.get("SUPABASE_CLEANUP_INTERVAL_S", "3600"))  # hourly by default


def _run_cleanup_once() -> list:
    """Lists the bucket, deletes anything older than CLEANUP_MAX_AGE_HOURS,
    returns the names removed. Runs in a thread (blocking Supabase calls)."""
    objects = storage.list_objects()
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=CLEANUP_MAX_AGE_HOURS)

    stale = []
    for obj in objects:
        created = obj.get("created_at") or obj.get("updated_at")
        if not created:
            continue
        try:
            created_dt = datetime.datetime.fromisoformat(created.replace("Z", "+00:00"))
        except ValueError:
            continue
        if created_dt < cutoff:
            stale.append(obj["name"])

    storage.delete_objects(stale)
    return stale


async def _periodic_cleanup_loop():
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_S)
        try:
            removed = await asyncio.to_thread(_run_cleanup_once)
            if removed:
                print(f"[cleanup] removed {len(removed)} stale object(s): {removed}")
        except Exception:
            traceback.print_exc()


@app.on_event("startup")
async def _start_background_cleanup():
    if CLEANUP_ENABLED:
        asyncio.create_task(_periodic_cleanup_loop())
    else:
        print("[cleanup] automatic sweep disabled (SUPABASE_CLEANUP_ENABLED not set to true) "
              "-- POST /booth/admin/cleanup is still available for a manual/on-demand sweep")


@app.post("/booth/admin/cleanup")
async def manual_cleanup():
    """Triggers an immediate cleanup sweep -- handy for an end-of-event
    cleanup without waiting for the next scheduled tick. Also temporary in
    spirit like the other /booth/admin/* endpoints; fine to leave running,
    but don't expose it publicly without auth if you keep it long-term."""
    try:
        removed = await asyncio.to_thread(_run_cleanup_once)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
    return {"removed_count": len(removed), "removed": removed}

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

        # No email is sent as part of this pipeline. Emailing is a separate,
        # opt-in action the visitor triggers themselves via the "email me
        # this poster" button on the reveal screen once they've actually
        # seen the result -- see /booth/email/{job_id} below. email_sent
        # starts False here and is only ever flipped by that endpoint.
        JOBS[job_id].update({
            "status": "done",
            "image_url": image_url,
            "qr_url": qr_url,
            "character": match["character"],
            "reasoning": match.get("reasoning", ""),
            "caption": match["caption"],
            "traits": match["detected_traits"],
            "email_sent": False,
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
# Opt-in email, triggered by the visitor clicking "email me this poster" on
# the reveal screen -- i.e. after the poster + QR are already showing, not
# during capture. This is deliberately fire-and-forget: the endpoint
# schedules the actual fetch+SMTP-send as a background asyncio task and
# returns immediately, so the click never blocks the browser and the send
# itself (which can take a few seconds) never blocks the event loop or a
# concurrent /booth/capture for the next visitor.
# --------------------------------------------------------------------------
async def _email_poster_background(job_id: str, to_address: str):
    job = JOBS.get(job_id)
    if not job:
        return
    try:
        await asyncio.to_thread(
            emailer.send_poster_email_from_url,
            to_address, job["character"], job["caption"], job["image_url"],
        )
        JOBS[job_id]["email_sent"] = True
        JOBS[job_id]["email_error"] = None
    except Exception as e:
        traceback.print_exc()
        JOBS[job_id]["email_sent"] = False
        JOBS[job_id]["email_error"] = str(e)


@app.post("/booth/email/{job_id}")
async def email_poster(job_id: str, email: str = Form(...)):
    job = JOBS.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "unknown job_id"})
    if job.get("status") != "done":
        return JSONResponse(status_code=409, content={"error": "poster not ready yet"})

    email = email.strip()
    if not emailer.is_plausible_email(email):
        return JSONResponse(status_code=400, content={"error": "that doesn't look like a valid email address"})

    JOBS[job_id]["email_sent"] = None  # None = in flight, distinct from True/False
    asyncio.create_task(_email_poster_background(job_id, email))
    return {"queued": True}


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


# --------------------------------------------------------------------------
# TEMPORARY DEV/ADMIN ENDPOINT -- remove before the event.
#
# Rebuilds the Gemini character-reference cache from whatever's currently in
# static/characters/. --reload only restarts on .py file changes, so dropping
# a new image in that folder while the server is already running won't pick
# it up on its own -- hit this endpoint after adding/replacing images instead
# of restarting uvicorn.
# --------------------------------------------------------------------------
@app.post("/booth/admin/rebuild-cache")
async def rebuild_cache():
    try:
        cache_name = await asyncio.to_thread(gemini_analysis.rebuild_character_cache)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
    return {
        "cache_name": cache_name,
        "cached": cache_name is not None,
        "characters_with_images": gemini_analysis.list_characters_with_images(),
    }
