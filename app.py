"""
Orchestrator: ties all four stages into one streamed request.

Run with: uvicorn app:app --host 127.0.0.1 --port 8000

Architecture note: this used to be a job-queue design -- POST /booth/capture
returned a job_id immediately, ran the pipeline in a background asyncio
task, and the frontend polled GET /booth/status/{job_id} until it saw
"done". That pattern doesn't survive on serverless hosts like Vercel: a
bare asyncio.create_task() can be torn down the moment the response is
sent, and even if it weren't, the in-memory JOBS dict it wrote into isn't
visible to whichever instance handles the next poll request. Instead,
/booth/capture now runs the whole pipeline inline and streams progress as
newline-delimited JSON within a single request/response -- the "character
matched" early reveal still works (it's just the first streamed line
instead of a separate poll response), and there's no cross-request state to
lose. See index.html's runFullPipeline() for the client side of this.
"""

import os
import time
import json
import asyncio
import datetime
import traceback
from urllib.parse import quote
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
load_dotenv()

import openai_analysis
import image_gen
import compose
import storage  # Supabase Storage upload/list/delete helpers (with fallback bucket)

app = FastAPI()

# Serves character showcase images if you drop real files into
# static/characters/<slug>.jpg later (see index.html for the naming
# convention). The folder doesn't need to exist yet -- StaticFiles just
# 404s per-file until you add one, the frontend already falls back to a
# generated placeholder card when that happens.
os.makedirs("static/characters", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Shown on the reveal screen next to the QR/share prompt -- the official
# fest Instagram account freshers are asked to tag themselves when sharing
# their poster (see index.html). Configurable so it isn't hardcoded into
# the frontend directly.
ARIES_INSTAGRAM_URL = os.environ.get("ARIES_INSTAGRAM_URL", "https://www.instagram.com/aries.iitd/")
ARIES_INSTAGRAM_HANDLE = os.environ.get("ARIES_INSTAGRAM_HANDLE", "@aries.iitd")


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
    matches whatever is actually in openai_analysis.py -- add a character
    there and it shows up here automatically, no frontend edit needed."""
    return {"characters": openai_analysis.CHARACTERS}


@app.get("/booth/config")
async def config():
    """Small config blob the frontend fetches on load -- currently just the
    Instagram tag prompt, kept server-side/env-configurable rather than
    hardcoded into index.html."""
    return {
        "aries_instagram_url": ARIES_INSTAGRAM_URL,
        "aries_instagram_handle": ARIES_INSTAGRAM_HANDLE,
    }


# --------------------------------------------------------------------------
# QR target: previously the QR just linked straight to the hosted poster
# image, which on scan just opens/downloads a photo -- there's no way for a
# plain image URL to hand that image directly to Instagram. This page
# fetches the image itself and uses the Web Share API's file-sharing
# support (navigator.canShare({files:...})) to bring up the phone's native
# share sheet with the poster already attached, so Instagram (Story or
# Feed) appears as a share target one tap away. Falls back to just opening
# the raw image (same as before) if the browser doesn't support sharing
# files -- desktop browsers mostly don't, but this page is meant to be
# opened from a QR scan on a phone.
# --------------------------------------------------------------------------
_SHARE_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Share your poster</title>
<style>
  body {{
    margin: 0; min-height: 100vh; display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 1rem; padding: 1.5rem;
    background: #0f0f14; color: #f4f4f6;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    text-align: center;
  }}
  img {{ max-width: 100%; max-height: 65vh; border-radius: 14px; }}
  button {{
    font: inherit; cursor: pointer; border: none; border-radius: 14px;
    padding: 1rem 1.5rem; font-size: 1.05rem; font-weight: 600; color: white;
    background: linear-gradient(135deg, #8b5cf6, #ec4899);
  }}
  p {{ color: #a3a3b0; font-size: .85rem; max-width: 32em; margin: 0; }}
</style>
</head><body>
  <img id="poster" src="{img_url}" alt="Your AI Movie Booth poster">
  <button id="shareBtn">Share to Instagram</button>
  <p>Tap Share, then pick Instagram (Story or Feed). Tag <b>{handle}</b> so we can see it!</p>
<script>
  const IMG_URL = {img_url_json};
  document.getElementById("shareBtn").addEventListener("click", async () => {{
    try {{
      const resp = await fetch(IMG_URL);
      const blob = await resp.blob();
      const file = new File([blob], "movie-booth-poster.jpg", {{ type: blob.type || "image/jpeg" }});
      if (navigator.canShare && navigator.canShare({{ files: [file] }})) {{
        await navigator.share({{ files: [file], title: "My AI Movie Booth poster" }});
        return;
      }}
    }} catch (e) {{ /* fall through to the plain-open fallback below */ }}
    // Browser doesn't support file sharing (common on desktop) -- just open
    // the image so the visitor can long-press/save and share manually.
    window.location.href = IMG_URL;
  }});
</script>
</body></html>"""


@app.get("/share")
async def share_page(img: str):
    """Renders the Web-Share-API page above for a given poster URL -- this
    is what the QR code now points at instead of the raw image."""
    return HTMLResponse(content=_SHARE_PAGE_TEMPLATE.format(
        img_url=img,
        img_url_json=json.dumps(img),
        handle=ARIES_INSTAGRAM_HANDLE,
    ))


# --------------------------------------------------------------------------
# On-demand storage cleanup. There is no automatic background sweep anymore
# -- storage.upload() now fails over to a fallback bucket instead (see
# storage.py), which is what actually keeps the booth running if the
# primary bucket fills up or errors, rather than a timer that deletes old
# files. A permanent background loop (the old _periodic_cleanup_loop) can't
# survive on serverless hosts like Vercel anyway -- an instance can be
# paused/recycled between requests, killing a `while True` loop along with
# it. This endpoint is a manual/on-demand tool only (e.g. an end-of-event
# purge), not something the app relies on to keep running.
# --------------------------------------------------------------------------
CLEANUP_MAX_AGE_HOURS = float(os.environ.get("SUPABASE_CLEANUP_AGE_HOURS", "6"))


def _run_cleanup_once() -> list:
    """Lists the primary bucket, deletes anything older than
    CLEANUP_MAX_AGE_HOURS, returns the names removed. Runs in a thread
    (blocking Supabase calls)."""
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


@app.post("/booth/admin/cleanup")
async def manual_cleanup():
    """Triggers an immediate cleanup sweep of the primary bucket -- handy
    for an end-of-event purge. Purely on-demand; nothing calls this
    automatically."""
    try:
        removed = await asyncio.to_thread(_run_cleanup_once)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
    return {"removed_count": len(removed), "removed": removed}


def _sse(payload: dict) -> str:
    """One newline-delimited JSON chunk. Not technically Server-Sent-Events
    framing (no 'data: ' prefix) -- kept as plain NDJSON so the client can
    parse it with a trivial split('\\n') instead of a full SSE parser."""
    return json.dumps(payload) + "\n"


async def _stream_capture(image_bytes: bytes, base_url: str):
    """Generator that runs the whole pipeline and yields progress as NDJSON
    lines. Replaces the old job-queue+poll design (see module docstring) --
    everything happens inline within this one request/response, so there's
    no cross-request state (no JOBS dict) for a serverless host to lose
    track of. base_url is this deployment's own root (e.g.
    "https://your-app.vercel.app/") -- needed to build an absolute /share
    link for the QR code, since QR codes can't point at a relative path.
    """
    timings = {}
    t_start = time.time()

    def mark(stage: str, t0: float):
        timings[stage] = round(time.time() - t0, 2)

    try:
        # Stage 2: single multimodal OpenAI call (analysis + character
        # match + reasoning + caption + diffusion_prompt). Network-bound,
        # run in a thread so it doesn't block the event loop.
        t0 = time.time()
        match = await asyncio.to_thread(openai_analysis.analyze_and_match, image_bytes)
        mark("openai_analysis", t0)

        # First streamed chunk: lets the frontend show the matched
        # character name immediately, same UX the old polling design gave,
        # before the (slower) poster generation stage even starts.
        yield _sse({
            "stage": "matched",
            "character": match["character"],
            "reasoning": match.get("reasoning", ""),
            "caption": match["caption"],
            "traits": match["detected_traits"],
        })

        # Stage 3: poster generation (has its own model/quality fallback --
        # see image_gen.py).
        t0 = time.time()
        poster_bytes = await asyncio.to_thread(
            image_gen.generate_poster, image_bytes, match["diffusion_prompt"]
        )
        mark("poster_generation", t0)

        # Stage 4: compose + upload (has its own fallback bucket -- see
        # storage.py) + QR.
        t0 = time.time()
        final_bytes = compose.compose_split_screen(
            image_bytes, poster_bytes, match["detected_traits"], match
        )
        file_id = f"{int(time.time() * 1000)}"
        image_url = await asyncio.to_thread(storage.upload, final_bytes, f"{file_id}.jpg")
        # QR points at our own /share page (Web Share API, see above), not
        # directly at the image -- that's what lets a scan open a "Share to
        # Instagram" button instead of just opening/downloading the photo.
        share_url = f"{base_url.rstrip('/')}/share?img={quote(image_url, safe='')}"
        qr_bytes = compose.make_qr_for_url(share_url)
        qr_url = await asyncio.to_thread(storage.upload, qr_bytes, f"{file_id}_qr.png")
        mark("compose_and_share", t0)

        timings["total"] = round(time.time() - t_start, 2)

        yield _sse({
            "stage": "done",
            "image_url": image_url,
            "qr_url": qr_url,
            "character": match["character"],
            "reasoning": match.get("reasoning", ""),
            "caption": match["caption"],
            "traits": match["detected_traits"],
            "timings": timings,
        })

    except Exception as e:
        print(f"\n--- CAPTURE FAILED ---")
        traceback.print_exc()  # exact file/line number in the terminal for real-time debugging
        print("-----------------------\n")
        yield _sse({"stage": "failed", "error": str(e), "timings": timings})


@app.post("/booth/capture")
async def capture(request: Request, file: UploadFile = File(...)):
    image_bytes = await file.read()
    return StreamingResponse(
        _stream_capture(image_bytes, str(request.base_url)),
        media_type="application/x-ndjson",
    )


# --------------------------------------------------------------------------
# TEMPORARY DEV/TESTING ENDPOINT -- remove before the event.
#
# Runs stage-2 analysis only (character match + reasoning + caption + the
# diffusion_prompt used in stage 3) and returns it directly. Deliberately
# does NOT call image_gen, compose, or storage, so you can sanity-check
# matches and prompt quality without burning image-generation credits/quota.
# --------------------------------------------------------------------------
@app.post("/booth/test-analyze")
async def test_analyze(file: UploadFile = File(...)):
    image_bytes = await file.read()
    t0 = time.time()
    try:
        match = await asyncio.to_thread(openai_analysis.analyze_and_match, image_bytes)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
    match["elapsed_s"] = round(time.time() - t0, 2)
    return match


# --------------------------------------------------------------------------
# TEMPORARY DEV/ADMIN ENDPOINT -- remove before the event.
#
# Left in place for interface parity with the previous Gemini-backed build.
# openai_analysis.py's "cache" is really a memoized in-process reference-
# image set (OpenAI's own prompt caching is automatic, no handle to return)
# -- this just forces a reload from static/characters/ without restarting
# uvicorn.
# --------------------------------------------------------------------------
@app.post("/booth/admin/rebuild-cache")
async def rebuild_cache():
    try:
        cache_name = await asyncio.to_thread(openai_analysis.rebuild_character_cache)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
    return {
        "cache_name": cache_name,
        "cached": cache_name is not None,
        "characters_with_images": openai_analysis.list_characters_with_images(),
    }
