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
import io
import sys
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

# Stage 2 (analysis) now runs locally via the Qwen3-VL pipeline already
# built out in local_llm/ (see local_llm/README.md and FINDINGS.md for what
# it is and the benchmark that validated it) instead of the old OpenAI
# reasoning-model call. local_llm/qwen_local.py does `import shared_prompt`
# as a bare/top-level import (it's designed to be run from inside that
# folder), so it needs local_llm/ on sys.path -- it is NOT a package, this
# is the same thing `cd local_llm && python benchmark.py` gets for free.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_llm"))
import qwen_local  # noqa: E402
import shared_prompt  # noqa: E402  (roster/CHARACTERS -- also imported inside qwen_local, same cached module)

from PIL import Image  # noqa: E402
import image_utils  # noqa: E402  (shared photo downscale, same helper image_gen.py uses)
import image_gen
import compose
import storage  # Supabase Storage upload/list/delete helpers (with fallback bucket)

# Which local model size / how many tokens to generate for stage 2 -- see
# local_llm/FINDINGS.md: 4B matched 8B on both quality axes it measured at
# roughly half the VRAM/load time, so 4B is the default. Override via env
# if you want to try 8B on your own hardware.
LOCAL_VLM_MODEL = os.environ.get("LOCAL_VLM_MODEL", "4b")
LOCAL_VLM_MAX_NEW_TOKENS = int(os.environ.get("LOCAL_VLM_MAX_NEW_TOKENS", "500"))
# Shared with image_gen.py's OPENAI_INPUT_MAX_DIM so every stage downscales
# the captured photo to the same size before doing anything with it.
LOCAL_VLM_MAX_DIM = int(os.environ.get("OPENAI_INPUT_MAX_DIM", "1280"))


def _analyze_photo(image_bytes: bytes) -> dict:
    """Stage 2 entry point, same contract the old
    openai_analysis.analyze_and_match(image_bytes) had: takes the captured
    photo's raw bytes, returns a dict with character/reasoning/caption/
    diffusion_prompt/detected_traits.

    qwen_local.analyze_and_match() (see local_llm/qwen_local.py) takes a
    PIL Image instead of raw bytes, and -- since Qwen3-VL isn't
    schema-constrained the way the OpenAI call was -- returns debug/
    validity fields (_valid_json, _error, etc.) rather than raising on a
    malformed response. This wrapper adapts both of those: downscale +
    decode bytes -> PIL Image the same way image_gen.py already does, then
    raise if generation didn't produce valid JSON (so the existing
    try/except in _stream_capture below turns it into a "failed" stage
    exactly like an OpenAI API error used to), and clamp/default the two
    fields the old code guaranteed were always present.
    """
    small_bytes = image_utils.prepare_image_bytes(image_bytes, max_dim=LOCAL_VLM_MAX_DIM)
    image = Image.open(io.BytesIO(small_bytes)).convert("RGB")

    result = qwen_local.analyze_and_match(
        image, model_key=LOCAL_VLM_MODEL, max_new_tokens=LOCAL_VLM_MAX_NEW_TOKENS,
    )
    if not result["_valid_json"]:
        raise RuntimeError(f"local analysis did not return valid JSON: {result['_error']}")

    if result.get("character") not in shared_prompt.CHARACTERS:
        result["character"] = shared_prompt.CHARACTERS[0]
    result.setdefault("reasoning", "")

    return result


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
    matches whatever is actually in the roster (shared_prompt.py, which
    reads it from openai_analysis.py's CHARACTER_STYLE_GUIDE -- that file's
    OpenAI *calling* code is unused now, but it's still the one place the
    roster text itself is defined) -- add a character there and it shows
    up here automatically, no frontend edit needed."""
    return {"characters": shared_prompt.CHARACTERS}


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
  img {{ max-width: 100%; max-height: 60vh; border-radius: 14px; }}
  .btn-row {{ display: flex; flex-direction: column; gap: .7rem; width: 100%; max-width: 22em; }}
  button {{
    font: inherit; cursor: pointer; border: none; border-radius: 14px;
    padding: 1rem 1.5rem; font-size: 1.05rem; font-weight: 600; color: white;
    width: 100%;
  }}
  #downloadBtn {{ background: #26263a; }}
  #shareBtn {{ background: linear-gradient(135deg, #8b5cf6, #ec4899); }}
  button:disabled {{ opacity: .6; cursor: default; }}
  p {{ color: #a3a3b0; font-size: .85rem; max-width: 32em; margin: 0; }}
  #status {{ min-height: 1.1em; }}
</style>
</head><body>
  <img id="poster" src="{img_url}" alt="Your AI Movie Booth poster">
  <div class="btn-row">
    <button id="downloadBtn">Download as JPEG</button>
    <button id="shareBtn">Share to Instagram Story</button>
  </div>
  <p id="status">Tag <b>{handle}</b> in your story so we can see it!</p>
<script>
  const IMG_URL = {img_url_json};
  const statusEl = document.getElementById("status");
  const defaultStatus = statusEl.innerHTML;

  async function fetchPosterBlob() {{
    const resp = await fetch(IMG_URL);
    if (!resp.ok) throw new Error("Could not fetch the poster image.");
    return await resp.blob();
  }}

  // ---- Download: always works, even on browsers with no Web Share API ----
  document.getElementById("downloadBtn").addEventListener("click", async () => {{
    const btn = document.getElementById("downloadBtn");
    btn.disabled = true;
    statusEl.innerText = "Preparing download...";
    try {{
      const blob = await fetchPosterBlob();
      const objectUrl = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = objectUrl;
      a.download = "movie-booth-poster.jpg";
      document.body.appendChild(a);
      a.click();
      a.remove();
      // Give the browser a moment to start the download before revoking.
      setTimeout(() => URL.revokeObjectURL(objectUrl), 4000);
      statusEl.innerHTML = "Saved! Check your downloads/photos.";
    }} catch (e) {{
      // Last-resort fallback: open the raw image so the visitor can
      // long-press -> Save Image manually.
      statusEl.innerText = "Couldn't auto-download -- opening the image, long-press to save.";
      window.open(IMG_URL, "_blank");
    }} finally {{
      btn.disabled = false;
    }}
  }});

  // ---- Share: Web Share API with a file, so Instagram shows up as a
  // native share target on supporting browsers (mainly Android + recent
  // iOS Safari). Desktop browsers mostly don't support file sharing at
  // all, so this degrades to a clear instruction instead of silently
  // doing nothing. ----
  document.getElementById("shareBtn").addEventListener("click", async () => {{
    const btn = document.getElementById("shareBtn");
    btn.disabled = true;
    statusEl.innerText = "Opening share sheet...";
    try {{
      const blob = await fetchPosterBlob();
      const file = new File([blob], "movie-booth-poster.jpg", {{ type: blob.type || "image/jpeg" }});
      if (navigator.canShare && navigator.canShare({{ files: [file] }})) {{
        await navigator.share({{ files: [file], title: "My AI Movie Booth poster" }});
        statusEl.innerHTML = defaultStatus;
      }} else {{
        statusEl.innerText = "Sharing isn't supported on this browser -- tap \\"Download as JPEG\\" instead, then upload it to your Instagram Story manually.";
      }}
    }} catch (e) {{
      // AbortError just means the visitor closed the native share sheet --
      // not a real failure, so don't show an error for that case.
      if (e && e.name === "AbortError") {{
        statusEl.innerHTML = defaultStatus;
      }} else {{
        statusEl.innerText = "Sharing isn't supported on this browser -- tap \\"Download as JPEG\\" instead, then upload it to your Instagram Story manually.";
      }}
    }} finally {{
      btn.disabled = false;
    }}
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
        # Stage 2: local Qwen3-VL call (analysis + character match +
        # reasoning + caption + diffusion_prompt) -- see _analyze_photo()
        # above and local_llm/README.md. GPU-bound, not network-bound, but
        # still run in a thread so a multi-second generate() call doesn't
        # block the event loop from serving other requests.
        t0 = time.time()
        match = await asyncio.to_thread(_analyze_photo, image_bytes)
        mark("local_analysis", t0)

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
        match = await asyncio.to_thread(_analyze_photo, image_bytes)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
    match["elapsed_s"] = round(time.time() - t0, 2)
    return match
