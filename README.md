# Movie Booth

AI-powered photo booth app with a browser front end (camera capture or file
upload) that analyzes the photo locally to pick a matching character,
generates a poster-style image via a cloud image-editing API, and returns a
shareable split-screen result with a witty catchphrase and the reasoning
behind the match.

Runs locally on your own machine -- there's no hosting/deployment step here.

## What it does

1. The front end (`index.html`) captures a photo -- either live via the
   browser camera or as a file upload -- compresses it client-side, and
   posts it to a FastAPI endpoint.
2. `POST /booth/capture` runs the whole pipeline in one streamed request
   and analyzes the photo **locally**, on your own GPU, using a Qwen3-VL
   vision-language model (see "Local analysis pipeline" below) -- no
   OpenAI call, no network round-trip for this stage. It picks the
   best-matching character from a fixed roster, writes a short caption,
   writes the reasoning behind the match, writes a ready-to-use
   image-editing prompt, and reports a few quick visual traits, all in one
   pass. This is streamed back to the browser immediately so the matched
   character name can be shown before the poster is even generated.
3. Sends the original photo + that prompt to OpenAI's `images.edit`
   endpoint to generate a poster-style version of the person, styled as
   the matched character. This step is still cloud-based -- there's no
   local image-editing model in this repo, so it needs `OPENAI_API_KEY`
   and internet access.
4. Composes a split-screen image (original + poster), uploads it to
   Supabase Storage, and streams back the final URLs plus a QR image. This
   is also cloud-based -- it's what lets a visitor scan a QR code on their
   own phone and get a link to their poster, independent of whatever
   machine is running the booth.
5. On the reveal screen, the visitor sees their poster, catchphrase, QR
   code, and "why you matched" reasoning, plus a prompt to share the
   poster on their Instagram Story and tag the fest's official account --
   no email collection, nothing sent on the visitor's behalf.

### Local analysis pipeline

Stage 2 (character analysis) is the local_llm/ folder's Qwen3-VL
experiment (see `local_llm/README.md` and `local_llm/FINDINGS.md`), wired
directly into `app.py`. That folder benchmarked `Qwen3-VL-4B-Instruct` and
`Qwen3-VL-8B-Instruct` against the exact same roster/prompt the old
OpenAI-backed version used, on all 21 `static/characters/` images, and
found both models hit 100% valid JSON and a 95.2% self-match rate (the one
miss -- Regina George matched to Gwen Stacy -- is a pair the roster's own
docs already flag as deliberately close). 4B matched 8B on both axes at
roughly half the VRAM/load time, so `app.py` defaults to it.

`app.py` imports `local_llm/qwen_local.py` directly (adding `local_llm/` to
`sys.path` so that module's own bare `import shared_prompt` resolves) and
wraps it in a small helper, `_analyze_photo()`, that converts the captured
photo's raw bytes into a PIL image the same way `image_gen.py` already
downscales photos, then adapts `qwen_local.analyze_and_match()`'s
debug-field-heavy return shape into the plain
character/reasoning/caption/diffusion_prompt/detected_traits dict the rest
of `app.py` expects. The `local_llm/` folder itself (its benchmark script,
attention-visualization tooling, and cached results) is untouched --
`app.py` just calls into it.

**The model is loaded lazily, on the first request that reaches stage 2**
(not at server startup), and stays resident in memory for the life of the
process. That first request will be noticeably slower than the rest
(downloading/loading weights, plus one-time CUDA kernel warmup) -- see
"Run the server" below for a way to warm it up before opening the booth to
real visitors.

## Project files

- `app.py` -- FastAPI API and orchestration (streamed capture, on-demand
  cleanup, config endpoints); imports `local_llm/qwen_local.py` for stage 2
- `local_llm/qwen_local.py` -- loads a Qwen3-VL model via `transformers`
  and runs the analysis call; this is what `app.py` actually calls now
- `local_llm/shared_prompt.py` -- the roster + instruction-prompt builder
  `qwen_local.py` uses, imported from `openai_analysis.py` via a dummy-key
  shim (see below)
- `local_llm/benchmark.py`, `local_llm/attention_viz.py`,
  `local_llm/FINDINGS.md`, `local_llm/README.md` -- the benchmark and
  attention-grounding analysis that justified using Qwen3-VL-4B for stage
  2; not part of the running app, but worth re-running if you change
  hardware or want to sanity-check match quality yourself (see "Running
  the benchmark/attention tooling directly" below)
- `openai_analysis.py` -- no longer makes any OpenAI calls (its
  `analyze_and_match()`/`_call_openai()` are unused dead code paths now),
  but it's still the one place the character roster
  (`CHARACTER_STYLE_GUIDE`) is defined, and `local_llm/shared_prompt.py`
  imports it for exactly that. **Don't delete this file** -- the local
  pipeline depends on it for the roster text, even though it never touches
  OpenAI anymore.
- `image_gen.py` -- poster generation via OpenAI `images.edit`, with a
  model/quality fallback (unchanged -- still the one cloud/OpenAI-dependent
  stage)
- `image_utils.py` -- shared photo downscale/re-encode helper, used by
  `image_gen.py` and by `app.py`'s stage-2 wrapper
- `compose.py` -- split-screen composition and QR generation
- `storage.py` -- Supabase Storage upload (with a fallback bucket/project)
  + list/delete helpers
- `index.html` -- front end: intro screen, camera/upload capture
  (client-side compressed), processing screen, poster reveal with QR +
  Instagram share prompt
- `static/characters/` -- optional folder for real character images; used
  by the frontend showcase (falls back to a generated placeholder card if
  a given character has no image)
- `testsupa.py` -- standalone script to sanity-check your Supabase
  credentials/bucket before running the full app (`python testsupa.py`)
- `gemini_analysis.py`, `emailer.py`, `vercel.json` -- **dead weight**,
  unused, kept only because they can't be deleted in this environment
  (nothing imports the two `.py` files, and there's no hosting step left
  to use `vercel.json` for). Safe to delete manually.

## Prerequisites

- Python 3.10+
- **An NVIDIA GPU with CUDA -- required, not optional.**
  `local_llm/qwen_local.py` hardcodes `device_map="cuda"`, so stage 2 will
  raise an error at the first capture if no CUDA device is visible to
  PyTorch. There's no CPU fallback in that file currently -- see "GPU /
  CUDA setup" below.
- Roughly **8GB free disk space** for the default `Qwen3-VL-4B-Instruct`
  model weights (or ~16GB for `Qwen3-VL-8B-Instruct` if you switch model
  sizes) -- downloaded automatically on first run to
  `~/.cache/huggingface/hub`.
- **VRAM**: ~10-12GB free is a comfortable minimum for the 4B model at
  `bfloat16` (roughly 8GB just for weights, plus activations/KV-cache for
  a ~5-6k token prompt -- the roster instructions plus the image itself).
  The 8B model needs roughly double that. This is what
  `local_llm/FINDINGS.md`'s benchmark numbers were measured at -- there's
  no quantization here, so these figures apply directly.

  If you hit a CUDA out-of-memory error, that's the first thing to
  check -- and check `nvidia-smi` for stray processes still holding GPU
  memory from a previous crashed/killed run before assuming your card is
  simply too small. If you need to run this on a smaller (e.g. 6-8GB
  laptop) GPU, a prior version of `local_llm/qwen_local.py` loaded the
  model through `bitsandbytes` (4bit/8bit) to fit that -- see git history
  if you need to bring that back.
- An [OpenAI](https://platform.openai.com) API key -- **only needed for
  stage 3** (poster generation via `images.edit`; paid, `gpt-image-2`
  specifically isn't available on the Free usage tier).
- A [Supabase](https://supabase.com) project with a **public** Storage
  bucket -- needed for stage 4's shareable poster/QR URLs. This is
  unrelated to "hosting the app"; it's just where the final images live so
  a visitor's phone can scan a QR code and actually see/download/share
  their poster, regardless of what machine is running the booth itself.

### GPU / CUDA setup

1. Confirm you have an NVIDIA GPU and a working driver:
   ```bash
   nvidia-smi
   ```
   If this isn't found or errors, install/update the NVIDIA driver for
   your GPU first (from NVIDIA's website or your OS's package manager) --
   everything below assumes this already works. The "CUDA Version" shown
   in `nvidia-smi`'s header is the newest CUDA runtime your *driver*
   supports, not something you need to separately install.
2. Install a CUDA-enabled build of PyTorch **before** installing the rest
   of `requirements.txt`. You don't need to install the full NVIDIA CUDA
   Toolkit separately for inference -- PyTorch's official wheels bundle the
   CUDA runtime libraries they need. Get the current recommended install
   command for your OS/CUDA version from
   [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)
   -- it typically looks like:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   ```
   (`cu124` above is an example -- use whatever CUDA version the PyTorch
   site currently recommends for your driver; a plain `pip install torch`
   can silently resolve to a CPU-only build depending on your platform, so
   don't skip the explicit index URL.)
3. Verify PyTorch can actually see your GPU:
   ```bash
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   This should print `True` followed by your GPU's name. If it prints
   `False`, stop here and fix that before continuing -- `qwen_local.py`
   will fail as soon as it tries to load the model otherwise.
4. Now install everything else:
   ```bash
   pip install -r requirements.txt
   ```
5. The first time stage 2 actually runs (see "Run the server" below), it
   downloads `Qwen/Qwen3-VL-4B-Instruct` from Hugging Face
   (~8GB) to `~/.cache/huggingface/hub` -- this can take a while depending
   on your connection, and only happens once.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # .venv\Scripts\activate on Windows
# install a CUDA build of torch first -- see "GPU / CUDA setup" above
pip install -r requirements.txt
```

Copy the example environment file and fill in the required values:

```bash
cp _env.example .env
```

In Supabase, create a Storage bucket (Storage -> New bucket) named to match
`SUPABASE_BUCKET` below, and mark it **public** -- the booth returns direct
public URLs for the result image and QR code rather than signed URLs.

## Environment variables

Create a `.env` file with values similar to (see `_env.example` for the
authoritative, commented version):

```env
OPENAI_API_KEY=your_openai_api_key

LOCAL_VLM_MODEL=4b
LOCAL_VLM_MAX_NEW_TOKENS=500

OPENAI_IMAGE_MODEL=gpt-image-2
OPENAI_IMAGE_MODEL_FALLBACK=gpt-image-1
OPENAI_IMAGE_SIZE=1024x1536
OPENAI_IMAGE_QUALITY=low
OPENAI_INPUT_MAX_DIM=1280
OPENAI_OUTPUT_FORMAT=jpeg
OPENAI_OUTPUT_COMPRESSION=85

SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_service_role_key
SUPABASE_BUCKET=movie-booth-uploads
SUPABASE_URL_FALLBACK=
SUPABASE_KEY_FALLBACK=
SUPABASE_BUCKET_FALLBACK=
SUPABASE_CLEANUP_AGE_HOURS=6

ARIES_INSTAGRAM_URL=https://www.instagram.com/aries.iitd/
ARIES_INSTAGRAM_HANDLE=@aries.iitd
```

There are no more `OPENAI_ANALYSIS_MODEL`/`OPENAI_REASONING_EFFORT`/
`GEMINI_*`/`GMAIL_*` variables -- stage 2 analysis moved from a hosted
OpenAI reasoning model to the local Qwen3-VL pipeline above, and emailing
was replaced with an Instagram share prompt well before that.

## Run the server

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/` in a browser to use the booth UI locally.
`127.0.0.1` (loopback-only) won't be reachable from another device on the
same network (a phone testing camera capture, say) -- use
`--host 0.0.0.0` and your machine's LAN IP if you need that.

**Warm up the model before opening the booth to real visitors.** The Qwen3-VL
model loads lazily on the first request that reaches stage 2, not at
server startup -- so the very first capture (or `/booth/test-analyze` call)
after starting `uvicorn` will be much slower than every one after it
(model load + first CUDA kernel compile). Send one throwaway request right
after starting the server so real visitors don't hit that cold start:

```bash
curl -X POST "http://127.0.0.1:8000/booth/test-analyze" -F "file=@static/characters/batman.jpg"
```

(Swap in any image you have handy -- this just needs to trigger the model
load, the actual match result doesn't matter.)

The frontend works on desktop (webcam) and mobile (front camera) via
`getUserMedia`, with a file-upload fallback if camera access is denied or
unavailable.

## API usage

Upload an image (this is what the front end does under the hood) -- the
response is streamed as newline-delimited JSON, one progress line at a
time, ending with a `"stage": "done"` (or `"stage": "failed"`) line:

```bash
curl -N -X POST "http://127.0.0.1:8000/booth/capture" \
  -F "file=@/path/to/photo.jpg"
```

Each line is a JSON object with a `"stage"` field:
- `{"stage": "matched", "character": ..., "reasoning": ..., "caption": ..., "traits": {...}}` -- sent as soon as local analysis finishes, before poster generation starts
- `{"stage": "done", "image_url": ..., "qr_url": ..., "character": ..., "reasoning": ..., "caption": ..., "traits": {...}, "timings": {...}}` -- the final result
- `{"stage": "failed", "error": ..., "timings": {...}}` -- if any stage raised

Get the current character roster (used by the front end to render the
showcase):

```bash
curl "http://127.0.0.1:8000/booth/characters"
```

Get the Instagram share config (also used by the front end):

```bash
curl "http://127.0.0.1:8000/booth/config"
```

Run analysis only, without generating a poster or touching storage --
useful for checking match quality and prompt output without spending
image-generation credits (also handy as the warm-up call above):

```bash
curl -X POST "http://127.0.0.1:8000/booth/test-analyze" \
  -F "file=@/path/to/photo.jpg"
```

> `/booth/test-analyze` and the "dev: test analysis only" toggle on the
> intro screen are temporary testing aids -- remove the endpoint, the
> toggle, and the `screen-test` block in `index.html` before the event.

Force an immediate storage cleanup sweep of the primary bucket (purely
on-demand -- nothing calls this automatically; see Notes):

```bash
curl -X POST "http://127.0.0.1:8000/booth/admin/cleanup"
```

## Running the benchmark/attention tooling directly (optional)

`local_llm/` has its own scripts for re-benchmarking model choice or
visualizing attention grounding, independent of running the full app --
see `local_llm/README.md` for full usage. Briefly:

```bash
cd local_llm
python benchmark.py --model 4b        # re-run the latency/accuracy benchmark
python attention_viz.py --model 4b --image ../static/characters/batman.jpg
```

`attention_viz.py` additionally needs `opencv-python` and `matplotlib`,
which aren't in the root `requirements.txt` since they're only used by
this optional tooling, not the running app:

```bash
pip install opencv-python matplotlib
```

## Notes

- The character roster is fixed in `openai_analysis.py`
  (`CHARACTER_STYLE_GUIDE`): Iron Man, Batman, Spider-Man, Thomas Shelby,
  Naruto, Professor from Money Heist, Gwen Stacy, Black Widow, Daenerys
  Targaryen, Hermione Granger, Wednesday Addams, Kakashi Hatake, Sasuke
  Uchiha, Joker, Jon Snow, Shinchan, Harry Potter, Dr. Doom, Regina George,
  Elle Woods, and Sheldon Cooper. Add a new character by adding one entry
  to `CHARACTER_STYLE_GUIDE`; the frontend showcase and `/booth/characters`
  pick it up automatically (via `local_llm/shared_prompt.py`, which both
  `qwen_local.py` and `app.py` read the roster from).
- **Matching has no fixed trait priority.** The prompt explicitly lists
  eight dimensions (facial resemblance, vibe/energy, complexion, gender
  presentation, facial hair, pose, clothing type, clothing color) with no
  ranking between them, and instructs the model to identify whichever
  single dimension is most distinctive for *this* photo rather than
  always defaulting to outfit color or glasses. The `reasoning` field
  names that standout dimension explicitly, and
  `detected_traits.standout_trait` surfaces it in the structured output
  too (visible via `/booth/test-analyze`).
- **The local model isn't schema-constrained the way the old OpenAI call
  was.** `qwen_local.py` asks for JSON via plain prompt text (no
  `response_format`/strict-schema enforcement), so `app.py`'s
  `_analyze_photo()` raises (turning the capture into a `"failed"` stage)
  if a response doesn't parse -- `local_llm/FINDINGS.md` measured 100%
  valid JSON across the 21-image benchmark set, but a live booth sees more
  variety than that sample, so keep an eye on the terminal.
- **No wasted work on oversized photos.** `image_utils.py` downscales the
  captured photo (default max dimension `1280`, shared by both the
  analysis call and the poster-edit call via `OPENAI_INPUT_MAX_DIM`) and
  applies `ImageOps.exif_transpose()` before resizing, so a phone photo
  stored "sideways" with an EXIF orientation tag doesn't come out rotated.
- **Client-side compression, server-side as fallback.** `index.html`
  compresses the captured/uploaded photo in the browser (canvas-based
  resize to the same `1280`px max dimension, JPEG quality 0.85) before
  it's ever uploaded, using `imageOrientation: "from-image"` so EXIF
  rotation is respected. If the browser doesn't support this, the original
  blob is uploaded as-is and `image_utils.py`'s server-side downscale still
  applies -- nothing breaks, it's just more bandwidth.
- `diffusion_prompt` output (from the local model) is intentionally kept
  short and dense (~50-70 words) rather than free-form, so less gets sent
  to the poster-edit call per request. It also explicitly instructs the
  model to preserve the subject's actual face, identity, and pose, pulling
  only costume/prop/backdrop details from the matched character -- this is
  a deliberate costume swap on the same person, not a redraw of a
  different one.
- Poster generation speed is controlled by two env vars:
  `OPENAI_IMAGE_QUALITY` (default `low`) and `OPENAI_INPUT_MAX_DIM`
  (default `1280`).
- `gpt-image-2` (default; override with `OPENAI_IMAGE_MODEL=gpt-image-1` to
  go back) is OpenAI's image model released April 2026, using the same
  `images.edit` endpoint/parameters as gpt-image-1. It isn't available on
  OpenAI's Free usage tier -- confirm your account tier supports it before
  relying on it (the automatic fallback to `gpt-image-1` at `low` quality
  covers this if it doesn't).
- **`OPENAI_IMAGE_QUALITY` is a much bigger lever on gpt-image-2 than it
  was on gpt-image-1.** gpt-image-2 runs an internal "agentic" reasoning
  pass (understand -> plan -> generate -> review) at higher quality tiers,
  so the four values behave very differently:
  - `low` (current default): ~3-8s, ~$0.006/image.
  - `medium`: ~20-40s, ~$0.053/image. Safe general-purpose choice if `low`
    ever looks too soft for your event.
  - `high`: ~150-235s (minutes, not seconds), ~$0.211/image. Not
    recommended for a live/synchronous booth.
  - `auto`: resolves to `medium` or `high` depending on prompt complexity
    -- avoid it here.

  (All figures at `1024x1024`; the 1536-series sizes this app uses add
  roughly another 1.5x latency at the same quality tier.)
- Both image models default to PNG output if you don't ask for anything
  else, which is lossless and noticeably bigger than a compressed
  photo-style poster needs to be -- `OPENAI_OUTPUT_FORMAT` (default
  `jpeg`) and `OPENAI_OUTPUT_COMPRESSION` (default `85`, range 0-100, only
  applies to `jpeg`/`webp`) switch the poster generation call itself to
  return a smaller JPEG directly, on top of the JPEG re-encoding
  `compose.py` already does for the final split-screen image one step
  later.
- Storage uploads require a valid, public Supabase bucket to return usable
  share URLs.
- **Fallback storage bucket instead of an auto-delete sweep.** `storage.py`
  doesn't rely on periodically deleting old files to stay under a storage
  quota -- `upload()` tries the primary bucket first and, if that raises
  for any reason (quota exceeded, transient error, bad credentials),
  retries once against a fallback bucket (`SUPABASE_BUCKET_FALLBACK`,
  optionally in a fully separate project via `SUPABASE_URL_FALLBACK`/
  `SUPABASE_KEY_FALLBACK`). All three vars are optional -- leave them unset
  and `upload()` behaves exactly as before, a failure just raises.
- **`POST /booth/admin/cleanup` is on-demand only, nothing runs it
  automatically.** Use it manually whenever you want a sweep of files
  older than `SUPABASE_CLEANUP_AGE_HOURS` (e.g. right after the event
  ends).
- **Instagram sharing, not email.** The reveal screen shows a prompt to
  share the poster to the visitor's own Instagram Story and tag the
  fest's official account -- `ARIES_INSTAGRAM_URL`/`ARIES_INSTAGRAM_HANDLE`,
  fetched by the frontend from `GET /booth/config`. The QR code points at
  `GET /share?img=<poster_url>` (see `app.py`), which shows the poster with
  two explicit buttons (**Download as JPEG**, which always works, and
  **Share to Instagram Story**, which uses the Web Share API's file-sharing
  support where available) rather than one do-everything button that could
  silently fail on an unsupported browser.
