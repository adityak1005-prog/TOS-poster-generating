# Movie Booth

AI-powered photo booth app with a browser front end (camera capture or file
upload) that sends the photo straight to a multimodal LLM for analysis +
character matching, generates a poster-style image via an image-editing API,
and returns a shareable split-screen result with a witty catchphrase and the
reasoning behind the match.

## What it does

1. The front end (`index.html`) captures a photo -- either live via the
   browser camera or as a file upload -- compresses it client-side, and
   posts it to a FastAPI endpoint.
2. `POST /booth/capture` runs the whole pipeline in one streamed request
   (see "Why streaming instead of a job queue" below) and sends the photo
   to OpenAI in a single multimodal call that picks the best-matching
   character from a fixed roster, writes a short caption, writes the
   reasoning behind the match, writes a ready-to-use image-editing prompt,
   and reports a few quick visual traits -- all in one structured JSON
   response. This is streamed back to the browser immediately so the
   matched character name can be shown before the poster is even generated.
3. Sends the original photo + that prompt to OpenAI's `images.edit`
   endpoint to generate a poster-style version of the person, styled as
   the matched character.
4. Composes a split-screen image (original + poster), uploads it to cloud
   storage, and streams back the final URLs plus a QR image.
5. On the reveal screen, the visitor sees their poster, catchphrase, QR
   code, and "why you matched" reasoning, plus a prompt to share the poster
   on their Instagram Story and tag the fest's official account -- no
   email collection, nothing sent on the visitor's behalf.

### Why streaming instead of a job queue

Earlier versions of this app returned a `job_id` from `/booth/capture`
immediately and ran the pipeline in a background `asyncio` task, with the
frontend polling `/booth/status/{job_id}` until it saw `"done"`. That
doesn't survive on serverless hosts (Vercel, etc.): a bare
`asyncio.create_task()` can be torn down the instant the response is sent,
and even if it weren't, the in-memory job dict it wrote into isn't visible
to whichever instance handles the next poll request. `/booth/capture` now
runs the entire pipeline inline and streams newline-delimited JSON progress
within a single request/response -- the early "character matched" reveal
still happens (it's just the first streamed line instead of a separate
poll response), and there's no cross-request state to lose. See
`app.py`'s module docstring and `index.html`'s `runFullPipeline()`.

## Project files

- `app.py` — FastAPI API and orchestration (streamed capture, on-demand cleanup, config/admin endpoints)
- `openai_analysis.py` — single multimodal call: character match + reasoning + caption + diffusion prompt + trait read (OpenAI, replaces the old Gemini-based `gemini_analysis.py`)
- `image_gen.py` — poster generation via OpenAI `images.edit`, with a model/quality fallback
- `image_utils.py` — shared photo downscale/re-encode helper used by both the analysis call and the poster-edit call
- `compose.py` — split-screen composition and QR generation
- `storage.py` — Supabase Storage upload (with a fallback bucket/project) + list/delete helpers
- `index.html` — front end: intro screen, camera/upload capture (client-side compressed), processing screen, poster reveal with QR + Instagram share prompt
- `static/characters/` — optional folder for real character images; used by both the frontend showcase and the analysis stage's reference-image grounding (see Notes)
- `gemini_analysis.py`, `emailer.py` — **deprecated/unused**, kept only as stubs because they couldn't be deleted from this environment; safe to delete outright, nothing imports them

## Requirements

- Python 3.10+
- Internet access for both the OpenAI calls and storage uploads
- An [OpenAI](https://platform.openai.com) API key with chat + image API access (paid; `gpt-image-2` specifically isn't available on the Free usage tier)
- A [Supabase](https://supabase.com) project with a **public** Storage bucket

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Copy the example environment file and fill in the required values:

```bash
cp _env.example .env
```

In Supabase, create a Storage bucket (Storage -> New bucket) named to match
`SUPABASE_BUCKET` below, and mark it **public** — the booth returns direct
public URLs for the result image and QR code rather than signed URLs.

## Environment variables

Create a `.env` file with values similar to (see `_env.example` for the
authoritative, commented version):

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_ANALYSIS_MODEL=gpt-5-mini
OPENAI_REASONING_EFFORT=minimal

CHARACTER_REF_DIR=static/characters
OPENAI_REFERENCE_IMAGE_MAX_DIM=400
OPENAI_EXTENDED_CACHE_RETENTION=24h

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

There are no more `GEMINI_*` or `GMAIL_*` variables -- analysis moved to
OpenAI (see Notes) and emailing was replaced with an Instagram share prompt.

## Run the server

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/` in a browser to use the booth UI locally.
`127.0.0.1` (loopback-only) is for local development on your own machine --
it won't be reachable from another device on the same network (a phone
testing camera capture, say). If you need that during local testing, run
with `--host 0.0.0.0` instead and use your machine's LAN IP; this doesn't
affect a Vercel deployment either way (see "Hosting on Vercel" below),
which handles its own binding.

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
- `{"stage": "matched", "character": ..., "reasoning": ..., "caption": ..., "traits": {...}}` — sent as soon as analysis finishes, before poster generation starts
- `{"stage": "done", "image_url": ..., "qr_url": ..., "character": ..., "reasoning": ..., "caption": ..., "traits": {...}, "timings": {...}}` — the final result
- `{"stage": "failed", "error": ..., "timings": {...}}` — if any stage raised

Get the current character roster (used by the front end to render the
showcase, so it always stays in sync with `openai_analysis.py`):

```bash
curl "http://127.0.0.1:8000/booth/characters"
```

Get the Instagram share config (also used by the front end):

```bash
curl "http://127.0.0.1:8000/booth/config"
```

Run analysis only, without generating a poster or touching storage —
useful for checking match quality and prompt output without spending image
generation credits:

```bash
curl -X POST "http://127.0.0.1:8000/booth/test-analyze" \
  -F "file=@/path/to/photo.jpg"
```

> `/booth/test-analyze` and the "dev: test analysis only" toggle on the
> intro screen are temporary testing aids — remove the endpoint, the toggle,
> and the `screen-test` block in `index.html` before the event.

Rebuild the in-process reference-image set after adding/replacing images in
`static/characters/` (see Notes below) — `--reload` only restarts on `.py`
changes, so new image files need this to actually take effect:

```bash
curl -X POST "http://127.0.0.1:8000/booth/admin/rebuild-cache"
```

> `/booth/admin/rebuild-cache` is also a temporary dev aid — remove it
> before the event, or at least don't expose it publicly.

Force an immediate storage cleanup sweep of the primary bucket (purely
on-demand -- nothing calls this automatically; see Notes):

```bash
curl -X POST "http://127.0.0.1:8000/booth/admin/cleanup"
```

## Hosting on Vercel

1. Add a `vercel.json` at the project root:
   ```json
   { "functions": { "app.py": { "runtime": "python3.12", "maxDuration": 120 } } }
   ```
2. `requirements.txt` is already Vercel-clean (no `google-genai`).
3. Set every variable from `.env` in the Vercel dashboard (Project Settings
   -> Environment Variables), or via `vercel env add`.
4. `static/characters/` deploys with the repo and is served read-only --
   no extra config needed.
5. Deploy via `vercel --prod`, or connect the GitHub repo for auto-deploy
   on push.

This works cleanly now specifically because of two earlier architecture
changes: there's no `JOBS` dict or background `asyncio.create_task()` that
needs to survive past a single request (see "Why streaming instead of a
job queue" above), and there's no permanent background cleanup loop that
needs to keep running between requests (see Notes on storage cleanup) --
both of those patterns broke on serverless hosts, and both are gone now.

## Notes

- The character roster is fixed in `openai_analysis.py`
  (`CHARACTER_STYLE_GUIDE`): Iron Man, Batman, Spider-Man, Thomas Shelby,
  Naruto, Professor from Money Heist, Gwen Stacy, Black Widow, Daenerys
  Targaryen, Hermione Granger, Wednesday Addams, Kakashi Hatake, Sasuke
  Uchiha, Joker, Jon Snow, Shinchan, Harry Potter, Dr. Doom, Regina George,
  Elle Woods, and Sheldon Cooper. Each entry carries a longer style anchor
  (build, outfit/colors, expression, iconic prop, backdrop/lighting) plus an
  explicit facial-hair status, gender presentation, and a short contrasting
  "vibe" line -- added specifically because the matching model was prone to
  converging on the same few characters (e.g. matching any bearded person
  wearing glasses to Sheldon Cooper) when only outfit/prop details were
  available to compare against. Add a new character by adding one entry to
  `CHARACTER_STYLE_GUIDE`; the frontend showcase and `/booth/characters`
  pick it up automatically.
- **Matching has no fixed trait priority.** The analysis prompt explicitly
  lists eight dimensions (facial resemblance, vibe/energy, complexion,
  gender presentation, facial hair, pose, clothing type, clothing color)
  with no ranking between them, and instructs the model to identify
  whichever single dimension is most distinctive for *this* photo rather
  than always defaulting to outfit color or glasses. The `reasoning` field
  is required to name that standout dimension explicitly, and
  `detected_traits.standout_trait` surfaces it in the structured output too
  (visible via `/booth/test-analyze`) so you can actually verify matches
  are varying across different people instead of converging on a handful
  of "safe" characters.
- **Reference images + OpenAI's automatic prompt caching.**
  `static/characters/<slug>.jpg` serves double duty: the frontend showcase
  uses it if present (falling back to a generated placeholder card
  otherwise), and `openai_analysis.py` also picks up any image found there
  to strengthen the character match. Unlike Gemini's explicit
  `caches.create()` API (which this app used previously, and which free-tier
  keys are permanently blocked from using --
  `TotalCachedContentStorageTokensPerModelFreeTier` is hardcoded to `0`),
  OpenAI's prompt caching is automatic: any repeated, identical prefix over
  ~1024 tokens is served from cache at a steep discount, no cache object or
  TTL to manage. This only pays off if the static part of the request (the
  roster instructions + reference images) is sent byte-for-byte identical,
  in the same order, on every call, with the one thing that changes
  per-request (the captured photo) placed *after* it -- which is exactly
  how `_build_roster_content()`/`_call_openai()` construct the request.
  Slugs are lowercased with spaces/punctuation turned into underscores,
  e.g. `static/characters/professor_from_money_heist.jpg`,
  `static/characters/dr_doom.jpg`. If you add images, you (or whoever
  supplies them) are responsible for having the rights to use and display
  them -- this app doesn't source or vet them; see also the IP/licensing
  discussion in `ARCHITECTURE.md`.
- **Analysis has a fallback, and every stage logs to the terminal in real
  time.** If the full analysis request (roster + reference images + photo)
  raises, `analyze_and_match()` retries once with the text-only roster (no
  images) before giving up -- both the failure and the fallback are
  printed. Poster generation similarly retries once with
  `OPENAI_IMAGE_MODEL_FALLBACK`/`"low"` quality (gpt-image-1 at low by
  default) if the configured model/quality combination fails. Storage
  uploads retry once against the fallback bucket/project if configured
  (see below). None of these fallbacks are silent -- check the terminal for
  `[openai_analysis]`, `[image_gen]`, and `[storage]` log lines to see
  which path actually ran.
- **No wasted tokens on oversized photos.** `image_utils.py` downscales
  the captured photo (default max dimension `1280`, shared by both the
  analysis call and the poster-edit call via `OPENAI_INPUT_MAX_DIM`) and
  applies `ImageOps.exif_transpose()` before resizing, so a phone photo
  stored "sideways" with an EXIF orientation tag doesn't come out rotated.
  Previously only the poster-edit call downscaled -- the analysis call got
  the full-resolution original, meaning a large phone photo was effectively
  uploaded twice at full size for no accuracy benefit. The frontend also
  compresses client-side before upload (see below), so by the time a photo
  reaches either OpenAI call it's typically already small; the server-side
  downscale is a safety net, not the only line of defense.
- **Client-side compression, server-side as fallback.** `index.html`
  compresses the captured/uploaded photo in the browser (canvas-based
  resize to the same `1280`px max dimension, JPEG quality 0.85) before
  it's ever uploaded, using `imageOrientation: "from-image"` so EXIF
  rotation is respected. This matters beyond just speed: Vercel Functions
  cap request bodies at 4.5MB, and an uncompressed modern phone photo can
  exceed that on its own. If the browser doesn't support
  `createImageBitmap`'s orientation option (or compression otherwise
  fails), the original blob is uploaded as-is and `image_utils.py`'s
  server-side downscale still applies -- nothing breaks, it's just more
  bandwidth.
- `openai_analysis.py`'s `diffusion_prompt` output is intentionally kept
  short and dense (~50-70 words) rather than free-form, so less gets sent
  to the poster-edit call per request while keeping the same visual
  fidelity. It also explicitly instructs the model to preserve the
  subject's actual face, identity, and pose, pulling only costume/prop/
  backdrop details from the matched character -- this is a deliberate
  costume swap on the same person, not a redraw of a different one.
- Poster generation speed is controlled by two env vars: `OPENAI_IMAGE_QUALITY`
  (default `low`) and `OPENAI_INPUT_MAX_DIM` (default `1280`).
- `gpt-image-2` (default; override with `OPENAI_IMAGE_MODEL=gpt-image-1` to
  go back) is OpenAI's image model released April 2026, using the same
  `images.edit` endpoint/parameters as gpt-image-1. It isn't available on
  OpenAI's Free usage tier -- confirm your account tier supports it before
  relying on it (the automatic fallback to `gpt-image-1` at `low` quality
  covers this if it doesn't).
- **`OPENAI_IMAGE_QUALITY` is a much bigger lever on gpt-image-2 than it was
  on gpt-image-1.** gpt-image-2 runs an internal "agentic" reasoning pass
  (understand -> plan -> generate -> review) at higher quality tiers, so the
  four values behave very differently:
  - `low` (current default): ~3-8s, ~$0.006/image. Quality at this tier is
    reported as meaningfully better than previous-generation models' `low`
    tier -- chosen as the default here since booth throughput (a line of
    people, not one careful portrait) matters more than maximum polish.
  - `medium`: ~20-40s, ~$0.053/image. Safe general-purpose choice if `low`
    ever looks too soft for your event.
  - `high`: ~150-235s (minutes, not seconds), ~$0.211/image. Not recommended
    for a live/synchronous booth -- nobody in a fest queue should wait that
    long for one poster.
  - `auto`: resolves to `medium` or `high` depending on prompt complexity --
    avoid it here, since `diffusion_prompt` is a full descriptive paragraph
    that tends to push `auto` toward the slow `high` tier.

  (All figures at `1024x1024`; the 1536-series sizes this app uses add
  roughly another 1.5x latency at the same quality tier.)
- **`OPENAI_ANALYSIS_MODEL` defaults to `gpt-5-mini`**, not `gpt-4o` --
  switched specifically for latency (the analysis call was taking ~7s on
  `gpt-4o`). `gpt-5-mini` is OpenAI's cheapest vision-capable chat model as
  of mid-2026 and is still covered by automatic prompt caching (all
  GPT-5-series models support it). This is a real speed/cost trade rather
  than a strict upgrade -- override back to `gpt-4o` if match/reasoning
  quality seems to suffer.
  - `gpt-5-mini` is a **reasoning model**: it burns invisible "thinking"
    tokens before producing the visible JSON answer, and that thinking time
    doesn't show up in token counts, only in wall-clock latency. Left at its
    default effort this actually made analysis calls slower than `gpt-4o`
    (30+s). `OPENAI_REASONING_EFFORT` (default `minimal`) caps that -- raise
    to `low`/`medium` only if match/reasoning quality seems to need more
    deliberation. It also does not accept a custom `temperature` (only its
    default of `1`), so match variety comes entirely from the tie-break
    prompt language in `_build_instruction()`, not sampling temperature.
  - The bigger latency cost, independent of reasoning effort, was the 21
    character reference images sent inline on every analysis call that
    misses OpenAI's prompt cache (the cache window is only ~5-10min, so gaps
    between booth visitors mean most calls miss it). Those were previously
    read straight off disk at full source resolution. `openai_analysis.py`
    now downscales each one to `OPENAI_REFERENCE_IMAGE_MAX_DIM` (default
    `400`px) once at load time -- fine detail isn't needed for visual
    grounding, so this cuts the non-cached request size drastically.
  - `OPENAI_EXTENDED_CACHE_RETENTION` (default `24h`) asks OpenAI to hold
    the cached prefix for up to a day instead of its default ~5-10min
    window. This is only documented for the GPT-5.1/5.4/5.5 model family --
    it is **not confirmed to work on `gpt-5-mini`**. If the API rejects it,
    `openai_analysis.py` catches that specific failure once, disables it for
    the rest of the process, and prints a line starting `EXTENDED CACHE
    RETENTION NOT SUPPORTED`. Every per-request cache log line also states
    whether extended retention is currently active or off -- check the
    terminal to know for certain rather than assuming it's working. Set to
    an empty string to skip trying it at all.
- Both models default to PNG output if you don't ask for anything else,
  which is lossless and noticeably bigger than a compressed photo-style
  poster needs to be -- `OPENAI_OUTPUT_FORMAT` (default `jpeg`) and
  `OPENAI_OUTPUT_COMPRESSION` (default `85`, range 0-100, only applies to
  `jpeg`/`webp`) switch the poster generation call itself to return a
  smaller JPEG directly, on top of the JPEG re-encoding `compose.py` already
  does for the final split-screen image one step later.
- Storage uploads require a valid, public Supabase bucket to return
  usable share URLs.
- **Fallback storage bucket instead of an auto-delete sweep.** `storage.py`
  no longer relies on periodically deleting old files to stay under a
  storage quota -- instead, `upload()` tries the primary bucket first and,
  if that raises for any reason (quota exceeded, transient error, bad
  credentials), retries once against a fallback bucket (`SUPABASE_BUCKET_FALLBACK`,
  optionally in a fully separate project via `SUPABASE_URL_FALLBACK`/`SUPABASE_KEY_FALLBACK`).
  Note that Supabase's free-tier storage quota is per-*project*, not per-bucket
  -- setting only `SUPABASE_BUCKET_FALLBACK` (same project) adds resilience
  against a misconfigured bucket, but not real extra capacity; for that you
  need a genuinely separate project via the URL/KEY fallback vars too. All
  three vars are optional -- leave them unset and `upload()` behaves exactly
  as before, a failure just raises.
- **`POST /booth/admin/cleanup` is on-demand only, nothing runs it
  automatically.** There used to be a permanent background sweep
  (`_periodic_cleanup_loop`, started on server startup) that deleted
  uploads older than `SUPABASE_CLEANUP_AGE_HOURS`. That's gone now, for two
  reasons: the fallback-bucket approach above is a better fit for staying
  under a storage quota than deleting user content, and a `while True`
  background loop can't survive on a serverless host anyway (an instance
  can be paused/recycled between requests, killing the loop along with
  it). Use `/booth/admin/cleanup` manually whenever you want a sweep (e.g.
  right after the event ends).
- **Instagram sharing, not email.** The reveal screen no longer collects an
  email address. Instead it shows a prompt to share the poster to the
  visitor's own Instagram Story and tag the fest's official account --
  `ARIES_INSTAGRAM_URL`/`ARIES_INSTAGRAM_HANDLE`, fetched by the frontend
  from `GET /booth/config`. This is a manual, visitor-driven share (there's
  no server-side Instagram API integration): posting to an individual
  visitor's own personal Instagram account isn't something Meta's API
  permits for third-party apps, and getting `instagram_content_publish`
  approved for even a single official account takes Meta 2-4 weeks of App
  Review, which usually doesn't fit a college-fest timeline. The QR code
  points at `GET /share?img=<poster_url>` (see `app.py`) instead of the raw
  poster image directly. That page shows the poster with two explicit
  buttons rather than one do-everything button, since a single "Share"
  button silently failing on an unsupported browser left visitors with no
  way to get their poster at all:
  - **Download as JPEG** -- fetches the image, builds a `Blob`/object URL,
    and triggers a save via a temporary `<a download>` link. Works on every
    modern mobile/desktop browser regardless of Web Share API support.
  - **Share to Instagram Story** -- uses the Web Share API's file-sharing
    support (`navigator.canShare({files:...})` / `navigator.share`) to bring
    up the phone's native share sheet with the poster attached, so Instagram
    appears as a share target one tap away. On browsers without file-sharing
    support (mainly desktop), it shows a plain-text instruction to use the
    Download button instead and upload manually, rather than silently doing
    nothing.
- **Found and fixed in passing:** `storage.py` was importing
  `from testsupa import create_client` instead of `from supabase import
  create_client`. `testsupa.py` is a standalone script for manually testing
  Supabase credentials (`python testsupa.py`) and has top-level code that
  performs a real upload and calls `exit(1)` if env vars are missing --
  since Python runs a module's top-level code on import, every
  `import storage` in `app.py` was executing that test script as a side
  effect, including on every server start. Fixed to import from the real
  `supabase` package; `storage.upload()`'s core behavior is unchanged.
