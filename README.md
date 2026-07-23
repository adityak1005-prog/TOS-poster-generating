# Movie Booth

AI-powered photo booth app with a browser front end (camera capture or file
upload) that sends the photo straight to a multimodal LLM for analysis +
character matching, generates a poster-style image via an image-editing API,
and returns a shareable split-screen result with a witty catchphrase and the
reasoning behind the match.

## What it does

1. The front end (`index.html`) captures a photo -- either live via the
   browser camera or as a file upload -- and posts it to a FastAPI endpoint.
2. Sends the photo to Gemini in a single multimodal call that picks the
   best-matching character from a fixed roster, writes a short caption,
   writes the reasoning behind the match, writes a ready-to-use image-editing
   prompt, and reports a few quick visual traits -- all in one structured
   JSON response.
3. Sends the original photo + that prompt to OpenAI's gpt-image-2
   (`images.edit`) to generate a poster-style version of the person,
   styled as the matched character.
4. Composes a split-screen image (original + poster), uploads it to cloud
   storage, and returns URLs plus a QR image.
5. The front end polls job status, showing the matched character name as
   soon as Gemini responds, then the full poster, catchphrase, QR code, and
   "why you matched" reasoning once the poster is ready.
6. On the reveal screen, the visitor can optionally type their email and
   hit "email me this poster" -- this fires a background send (fetches the
   already-uploaded poster from its Supabase URL and emails it) that never
   blocks the UI or the next visitor's capture. Nothing is emailed unless
   this button is clicked.
7. In the background, a periodic sweep deletes uploaded posters/QR codes
   from Supabase once they're older than a configurable age, so storage
   never grows unbounded over an event (see Notes).

## Project files

- `app.py` — FastAPI API and orchestration, background storage cleanup
- `gemini_analysis.py` — single multimodal call: character match + reasoning + caption + diffusion prompt + trait read (Gemini)
- `image_gen.py` — poster generation via OpenAI gpt-image-2 image-to-image edit
- `compose.py` — split-screen composition and QR generation
- `storage.py` — Supabase Storage upload + list/delete helpers (for cleanup)
- `emailer.py` — Gmail SMTP send of the finished poster, triggered from the reveal screen
- `index.html` — front end: intro screen, camera/upload capture, processing screen, poster reveal with QR + "email me" button
- `static/characters/` — optional folder for real character images; used by both the frontend showcase and Gemini's reference-image cache (see Notes)

## Requirements

- Python 3.10+
- Internet access for both LLM calls and storage uploads
- A [Google AI Studio](https://aistudio.google.com) API key (Gemini, free tier)
- An [OpenAI](https://platform.openai.com) API key with image API access (paid)
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

Create a `.env` file with values similar to:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash-lite
CHARACTER_REF_DIR=static/characters
GEMINI_CACHE_TTL=86400s

OPENAI_API_KEY=your_openai_api_key
OPENAI_IMAGE_MODEL=gpt-image-2
OPENAI_IMAGE_SIZE=1024x1536
OPENAI_IMAGE_QUALITY=medium
OPENAI_INPUT_MAX_DIM=1280
OPENAI_OUTPUT_FORMAT=jpeg
OPENAI_OUTPUT_COMPRESSION=85

SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_service_role_key
SUPABASE_BUCKET=movie-booth-uploads
SUPABASE_CLEANUP_ENABLED=false
SUPABASE_CLEANUP_AGE_HOURS=6
SUPABASE_CLEANUP_INTERVAL_S=3600

GMAIL_ADDRESS=your_gmail_address@gmail.com
GMAIL_APP_PASSWORD=your_16_char_app_password
EMAIL_FROM_NAME=AI Movie Booth
```

`GMAIL_APP_PASSWORD` is not your regular Gmail password. To generate one:

1. Turn on 2-Step Verification first -- app passwords don't exist without
   it: `myaccount.google.com` -> Security -> "2-Step Verification" -> On.
2. Go to https://myaccount.google.com/apppasswords (re-authenticate if asked).
3. Give it a name (e.g. "Movie Booth") and click Generate.
4. Google shows a 16-character password once -- copy it straight into
   `GMAIL_APP_PASSWORD` before closing the window; it can't be viewed again
   afterward (you'd generate a new one instead).

Don't enable Advanced Protection on that Google account -- it disables app
passwords entirely. If `GMAIL_ADDRESS`/`GMAIL_APP_PASSWORD` aren't set (or
sending fails for any other reason), the job still completes normally with
`email_sent: false` in the status response -- check the server log for the
traceback if emails aren't going out.

## Run the server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/` in a browser to use the actual booth UI. It
works on desktop (webcam) and mobile (front camera) via `getUserMedia`, with
a file-upload fallback if camera access is denied or unavailable.

## API usage

Upload an image (this is what the front end does under the hood):

```bash
curl -X POST "http://localhost:8000/booth/capture" \
  -F "file=@/path/to/photo.jpg"
```

Check the job status:

```bash
curl "http://localhost:8000/booth/status/<job_id>"
```

The status response includes the job state, generated URLs, matched
character, reasoning, caption, detected traits, whether an email was sent
(`email_sent` -- `false` until the visitor uses the mail button, `null`
while a send is in flight), and timing information.

Once a job is `"done"`, email the poster to a given address (this is what
the reveal screen's "email me" button does under the hood -- returns
immediately, the actual send happens in a background task):

```bash
curl -X POST "http://localhost:8000/booth/email/<job_id>" \
  -F "email=someone@example.com"
```

Get the current character roster (used by the front end to render the
showcase, so it always stays in sync with `gemini_analysis.py`):

```bash
curl "http://localhost:8000/booth/characters"
```

Run Gemini analysis only, without generating a poster or touching storage —
useful for checking match quality and prompt output without spending
gpt-image-2 credits:

```bash
curl -X POST "http://localhost:8000/booth/test-analyze" \
  -F "file=@/path/to/photo.jpg"
```

> `/booth/test-analyze` and the "dev: test analysis only" toggle on the
> intro screen are temporary testing aids — remove the endpoint, the toggle,
> and the `screen-test` block in `index.html` before the event.

Rebuild the Gemini character-reference cache after adding/replacing images in
`static/characters/` (see Notes below) — `--reload` only restarts on `.py`
changes, so new image files need this to actually take effect:

```bash
curl -X POST "http://localhost:8000/booth/admin/rebuild-cache"
```

> `/booth/admin/rebuild-cache` is also a temporary dev aid — remove it
> before the event, or at least don't expose it publicly.

Force an immediate storage cleanup sweep (the automatic timer is off by
default -- see Notes below; this always works regardless, for an on-demand
run e.g. right after the event ends):

```bash
curl -X POST "http://localhost:8000/booth/admin/cleanup"
```

## Notes

- The character roster is fixed in `gemini_analysis.py`
  (`CHARACTER_STYLE_GUIDE`): Iron Man, Batman, Spider-Man, Thomas Shelby,
  Naruto, Professor from Money Heist, Gwen Stacy, Black Widow, Daenerys
  Targaryen, Hermione Granger, Wednesday Addams, Kakashi Hatake, Sasuke
  Uchiha, Joker, Jon Snow, Shinchan, Harry Potter, Dr. Doom, Regina George,
  Elle Woods, and Sheldon Cooper. Each has a longer text style anchor
  (build, outfit/colors, expression, iconic prop, backdrop/lighting) that
  both Gemini's match/reasoning and the poster prompt lean on. Add a new
  character by adding one entry to `CHARACTER_STYLE_GUIDE`; the frontend
  showcase and `/booth/characters` pick it up automatically.
- **Reference images + Gemini context caching.** `static/characters/<slug>.jpg`
  serves double duty: the frontend showcase uses it if present (falling back
  to a generated placeholder card otherwise), and `gemini_analysis.py` also
  picks up any image found there to strengthen the character match — the
  image plus its style-guide text get baked into a Gemini context cache
  once, and every `analyze_and_match` call afterward references that cache
  instead of re-uploading/re-encoding the same images on every single
  capture. This is fully additive: with zero images present, nothing is
  cached and the booth behaves exactly like the text-only version. Slugs
  are lowercased with spaces/punctuation turned into underscores, e.g.
  `static/characters/professor_from_money_heist.jpg`,
  `static/characters/dr_doom.jpg`. If you add images, you (or whoever
  supplies them) are responsible for having the rights to use and display
  them — this app doesn't source or vet them; see also the IP/licensing
  discussion in `ARCHITECTURE.md`. Cache build failures (e.g. below Gemini's
  explicit-caching minimum of ~32,768 tokens across the combined
  images+text, or an SDK/model that doesn't support caching) fail soft back
  to the original text-only behavior rather than breaking a capture — check
  the server log for `[gemini_analysis] character cache build failed` if a
  cache doesn't seem to be taking effect. `GEMINI_CACHE_TTL` (default
  `86400s`/24h) is as close to "no time limit" as the API allows -- there's
  no true unlimited option, only a duration, but there's also no documented
  hard maximum and storage cost at this content size is trivial even at
  much longer TTLs. Either way, `analyze_and_match()` self-heals if a cache
  ever expires or becomes invalid mid-event: it drops the dead cache,
  retries that one request without it so the capture in progress still
  succeeds, and the next call rebuilds a fresh cache automatically.
- Gemini's `diffusion_prompt` output is intentionally kept short and dense
  (~50-70 words) rather than free-form, so less gets sent to OpenAI per
  request while keeping the same visual fidelity.
- Poster generation speed is controlled by two env vars in `image_gen.py`:
  `OPENAI_IMAGE_QUALITY` (default `medium`) and `OPENAI_INPUT_MAX_DIM`
  (default `1280` -- the captured photo is downscaled to this max dimension
  and re-encoded as JPEG before being sent to OpenAI's edit call, cutting
  upload size and input-processing time for large phone photos). Only the
  copy sent to OpenAI is downscaled; the original photo used in the
  split-screen compose and storage upload is untouched.
- `gpt-image-2` (default; override with `OPENAI_IMAGE_MODEL=gpt-image-1` to
  go back) is OpenAI's image model released April 2026, using the same
  `images.edit` endpoint/parameters as gpt-image-1 -- drop-in swap, no other
  code changes needed. It isn't available on OpenAI's Free usage tier --
  confirm your account tier supports it before relying on it.
- **`OPENAI_IMAGE_QUALITY` is a much bigger lever on gpt-image-2 than it was
  on gpt-image-1.** gpt-image-2 runs an internal "agentic" reasoning pass
  (understand -> plan -> generate -> review) at higher quality tiers, so the
  four values behave very differently:
  - `low`: ~3-8s, ~$0.006/image. Quality at this tier is reported as
    meaningfully better than previous-generation models' `low` tier -- worth
    trying for a booth where throughput through a line of people matters
    more than maximum polish.
  - `medium` (current default): ~20-40s, ~$0.053/image. Safe general-purpose
    choice.
  - `high`: ~150-235s (minutes, not seconds), ~$0.211/image. Not recommended
    for a live/synchronous booth -- nobody in a fest queue should wait that
    long for one poster.
  - `auto`: resolves to `medium` or `high` depending on prompt complexity --
    avoid it here, since `diffusion_prompt` is a full descriptive paragraph
    that tends to push `auto` toward the slow `high` tier. The explicit
    `medium` default in `_env.example`/`.env` already avoids this trap.
  
  (All figures at `1024x1024`; the 1536-series sizes this app uses add
  roughly another 1.5x latency at the same quality tier.) If event-day speed
  becomes the priority, switching `OPENAI_IMAGE_QUALITY=low` is the single
  biggest available speedup.
- Both models default to PNG output if you don't ask for anything else,
  which is lossless and noticeably bigger than a compressed photo-style
  poster needs to be -- `OPENAI_OUTPUT_FORMAT` (default `jpeg`) and
  `OPENAI_OUTPUT_COMPRESSION` (default `85`, range 0-100, only applies to
  `jpeg`/`webp`) switch the poster generation call itself to return a
  smaller JPEG directly, on top of the JPEG re-encoding `compose.py` already
  does for the final split-screen image one step later.
- `POSTER_TIMEOUT_S` in `app.py` is currently defined but not enforced —
  `generate_poster` runs without a hard timeout, so a slow OpenAI call will
  make the job take longer rather than fail fast. Wrap the call in
  `asyncio.wait_for` if you want that safety valve back.
- Storage uploads require a valid, public Supabase bucket to return
  usable share URLs.
- **Email is opt-in, triggered after the reveal -- not during capture.**
  Nothing is emailed automatically. The visitor sees their poster + QR
  first, then can type an address and click "email me this poster."
  `POST /booth/email/{job_id}` validates the address, schedules the actual
  fetch-from-Supabase-and-send as a background `asyncio` task, and returns
  immediately -- the click never blocks the browser, and because the send
  runs via `asyncio.to_thread` inside that background task, it never blocks
  the server's event loop either, so a concurrent `/booth/capture` for the
  next visitor is unaffected. A bad address or a Gmail send error just
  leaves `email_sent: false` (with an `email_error` message) in the status
  response -- the frontend does one follow-up status check ~4s after
  queuing to surface a failure and let the visitor retry.
- **Storage cleanup is opt-in, off by default.** Set
  `SUPABASE_CLEANUP_ENABLED=true` to turn on the automatic background sweep
  in `app.py`, which deletes any uploaded poster/QR file from Supabase once
  it's older than `SUPABASE_CLEANUP_AGE_HOURS` (default 6), checked every
  `SUPABASE_CLEANUP_INTERVAL_S` (default hourly) -- useful if you expect
  enough volume to approach the free tier's 1GB storage limit. With it left
  off (the default), nothing gets auto-deleted; `POST /booth/admin/cleanup`
  is still available any time you want to force a manual sweep (e.g. right
  after the event ends), independent of this setting.
- Gmail SMTP specifically rate-limits outbound mail to roughly 500
  messages/day per account, and attachments from a personal Gmail account
  are somewhat more likely to land in spam than mail from a dedicated
  transactional email provider (Resend, SendGrid, etc.) -- an acceptable
  trade-off for zero-signup, free sending at college-fest scale, but worth
  knowing if you outgrow it.
- **Found and fixed in passing:** `storage.py` was importing
  `from testsupa import create_client` instead of `from supabase import
  create_client`. `testsupa.py` is a standalone script for manually testing
  Supabase credentials (`python testsupa.py`) and has top-level code that
  performs a real upload and calls `exit(1)` if env vars are missing --
  since Python runs a module's top-level code on import, every
  `import storage` in `app.py` was executing that test script as a side
  effect, including on every server start. Fixed to import from the real
  `supabase` package; `storage.upload()`'s behavior is unchanged.
