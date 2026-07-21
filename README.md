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
3. Sends the original photo + that prompt to OpenAI's gpt-image-1
   (`images.edit`) to generate a poster-style version of the person,
   styled as the matched character.
4. Composes a split-screen image (original + poster), uploads it to cloud
   storage, and returns URLs plus a QR image.
5. The front end polls job status, showing the matched character name as
   soon as Gemini responds, then the full poster, catchphrase, and "why you
   matched" reasoning once the poster is ready.

## Project files

- `app.py` — FastAPI API and orchestration
- `gemini_analysis.py` — single multimodal call: character match + reasoning + caption + diffusion prompt + trait read (Gemini)
- `image_gen.py` — poster generation via OpenAI gpt-image-1 image-to-image edit
- `compose.py` — split-screen composition and QR generation
- `storage.py` — uploads images to Supabase Storage
- `index.html` — front end: intro screen, camera/upload capture, processing screen, poster reveal
- `static/characters/` — optional folder for real character showcase images (see Notes)

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

OPENAI_API_KEY=your_openai_api_key
OPENAI_IMAGE_MODEL=gpt-image-1
OPENAI_IMAGE_SIZE=1024x1536

SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_service_role_key
SUPABASE_BUCKET=movie-booth-uploads
```

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
character, reasoning, caption, detected traits, and timing information.

Get the current character roster (used by the front end to render the
showcase, so it always stays in sync with `gemini_analysis.py`):

```bash
curl "http://localhost:8000/booth/characters"
```

Run Gemini analysis only, without generating a poster or touching storage —
useful for checking match quality and prompt output without spending
gpt-image-1 credits:

```bash
curl -X POST "http://localhost:8000/booth/test-analyze" \
  -F "file=@/path/to/photo.jpg"
```

> `/booth/test-analyze` and the "dev: test analysis only" toggle on the
> intro screen are temporary testing aids — remove the endpoint, the toggle,
> and the `screen-test` block in `index.html` before the event.

## Notes

- The character roster is fixed in `gemini_analysis.py`
  (`CHARACTER_STYLE_GUIDE`): Batman, Iron Man, Joker, Spider-Man, Thomas
  Shelby, Naruto, Black Widow, and Barbie. Each has a longer text style
  anchor (build, outfit/colors, expression, iconic prop, backdrop/lighting —
  not a reference image, see `ARCHITECTURE.md` for why) that both Gemini's
  match/reasoning and the poster prompt lean on. Add a new character by
  adding one entry to `CHARACTER_STYLE_GUIDE`; the frontend showcase and
  `/booth/characters` pick it up automatically.
- The front-end character showcase uses generated placeholder cards
  (color + emoji) rather than real character stills, since studio/franchise
  images carry licensing questions. Drop a real image into
  `static/characters/<slug>.jpg` (e.g. `static/characters/batman.jpg`,
  spaces/punctuation in the name become underscores) and the showcase will
  use it automatically instead of the placeholder — no code changes needed.
- Gemini's `diffusion_prompt` output is intentionally kept short and dense
  (~50-70 words) rather than free-form, so less gets sent to OpenAI per
  request while keeping the same visual fidelity.
- `POSTER_TIMEOUT_S` in `app.py` is currently defined but not enforced —
  `generate_poster` runs without a hard timeout, so a slow OpenAI call will
  make the job take longer rather than fail fast. Wrap the call in
  `asyncio.wait_for` if you want that safety valve back.
- Storage uploads require a valid, public Supabase bucket to return
  usable share URLs.
