# Movie Booth

AI-powered photo booth app that sends a captured photo straight to a
multimodal LLM for analysis + character matching, generates a poster-style
image via an image-editing API, and returns a shareable split-screen result.

## What it does

1. Accepts a photo upload through a FastAPI endpoint.
2. Sends the photo to Gemini in a single multimodal call that picks the
   best-matching movie character, writes a caption, writes a ready-to-use
   image-editing prompt, and reports a few quick visual traits -- all in
   one structured JSON response.
3. Sends the original photo + that prompt to OpenAI's gpt-image-1
   (`images.edit`) to generate a poster-style version of the person,
   styled as the matched character.
4. Composes a split-screen image (original + poster), uploads it to cloud
   storage, and returns URLs plus a QR image.

## Project files

- `app.py` — FastAPI API and orchestration
- `gemini_analysis.py` — single multimodal call: character match + caption + diffusion prompt + trait read (Gemini)
- `image_gen.py` — poster generation via OpenAI gpt-image-1 image-to-image edit
- `compose.py` — split-screen composition and QR generation
- `storage.py` — uploads images to Supabase Storage

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

## API usage

Upload an image:

```bash
curl -X POST "http://localhost:8000/booth/capture" \
  -F "file=@/path/to/photo.jpg"
```

Check the job status:

```bash
curl "http://localhost:8000/booth/status/<job_id>"
```

The status response includes the job state, generated URLs, matched
character, caption, detected traits, and timing information.

## Notes

- The character roster is fixed in `gemini_analysis.py`
  (`CHARACTER_STYLE_GUIDE`): Batman, Iron Man, Joker, Spider-Man, Thomas
  Shelby, Naruto. Each has a short text style anchor (not a reference
  image — see `ARCHITECTURE.md` for why) that both Gemini and the poster
  prompt lean on for a consistent look.
- Poster generation depends on OpenAI's `images.edit` availability/quota
  for your account and may take several seconds; it's wrapped in a hard
  20s timeout so a slow call fails fast instead of hanging the booth.
- Storage uploads require a valid, public Supabase bucket to return
  usable share URLs.
