# AI Movie Booth — new architecture (no local CV), timing budget

## 1. The new pipeline

```
Capture photo (local, instant)
        |
Gemini multimodal call: character match + caption + diffusion prompt + traits (cloud API, ~1-3s)
        |
Poster generation via gpt-image-1 images.edit (cloud API, ~5-15s)
        |
Compose split-screen + upload + QR (local + upload, ~1-2s)
```

This collapses the old six-stage pipeline (MediaPipe pose → MediaPipe face
→ color quantization → rule-based scorer → occasional LLM tie-break →
separate prompt-builder) into four stages. Everything that used to be
local computer vision plus a hand-tuned rule table is now one Gemini call
that looks at the photo directly and returns character, caption, prompt,
and traits together.

## 2. Why remove the local CV stack entirely

The old design existed because there was no GPU and MediaPipe is
CPU-first. That's still true, but it was solving the wrong problem for
what this call actually needs: MediaPipe gives you *coordinates* (keypoint
positions, landmark meshes) that then had to be turned into semantic tags
("arm_extended_right", "outfit_color:red") by hand-written geometric rules
before a character table could score them. A multimodal LLM call skips
that whole coordinate-to-semantics translation — it can just look at the
photo and say "arms crossed, red jacket, no glasses, this reads as Iron
Man" in one step. The trade-off: you lose the free, instant, fully local
nature of MediaPipe and gain a network dependency and a per-call cost on
every single photo (not just the ambiguous ones, like the old LLM
tie-break). That's an acceptable trade here because the booth already had
a hard network dependency anyway (poster generation always required
cloud), so this doesn't introduce a new failure mode, it just moves more
work behind an already-required network call.

## 3. Why Gemini 2.5 Flash-Lite (or 2.0 Flash) for stage 2

**What:** Google's smallest/fastest current Gemini tier, called through
the current `google-genai` SDK (not the deprecated `google-generativeai`
package, which is EOL and doesn't get new-model access going forward).
**Why here:** This call needs to be multimodal (photo in), fast (single
low-latency response, not a chat loop), and cheap enough to run on every
photo rather than only ambiguous ones — flash-lite is free-tier eligible
and built for exactly this shape of workload. `gemini-2.0-flash` is kept
as a drop-in env-var fallback since it's on the same free tier and same
SDK surface, in case flash-lite quota runs out mid-event.
**How it works:** One `client.models.generate_content()` call with the
photo passed as inline bytes (`types.Part.from_bytes`) alongside the
instruction text, using `response_mime_type="application/json"` and a
Pydantic `response_schema` (`BoothMatch`) so the SDK validates and parses
the structured output for you (`response.parsed`) instead of you having to
regex/strip code fences out of free-form text.

## 4. Why no reference images for the character roster

The request describes wanting 2-5 reference images per character so the
model has something concrete to match against. This implementation
deliberately uses short **text** style anchors instead
(`CHARACTER_STYLE_GUIDE` in `gemini_analysis.py`) for two reasons:

- All six characters (Batman, Iron Man, Joker, Spider-Man, Thomas Shelby,
  Naruto) are protected studio/franchise IP. Sourcing, storing, and
  repeatedly re-transmitting actual character artwork to two different
  third-party APIs adds a real licensing question that a photo-booth
  side project doesn't need to take on.
- It isn't a capability trade-off in practice: Gemini already knows what
  these characters look like from training, so a short prose description
  is enough to anchor the *style* of the match and the caption. And stage
  3 (`gpt-image-1`) only ever receives a **text** prompt anyway — it
  never sees reference character images either way — so the final poster
  quality is identical whether the style anchor lives in an image or in a
  sentence.

If you do want true reference-image grounding later, the clean way to add
it without a licensing question is to generate your own stylized
reference images once (e.g. with gpt-image-1 itself, prompted for "a hero
in a red-and-gold powered suit," never naming the franchise), cache those,
and pass them alongside the captured photo in the same Gemini call.

## 5. Why gpt-image-1's `images.edit` (not `images.generate`) for stage 3

**What:** OpenAI's current image model, used through the `images.edit`
endpoint, which takes an input image plus a prompt and modifies that
image, rather than `images.generate`, which only takes a text prompt and
produces an unrelated image from scratch.
**Why here:** The booth's whole premise is "you, styled as a character in
your own pose" — a text-only `generate` call would produce *a* person
wearing that costume, not the person in the photo. `edit` keeps the
original photo as the starting point, so pose and rough likeness carry
through into the styled result, which is the same goal the old
ControlNet+img2img design existed for, just reached through OpenAI's
editing endpoint instead of a pose-conditioned diffusion model.
**How it works:** The captured photo is wrapped as an in-memory file-like
object (no temp file on disk) and sent with Gemini's `diffusion_prompt` to
`client.images.edit(model="gpt-image-1", image=..., prompt=..., size=...)`.
The response returns base64-encoded image bytes (`b64_json`) by default,
which are decoded directly — no second network round-trip to fetch a
hosted URL is needed in the common case.

## 6. Timing budget — target vs. realistic

| Stage | Realistic time | Notes |
|---|---|---|
| Capture | 0-2s (human reaction, tap shutter) | not really "processing" time |
| Gemini analysis + match | 1-3s | single multimodal call, JSON-only response, small output |
| Poster generation (gpt-image-1 edit) | 5-15s | dominated by the model's generation time + network round-trip; image-to-image edits typically take a bit longer than a plain text-to-image call |
| Compose + upload + QR | 1-2s | local compositing (~50ms) + two small uploads |
| **Total** | **~7-22s**, worst case ~27s | inside the 30s target, tighter margin than before since stage 2 is now a network call too |

The margin against the 30s budget is a little tighter than the old
architecture's, because stage 2 moved from "sub-second local CPU work" to
"a network-bound API call." The 20s hard timeout on poster generation
(`app.py`) is unchanged and still the main safety valve — it's the
stage most likely to run long, and it fails fast into a `"failed"` job
status rather than hanging, same as before.

## 7. Async job handling — unchanged

The FastAPI + `asyncio` job-dict pattern, `asyncio.to_thread` for every
blocking call (both the Gemini call and the OpenAI call are blocking SDK
calls under the hood), and the poll-based `/booth/status/{job_id}` flow
are all unchanged from the previous design — none of that logic depended
on what stage 2 or stage 3 actually were internally, so there was nothing
here that needed to change when the CV stack was removed.

## 8. One thing worth deciding before the event

`OPENAI_IMAGE_SIZE` defaults to a portrait `1024x1536`, matching the
split-screen's poster panel orientation — check this against whatever
final aspect ratio you land on for the compose step, and adjust the env
var rather than hardcoding a different size in `image_gen.py`. As before,
a timed dry run with real venue Wi-Fi a day or two before the event is
worth doing — network latency, not the model itself, is still the
biggest wildcard in the 30-second budget.
