# AI Movie Booth — final architecture, timing budget, hardware notes

## 1. The final pipeline

```
Capture photo (local, instant)
        |
CV trait extraction (local CPU, ~0.3-0.8s)
        |
Character matching (local, instant -> occasional LLM call ~1-2s)
        |
Prompt construction (local, negligible)
        |
Poster generation (cloud API, ~4-12s)
        |
Compose split-screen + upload + QR (local + upload, ~1-2s)
```

## 2. Hardware reality check

This changes almost nothing about stages 1, 2, 3, 4, and 6 — they were
never GPU-bound to begin with:

| Stage | What it needs | Runs on a bare laptop CPU? |
|---|---|---|
| Capture | webcam / USB camera | Yes — trivial |
| CV trait extraction | MediaPipe Pose + Face Mesh | **Yes.** MediaPipe's whole design goal is real-time CPU inference on phones. On a mid-range laptop CPU, a single-frame pose + face pass is 30-80ms each. |
| Character matching | a Python dict lookup | Yes — microseconds |
| Prompt construction | string formatting | Yes — microseconds |
| Poster generation | a diffusion model with several billion parameters | **No.** This needs a GPU. Since you don't have one, this step is a cloud API call, full stop. Don't try to self-host this on the day. |
| Compose + QR | PIL image compositing | Yes — tens of milliseconds |

So the hardware constraint doesn't change your architecture — it *confirms*
the architecture you already want: keep the cheap, deterministic stages
local, and send only the one genuinely heavy stage (diffusion) to a hosted
API. "No good storage" matters less than it sounds — you're never writing
large files to disk in this flow; images pass through memory (bytes/PIL
Images) and only touch disk if you choose to log them.

**Minimum laptop spec that's comfortable:** any quad-core CPU from the last
6-7 years, 8GB RAM, a stable internet connection (this matters more than the
CPU — you're waiting on a network round-trip for the slowest stage). A
2018 i5 ultrabook handles this fine. Wi-Fi quality at the venue is a bigger
risk than the laptop itself.

## 3. Timing budget — target vs. realistic

| Stage | Realistic time | Notes |
|---|---|---|
| Capture | 0-2s (human reaction, tap shutter) | not really "processing" time |
| CV trait extraction | 0.3-0.8s | two MediaPipe passes + numpy color math, all CPU |
| Character matching | 0.01s (rule path) or 1-2s (LLM tie-break path) | only ambiguous cases pay the LLM cost |
| Prompt construction | <0.01s | string building |
| Poster generation | 4-12s | dominated by network + queueing on the hosted API, not raw model speed |
| Compose + upload + QR | 1-2s | local compositing (~50ms) + two small uploads |
| **Total** | **~6-17s**, worst case ~20s | comfortably inside your 30s target, with margin for a slow API response |

Why poster generation is 4-12s and not "sub-second" even though some of
these models generate in under a second on the right hardware: that
sub-second figure is *inference time on a data-center GPU the provider
owns*. Your actual latency also includes queueing behind other users on a
shared endpoint, network round-trip to upload your photo and download the
result, and the extra steps needed for image-to-image/pose-conditioned
generation (which typically needs a few more sampling steps than a plain
text prompt, to actually preserve the person's identity and stance). Budget
for the API's real-world response time, not its benchmark number.

**Where your 30-second margin comes from:**
- A 20s hard timeout on the poster generation call (already in `image_gen.py`) so one slow request can't silently blow past the target — it fails fast instead, and you can show a "still working, hang tight" state or retry once.
- Async job handling in `app.py` means one person's slow diffusion call never blocks the *next* person's capture and CV analysis — those can queue up in parallel since they're separate CPU threads, not waiting on the same GPU.

## 4. Every piece of the architecture — what it is, why it's here, how it works

### MediaPipe Pose (BlazePose)
**What:** Google's lightweight pose estimation model, distilled specifically to run in real time on CPUs and phones (it powers pose features in the MediaPipe/ML Kit ecosystem).
**Why here:** You have no GPU. Most pose models (OpenPose, HRNet) assume one. BlazePose is architected the opposite way — small, fast, CPU-first — trading a bit of precision for speed you can actually get on your hardware.
**How it works:** A first lightweight detector locates the person in the frame, then a second regression network predicts 33 keypoints (shoulders, elbows, wrists, hips, etc.) directly as coordinates, without building a full heatmap per joint the way heavier models do — that heatmap-free approach is what makes it fast on CPU.

### MediaPipe Face Mesh
**What:** A 468-point face landmark model, same CPU-first design philosophy as BlazePose.
**Why here:** You need eye/face region coordinates for the glasses heuristic and roughly where the head is for hair-region cropping — you don't need a full face recognition system, just landmarks.
**How it works:** Similar two-stage design — detect the face region, then regress dense landmark coordinates in a single forward pass.

### Dominant color extraction (custom, no model)
**What:** A hand-rolled color quantization — downsample the torso crop, bucket pixels into coarse color bins, return the most populous bin's average color.
**Why here:** A full k-means clustering (scikit-learn) is overkill for "what color is this shirt" and pulls in a heavier dependency. Simple binning gets you a good-enough answer in single-digit milliseconds with zero extra libraries beyond numpy/Pillow, which you already need.
**How it works:** Divide each RGB channel into 32-level buckets, count how many pixels fall in each combined bucket, pick the winner, then average the *actual* pixel values in that bucket for a cleaner result than just returning the bucket's midpoint.

### Rule-based character scorer
**What:** A plain Python dict mapping character -> {trait: weight}.
**Why here:** This is the deterministic backbone of the whole matching step. It's free, instant, fully debuggable (you can point at exactly why "Iron Man" scored higher than "Superman" for a given photo), and it covers the large majority of cases without ever touching the network.
**How it works:** Extracted traits become a small set of active keys (e.g. `outfit_color:red`, `pose:arm_extended_right`). Each character's score is the sum of weights for the keys it has an entry for. Highest score wins, unless it's a near-tie.

### Claude Haiku (LLM tie-break)
**What:** Anthropic's fastest, smallest model in the Claude lineup.
**Why here:** This is the *only* place in the whole pipeline where genuine reasoning ("which of these two ambiguous matches is funnier/better here, and what's a witty caption") earns its latency cost. Using the smallest available model keeps that cost as low as possible — you don't need a large model's depth for an 8-12 word caption and a pick between two named options.
**How it works:** A single request/response call (not a chat loop, not tool use) with a tightly constrained prompt asking for JSON output only, so parsing is trivial and there's no multi-turn back-and-forth to add latency.

### ControlNet-style pose conditioning + image-to-image
**What:** A conditioning mechanism that lets a diffusion model generate a new image constrained by a reference pose (from your MediaPipe keypoints) and a reference photo (the person themselves), rather than generating purely from a text description.
**Why here:** Plain text-to-image would just generate *a* person in an Iron Man pose — not *this* person, and not necessarily *their* pose. You explicitly wanted the split-screen to show how *their* traits mapped onto the poster, so identity and pose fidelity matter. img2img (starting from their real photo, partially denoising) plus pose conditioning is what keeps both.
**How it works:** The pose keypoints are turned into a conditioning signal the model attends to during generation, steering the output's body position to match; the `strength` parameter controls how much of the original photo's structure survives versus how much the prompt/character styling overrides it — lower strength keeps more of them, higher strength leans more into the character.

### Fast/turbo-class diffusion model (Z-Image Turbo / FLUX schnell-klein / SD3.5 Turbo, hosted)
**What:** A distilled diffusion model that generates a usable image in a handful of sampling steps (often 1-8) instead of the 20-50 steps a standard diffusion model needs.
**Why here:** Standard diffusion models are too slow for your 30-second budget once you add network overhead on top. Turbo-class models exist specifically for this kind of latency-sensitive, user-facing use case.
**How it works (short version):** These models are trained with a "distillation" process where a smaller/faster model is taught to approximate what a full, many-step diffusion model would have produced, but in far fewer steps — trading a small amount of fine detail for a large speed gain. Hosted via a hosted inference API since you don't have the GPU to run it yourself.

### Async job queue (FastAPI + `asyncio`, not an agent framework)
**What:** A simple in-memory job dictionary and background tasks, no LangGraph/AutoGPT-style planning loop.
**Why here:** As discussed earlier, every step in this pipeline is a fixed, known sequence — there's nothing for an agent to *decide*. What you do need is concurrency: while one person's photo is waiting on the diffusion API (network-bound, several seconds), the next person's photo should already be running through pose/color extraction (CPU-bound) rather than sitting idle in a queue. `asyncio.to_thread` gets you that concurrency without any agentic complexity.
**How it works:** Each captured photo gets a job ID and runs as a background asyncio task through all six stages; the booth's front-end polls `/booth/status/{job_id}` until it flips to `done`, then shows the split-screen result and QR code.

### PIL compositing + `qrcode`
**What:** Plain image manipulation (no model) to build the two-panel image, and a standard QR code generator.
**Why here:** This is pure, deterministic image layout — pasting two images side by side, drawing dots for the skeleton, drawing a color swatch, and rendering a QR that points at wherever you uploaded the final image. No AI needed for any of it.
**How it works:** `qrcode` encodes the image URL as a standard QR matrix; PIL draws the skeleton dots by scaling the MediaPipe keypoint coordinates to the resized panel and placing small filled circles, straightforward affine scaling from the original photo's pixel space to the display panel's pixel space.

## 5. One thing worth deciding before the event
You'll need to pick an actual hosted diffusion provider and wire the real
request/response shape into `image_gen.py` (it's currently written against a
generic submit/poll contract that matches how most fast-inference hosts
work, but the exact field names differ per provider). Worth doing a timed
dry run with real venue Wi-Fi a day or two before the event — network
latency at the venue, not the model itself, is the biggest wildcard in your
30-second budget.
