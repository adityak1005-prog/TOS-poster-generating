# Findings: Qwen3-VL as a local replacement for stage 2, + attention grounding

## 1. Benchmark: is a local VLM decent enough to bother with?

Ran both cached model sizes over all 21 `static/characters/` images (see
`README.md` for why this benchmark's "self-match rate" is a sanity check,
not a real accuracy number -- there's no dataset of real booth selfies to
test true person-to-character matching against).

| Model | Valid JSON | Self-match rate | Mean latency | Tokens/s |
|---|---|---|---|---|
| Qwen3-VL-4B-Instruct | 100% (21/21) | 95.2% (20/21) | 6.39s | 26.9 |
| Qwen3-VL-8B-Instruct | 100% (21/21) | 95.2% (20/21) | 6.46s | 26.0 |

Both sizes made the exact same single "miss" (Regina George -> Gwen
Stacy) -- the roster's own docs already flag these two as a deliberately
close pair (even two-sided smirk vs. one-sided smirk+head-tilt), so this
reads as a genuinely hard case rather than a model weakness.

**Verdict: decent, and 4B is the practical pick.** 4B matches 8B on both
axes, so there's no reason to pay 8B's extra VRAM/load time here. Mean
latency (6.4s, on an idle A100 slice, `sdpa` attention) is in the same
ballpark as the *deployed app's* documented OpenAI baseline (~7s on
`gpt-4o`, which is why that app switched to `gpt-5-mini` at
`reasoning_effort=minimal` in the first place) -- so a local 4B model is a
believable drop-in for this specific stage, latency-wise, on hardware
like this. (Caveat: this machine has an idle A100; a real deployment
would need to size hardware/concurrency for booth throughput, not a
single warm GPU.)

Full per-image results: `results/benchmark_4b.json`, `results/benchmark_8b.json`.

## 2. Attention visualization: is there real grounding?

**Method** (see `attention_viz.py` docstring for full detail): captured
real attention weights during generation via forward hooks on every
`Qwen3VLTextAttention` layer (not a post-hoc saliency proxy like
Grad-CAM), reduced to attention-on-image-tokens immediately per layer to
avoid materializing full (heads x 5700 x 5700) matrices, reshaped to the
image's ViT patch grid, and overlaid on the original photo. Two things
had to be corrected for after the first look at the raw maps:

- **Attention sinks.** Raw maps were dominated by 1-2 grid cells carrying
  up to ~80% of the total attention mass (see `sink_diagnostics()`) --
  the well-documented "attention sink" / "register token" phenomenon,
  unrelated to image content. Fixed by winsorizing the top 1% before
  visualizing/measuring ("sink-robust" variant); both raw and sink-robust
  numbers are reported for transparency.
- **Face-box grounding metric.** Used OpenCV's Haar cascade to find a
  face box per image (worked on 15/21 -- correctly failed on masked
  characters like Iron Man/Spider-Man/Dr. Doom, and on the 2 anime-style
  images, as expected), then computed `attention mass inside the face
  box / area the box covers` ("enrichment": >1 means the model attends to
  the face more than its size alone would predict).

**Result -- mean face-attention enrichment, sink-robust, n=15 images:**

| Generated span | Enrichment | What it means |
|---|---|---|
| `reasoning` sentence | **2.07x** (0.36-3.77) | strongest grounding |
| `standout_trait` | **1.80x** (0.23-3.06) | |
| `character` name | **1.54x** (0.33-2.56) | |
| whole response (all layers avg) | 1.23x (0.47-2.08) | diluted by costume/JSON-syntax tokens |
| whole response (**last layer only**) | 0.96x (~chance) | see note below |

**This is real grounding, not noise**, and it's strongest exactly where
you'd hope: the `reasoning` text (the field that explicitly names a
visual cue) is the most face-attentive, while the full response average
(mostly `diffusion_prompt` costume/backdrop text and JSON punctuation) is
close to chance. Spot-checking individual maps against each character's
own "MATCH ON" text in `openai_analysis.py` is the most convincing part:

- **Jon Snow** (`reasoning`): hotspot sits directly on the beard/jaw --
  his roster entry's deciding cue is literally "visible beard or heavy
  stubble covering the jaw."
- **Kakashi Hatake** (`reasoning`): hotspot sits on the visible eyes above
  the mask -- his entry's cue is "heavy, half-lidded or sleepy-looking
  eyes." (His raw face-box enrichment number is actually *low* --
  because the Haar box includes his fabric-masked lower face, which gets
  ~no attention, diluting the box-average metric even though the
  attention is tightly, correctly focused on the one visible feature the
  prompt asked about. A reminder that a single box-overlap number can
  undersell genuinely precise grounding.)
- **Gwen Stacy** / **Regina George** (`standout_trait`): hotspots
  concentrate tightly on the eyes/eyebrows -- both characters' roster
  entries are keyed on eyebrow asymmetry/smirk shape specifically.

All of these are in `results/attention/<character>/*.png` -- worth a
visual look, they're more convincing than the table.

**Secondary finding: layer choice matters.** Averaging over *all* decoder
layers shows clear grounding; the *last layer alone* looks close to
random/chance (0.96x) and visually speckles all over the background
(`results/attention/batman/aggregate_last_layer_only.png`). This suggests
the semantically grounded attention in this model lives in middle layers,
not the output layer -- a last-layer-only interpretability method (a
common simplification) would have concluded "no grounding" here, which
would've been the wrong conclusion.

## 3. Caveats

- "Self-match rate" (Section 1) and "face-box enrichment" (Section 2)
  are both proxy metrics built from the only images available
  (`static/characters/`, pictures of the characters themselves), not
  real booth visitor photos. Both likely overstate how easy the *real*
  matching task is; treat the sign/direction of these findings ("Qwen3-VL
  is minimally viable here", "attention is genuinely grounded, not
  noise") as more trustworthy than the exact numbers.
- Attention weights show *correlation* between generated text and image
  regions, not proof of *causal* reliance -- a stronger (and much more
  expensive) test would be occluding the grounded region and checking if
  the answer changes.
- This was all run on an idle A100-40GB; none of the latency numbers
  account for concurrent booth traffic.
