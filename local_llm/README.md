# Local VLM experiment: Qwen3-VL as a drop-in for stage 2 (`openai_analysis.py`)

This folder is a self-contained experiment, separate from the deployed app.
It does **not** modify `app.py`/`openai_analysis.py` -- it reuses their
character roster and instruction text (via `shared_prompt.py`, which imports
`openai_analysis.py` with a dummy `OPENAI_API_KEY` so no real API key or
network call is needed) and re-implements the analysis call against a local
Qwen3-VL model instead of the OpenAI API.

## Why Qwen3-VL, which size

Both `Qwen3-VL-4B-Instruct` and `Qwen3-VL-8B-Instruct` were already cached
locally (`~/.cache/huggingface/hub`), so both are benchmarked. `-Thinking`
variants and the 32B model are skipped for this task: this is a
single-pass classification + short structured-text job on a live booth
with a tight latency budget, the same reasoning `README.md` already gives
for why the OpenAI side uses `reasoning_effort=minimal` rather than a
deliberate/thinking mode -- a reasoning/thinking model would burn the
booth's latency budget on invisible thinking tokens for a task that
doesn't need multi-step deliberation.

## Files

- `shared_prompt.py` -- imports the roster + instruction builder from the
  real `openai_analysis.py` (dummy API key, no network), so the local model
  is judged against the *exact* same prompt/roster the deployed app uses.
- `qwen_local.py` -- loads a Qwen3-VL model via `transformers`
  (`Qwen3VLForConditionalGeneration`) and runs the same analysis task,
  returning the same JSON shape (`character`, `reasoning`, `caption`,
  `diffusion_prompt`, `detected_traits`). No `images.edit`/poster stage --
  this only replaces stage 2.
- `benchmark.py` -- runs the pipeline over every image in
  `static/characters/` (see "Benchmark design/limitations" below), for
  each model size, and reports latency, JSON-validity rate, and a
  self-match rate. Writes `results/benchmark_<model>.json`.
- `attention_viz.py` -- (only run if the benchmark looks decent) re-runs
  generation with `output_attentions=True`, reshapes the attention from
  generated tokens back onto the image's ViT patch grid, and saves heatmap
  overlays to `results/attention/`.

## Benchmark design & limitations (read before trusting the numbers)

There's no dataset of real "booth" visitor selfies to test character
*matching* against -- the only images available are
`static/characters/<slug>.{jpg,webp}`, which are pictures of the
characters themselves, not ordinary people's photos. So the benchmark
uses those images as the "captured photo" input and checks whether the
model's pick matches the character the image is actually of ("self-match
rate"). This is a much easier task than the real one (matching a random
person's face/vibe to a character), so a high self-match rate is a
necessary sanity check, not proof the model is good at the real,
harder job -- treat it as "does this model even work at this task /
follow the roster / produce valid structured output," not as an accuracy
number to quote.

## Running

```bash
ENV=/DATA1/scratch1/samyak/micromamba/envs/qwen/bin/python3
CUDA_VISIBLE_DEVICES=3 $ENV benchmark.py --model 4b
CUDA_VISIBLE_DEVICES=3 $ENV benchmark.py --model 8b
CUDA_VISIBLE_DEVICES=3 $ENV attention_viz.py --model 4b --image static/characters/batman.jpg
```
