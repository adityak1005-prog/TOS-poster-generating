"""
Attention-grounding visualization for the local Qwen3-VL analysis pipeline.

Question this answers: when Qwen3-VL names a character / names a
"standout trait" / writes its reasoning sentence, is it actually attending
to the relevant part of the *image* (e.g. the face), or is the text
mostly self-attending / attending to generic image regions? We answer
this by capturing real attention weights during generation (not a
post-hoc saliency proxy like Grad-CAM) and overlaying them on the image.

## Why not `model.generate(..., output_attentions=True)`

That convenience flag makes transformers stack a full (batch, heads,
q_len, k_len) tensor for *every layer* into `outputs.attentions`. For this
pipeline's prompt (~5-6k tokens: the full character roster instructions +
~1900 image tokens), the prefill step alone would require materializing
and RETAINING roughly num_layers x heads x seq^2 x 2 bytes -- tens of GB,
more than a single A100 slice has. That memory is only a *bookkeeping*
cost (eager attention computes the full matrix transiently regardless,
that part is unavoidable) -- the blowup is specifically from keeping every
layer's full matrix alive simultaneously for the whole forward pass.

## The fix: forward hooks that reduce-then-discard, per layer, in real time

`Qwen3VLTextAttention.forward` (see transformers' modeling_qwen3_vl.py)
always computes and returns `attn_weights` when `attn_implementation="eager"`
-- this isn't gated by `output_attentions` at all, that flag only controls
whether an *outer* layer chooses to keep/stack it. So we never pass
`output_attentions=True` anywhere; instead we register a forward hook on
every `Qwen3VLTextAttention` submodule that, the instant that layer's
attn_weights tensor exists:
  1. slices out only the columns for the image-token positions (the only
     part we care about) and averages over attention heads,
  2. stores that tiny (n_image_tokens,) vector,
  3. returns `(attn_output, None)` so nothing upstream ever retains the
     full (heads, seq, seq) tensor.
This bounds memory to O(layers x generation_steps x n_image_tokens), a
few tens of MB instead of tens of GB, with no accuracy loss for what we
actually want to measure.

## Design choices for "per phrase or at the end" (see README for more)

Both, deliberately:
  - Per-phrase maps for the three most semantically distinct fields the
    model writes: the `character` name, `detected_traits.standout_trait`,
    and `reasoning`. These are the fields most directly claiming "I saw
    X in the image", so they're the most meaningful test of grounding.
  - One aggregate map over the entire generated response, as a baseline/
    sanity comparison -- if the per-phrase maps all look identical to the
    aggregate, that's itself a finding (the model isn't shifting focus
    per-claim, it "looks once" and writes from that).
  - Two layer-aggregation variants are computed for every map: mean over
    ALL decoder layers (the default/primary one used for the saved PNGs)
    and last-layer-only (saved once, for the aggregate map, as a second
    opinion) -- late layers in decoder LLMs are typically considered more
    semantically/output-grounded than early ones, which tend to be more
    positional/syntactic, so it's worth checking whether the two tell a
    different story rather than assuming a single aggregation is "correct".

## Quantitative grounding metric

A heatmap is convincing when eyeballed but hard to compare across images.
We add one objective number: run OpenCV's Haar-cascade frontal-face
detector on the input photo, then compute what fraction of the (image-
token) attention mass falls inside a padded box around the detected face,
versus what fraction of the image AREA that box covers. The ratio
("enrichment") is >1 if attention concentrates on the face more than
chance given the box's size, ~1 if attention is roughly uniform, <1 if
attention actively avoids the face. This only works on images where a
face is detectable -- several roster images are masked/animated
characters where Haar cascades are expected to fail; those are reported
as `null` rather than guessed at.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

import benchmark
import qwen_local
import shared_prompt
# AttentionCollector, IMAGE_TOKEN_ID, aggregate_map, clip_outliers,
# render_overlay, token_char_spans, find_phrase_token_indices all now live
# in qwen_local.py (promoted there so the production booth pipeline can
# reuse them for its own live heatmap -- see qwen_local.analyze_and_match's
# capture_attention path). This module keeps only what's specific to
# offline benchmarking: face-box grounding metrics, sink diagnostics, and
# the per-image/all-roster-images CLI loop.
from qwen_local import (
    AttentionCollector,
    IMAGE_TOKEN_ID,
    aggregate_map,
    clip_outliers,
    render_overlay,
)

RESULTS_DIR = Path(__file__).parent / "results" / "attention"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def generate_with_attention(image: Image.Image, model_key: str, max_new_tokens: int):
    model, processor = qwen_local._get_model(model_key, attn_implementation="eager")
    prompt = shared_prompt.full_prompt()
    inputs = qwen_local.build_inputs(processor, image, prompt).to(model.device)

    image_token_positions = (inputs["input_ids"][0] == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
    t, h, w = inputs["image_grid_thw"][0].tolist()
    merge = processor.image_processor.merge_size
    grid_h, grid_w = h // merge, w // merge
    assert t == 1, "multi-frame input not expected for a single photo"
    assert grid_h * grid_w == image_token_positions.numel(), (
        f"grid mismatch: {grid_h}x{grid_w} != {image_token_positions.numel()} image tokens"
    )

    collector = AttentionCollector(model, image_token_positions)
    try:
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    finally:
        collector.remove()

    new_tokens = generated[0, inputs["input_ids"].shape[1]:]
    raw_text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
    attn_stack = collector.stacked()
    assert attn_stack.shape[1] == new_tokens.shape[0], (
        f"step count {attn_stack.shape[1]} != generated token count {new_tokens.shape[0]}"
    )

    return {
        "raw_text": raw_text,
        "new_token_ids": new_tokens.cpu().tolist(),
        "attn_stack": attn_stack,        # (layers, steps, n_img_tokens)
        "grid_hw": (grid_h, grid_w),
        "tokenizer": processor.tokenizer,
    }


def find_phrase_token_indices(tokenizer, token_ids: list[int], full_text: str, phrase: str) -> list[int]:
    return qwen_local.find_phrase_token_indices(tokenizer, token_ids, full_text, phrase)


def sink_diagnostics(raw_vec: np.ndarray) -> dict:
    """Quantifies how 'peaky' a map is. LLM/ViT attention is well known to
    dump disproportionate weight onto a handful of tokens for bookkeeping
    reasons unrelated to their content ("attention sinks" /
    "register tokens" -- see e.g. StreamingLLM, "Vision Transformers Need
    Registers"). A single grid cell absorbing a large fraction of the
    total is a red flag that a naive min-max-normalized heatmap will be
    dominated by that one artifact cell rather than showing anything
    content-related -- this is exactly what a first look at this
    pipeline's maps showed (one cell ~800x the mean on one image)."""
    total = float(raw_vec.sum())
    if total <= 0:
        return {"top1_fraction": None, "top3_fraction": None}
    sorted_vals = np.sort(raw_vec)[::-1]
    return {
        "top1_fraction": round(float(sorted_vals[0] / total), 4),
        "top3_fraction": round(float(sorted_vals[:3].sum() / total), 4),
    }


def detect_face_box(image: Image.Image):
    """Several roster reference images are quite small (e.g. 183x275px),
    so a fixed 60x60 minSize (reasonable for phone-camera-resolution
    photos) misses faces that only span ~40-50px here. Equalizing
    histogram + a smaller minSize/finer scaleFactor trades a few more
    false positives for meaningfully better recall on these low-res
    stylized images -- we still only keep the single largest detection."""
    arr = np.array(image.convert("L"))
    arr = cv2.equalizeHist(arr)
    faces = _FACE_CASCADE.detectMultiScale(arr, scaleFactor=1.05, minNeighbors=4, minSize=(40, 40))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return int(x), int(y), int(w), int(h)


def face_attention_stats(raw_vec: np.ndarray, grid_hw: tuple[int, int], face_box, image_size):
    if raw_vec is None or face_box is None:
        return None
    grid_h, grid_w = grid_hw
    grid = np.clip(raw_vec.reshape(grid_h, grid_w).astype(np.float32), 0, None)
    W, H = image_size
    resized = cv2.resize(grid, (W, H), interpolation=cv2.INTER_LINEAR)
    total = float(resized.sum())
    if total <= 0:
        return None
    x, y, w, h = face_box
    pad_x, pad_y = int(w * 0.2), int(h * 0.2)
    x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
    x1, y1 = min(W, x + w + pad_x), min(H, y + h + pad_y)
    mass_fraction = float(resized[y0:y1, x0:x1].sum() / total)
    area_fraction = float((x1 - x0) * (y1 - y0) / (W * H))
    return {
        "mass_fraction": round(mass_fraction, 4),
        "area_fraction": round(area_fraction, 4),
        "enrichment": round(mass_fraction / area_fraction, 3) if area_fraction > 0 else None,
    }


def run_one_image(image_path: Path, model_key: str, max_new_tokens: int, layer_agg: str) -> dict:
    image = Image.open(image_path).convert("RGB")
    slug = shared_prompt.slugify(image_path.stem)
    out_dir = RESULTS_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    gen = generate_with_attention(image, model_key, max_new_tokens)
    raw_text, token_ids, attn_stack, grid_hw, tokenizer = (
        gen["raw_text"], gen["new_token_ids"], gen["attn_stack"], gen["grid_hw"], gen["tokenizer"]
    )

    try:
        parsed = qwen_local._extract_json(raw_text)
    except Exception as e:
        print(f"[attention_viz] {image_path.name}: JSON parse failed ({e})")
        parsed = {}

    face_box = detect_face_box(image)
    if face_box:
        debug_img = image.copy()
        d = ImageDraw.Draw(debug_img)
        x, y, w, h = face_box
        d.rectangle([x, y, x + w, y + h], outline="lime", width=4)
        debug_img.save(out_dir / "face_box_debug.png")

    result = {
        "image": str(image_path),
        "true_character": image_path.stem,
        "predicted_character": parsed.get("character"),
        "raw_text": raw_text,
        "face_box": face_box,
        "layer_agg": layer_agg,
        "maps": {},
    }

    all_steps = list(range(attn_stack.shape[1]))
    phrase_specs = [
        ("aggregate", all_steps, layer_agg),
        ("aggregate_last_layer_only", all_steps, "last"),
    ]
    phrase_values = {
        "character": parsed.get("character"),
        "standout_trait": (parsed.get("detected_traits") or {}).get("standout_trait"),
        "reasoning": parsed.get("reasoning"),
    }
    for name, value in phrase_values.items():
        if not value:
            continue
        idxs = find_phrase_token_indices(tokenizer, token_ids, raw_text, str(value))
        if not idxs:
            print(f"[attention_viz]   couldn't locate token span for '{name}' ({value!r})")
            continue
        phrase_specs.append((name, idxs, layer_agg))

    for name, step_idxs, agg in phrase_specs:
        vec = aggregate_map(attn_stack, step_idxs, agg)
        if vec is None:
            continue
        overlay_img = render_overlay(image, vec, grid_hw)
        out_path = out_dir / f"{name}.png"
        overlay_img.save(out_path)
        sink = sink_diagnostics(vec)
        stats_raw = face_attention_stats(vec, grid_hw, face_box, image.size)
        stats_clipped = face_attention_stats(clip_outliers(vec), grid_hw, face_box, image.size)
        result["maps"][name] = {
            "file": out_path.name,
            "n_steps": len(step_idxs),
            "sink_diagnostics": sink,
            "face_stats_raw": stats_raw,
            "face_stats_sink_robust": stats_clipped,
        }
        print(f"    [{name:26s}] n_steps={len(step_idxs):3d}  sink_top1={sink['top1_fraction']}  "
              f"enrichment(raw/robust)={((stats_raw or {}).get('enrichment'))}/{((stats_clipped or {}).get('enrichment'))}")

    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="4b", choices=list(qwen_local.MODEL_IDS.keys()))
    parser.add_argument("--image", default="all", help="'all' for every roster image, or a path to one image")
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--layer-agg", default="all", choices=["all", "last_half", "last"])
    args = parser.parse_args()

    if args.image == "all":
        targets = [path for _, path in benchmark.iter_character_images()]
    else:
        targets = [Path(args.image)]

    all_results = []
    for path in targets:
        print(f"\n[attention_viz] {path.name}")
        res = run_one_image(path, args.model, args.max_new_tokens, args.layer_agg)
        all_results.append(res)

    # Summary: mean enrichment per map type (sink-robust version is primary
    # -- see sink_diagnostics/clip_outliers docstrings for why the raw
    # version can be swamped by one or two content-unrelated cells), over
    # images where a face was detected, plus how often a sink was even
    # present so the caveat is quantified rather than just asserted.
    summary_robust: dict[str, list[float]] = {}
    summary_raw: dict[str, list[float]] = {}
    sink_rates: dict[str, list[float]] = {}
    for res in all_results:
        for map_name, info in res["maps"].items():
            robust = info.get("face_stats_sink_robust")
            raw = info.get("face_stats_raw")
            sink = info.get("sink_diagnostics") or {}
            if robust and robust.get("enrichment") is not None:
                summary_robust.setdefault(map_name, []).append(robust["enrichment"])
            if raw and raw.get("enrichment") is not None:
                summary_raw.setdefault(map_name, []).append(raw["enrichment"])
            if sink.get("top1_fraction") is not None:
                sink_rates.setdefault(map_name, []).append(sink["top1_fraction"])

    n_total = len(all_results)
    n_faces = sum(1 for r in all_results if r["face_box"])
    print("\n" + "=" * 70)
    print(f"GROUNDING SUMMARY ({n_faces}/{n_total} images had a detectable face)")
    print("enrichment: >1 = attends to face more than chance given its size, ~1 = uniform, <1 = avoids face")
    for map_name in summary_robust:
        rv = summary_robust[map_name]
        raw_v = summary_raw.get(map_name, [])
        sv = sink_rates.get(map_name, [])
        print(f"  {map_name:26s} n={len(rv):2d}/{n_total}  "
              f"enrichment[sink-robust]={sum(rv) / len(rv):.2f} (min={min(rv):.2f} max={max(rv):.2f})  "
              f"enrichment[raw]={sum(raw_v) / len(raw_v):.2f}  "
              f"mean_top1_sink_fraction={sum(sv) / len(sv):.3f}")
    print("=" * 70)
    summary = summary_robust

    (RESULTS_DIR / "summary.json").write_text(json.dumps(
        {"per_image": all_results, "enrichment_summary": {k: v for k, v in summary.items()}}, indent=2
    ))
    print(f"[attention_viz] wrote {RESULTS_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
