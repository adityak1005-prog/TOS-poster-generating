"""
Local drop-in replacement for stage 2 (openai_analysis.analyze_and_match),
using Qwen3-VL via `transformers` instead of the OpenAI API.

Kept deliberately simple/single-file: load once, call `analyze_and_match()`
repeatedly. Model/processor are cached at module level (see `_get_model`)
so the benchmark harness can loop over many images without reloading
weights each time.

Loads at full bfloat16 -- needs a GPU with ~10-12GB+ free VRAM for the 4B
model (roughly 8GB for weights alone, plus activations/KV-cache for the
roster prompt + image). No quantization here: this was previously added
(via bitsandbytes) to squeeze the model onto 6-8GB laptop GPUs, but was
removed once the deployment moved to hardware with real VRAM headroom --
see git history if you need to run this on a smaller card again.

Also owns the attention-capture machinery (AttentionCollector and friends,
originally prototyped in attention_viz.py for offline grounding analysis --
see FINDINGS.md section 2). analyze_and_match(capture_attention=True) below
reuses these to produce a live "where the AI looked" heatmap for the booth's
reveal screen, in the SAME generate() call that produces the JSON analysis
-- running the model a second time just for a heatmap would double GPU cost
per capture, which a live booth can't afford. attention_viz.py now imports
these helpers from here instead of defining its own copies.
"""

import gc
import io
import json
import re
import time

import cv2
import matplotlib
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention

import shared_prompt

MODEL_IDS = {
    "4b": "Qwen/Qwen3-VL-4B-Instruct",
    "8b": "Qwen/Qwen3-VL-8B-Instruct",
}

IMAGE_TOKEN_ID = 151655  # Qwen's <|image_pad|> token id, confirmed via processor inspection

_cache = {"key": None, "model": None, "processor": None}


def _get_model(model_key: str, attn_implementation: str = "sdpa"):
    """Loads (once) and returns (model, processor) for the given size +
    attention backend. Swapping sizes/backends unloads the previous model
    first -- a single A100 slice doesn't need to hold multiple copies
    resident at once for this experiment, and freeing avoids stacking VRAM
    across benchmark runs.

    attn_implementation matters a lot here: "sdpa" (default, fused kernel)
    is what a real deployment would use for speed, but it doesn't expose
    per-head attention weights. "eager" (plain matmul attention) is much
    slower and more memory-hungry but is required for
    `output_attentions=True` to return real weights -- transformers'
    fused backends (sdpa/flash-attention) never materialize the full
    attention matrix, so they have nothing to return. Benchmarking should
    use "sdpa"; attention_viz.py should use "eager".
    """
    cache_key = (model_key, attn_implementation)
    if _cache["key"] == cache_key:
        return _cache["model"], _cache["processor"]

    if _cache["model"] is not None:
        del _cache["model"]
        gc.collect()
        torch.cuda.empty_cache()

    model_id = MODEL_IDS[model_key]
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation=attn_implementation,
    )
    model.eval()

    _cache.update(key=cache_key, model=model, processor=processor)
    return model, processor


class AttentionCollector:
    """Registers forward hooks on every Qwen3VLTextAttention layer to
    capture real attention weights during generation, reduced-and-discarded
    per layer in real time so memory never blows up (see attention_viz.py's
    original module docstring in git history for the full derivation of why
    this replaces `output_attentions=True`). Only usable with
    attn_implementation="eager" -- fused kernels (sdpa) never materialize a
    full attention matrix, so they have nothing for this to hook."""

    def __init__(self, model, image_token_positions: torch.Tensor):
        self.image_token_positions = image_token_positions
        self.per_layer: dict[int, list[np.ndarray]] = {}
        self._handles = []
        for _, module in model.named_modules():
            if isinstance(module, Qwen3VLTextAttention):
                self.per_layer[module.layer_idx] = []
                self._handles.append(module.register_forward_hook(self._make_hook(module.layer_idx)))

    def _make_hook(self, layer_idx):
        def hook(module, inputs, output):
            attn_output, attn_weights = output
            if attn_weights is None:
                return output
            q_len = attn_weights.shape[2]
            row = -1 if q_len > 1 else 0  # prefill: last prompt position predicts token 0; decode: the only row
            row_weights = attn_weights[0, :, row, :]  # (heads, k_len)
            cols = self.image_token_positions.to(row_weights.device)
            img_weights = row_weights.index_select(1, cols)  # (heads, n_img_tokens)
            reduced = img_weights.mean(dim=0).float().cpu().numpy()
            self.per_layer[layer_idx].append(reduced)
            return (attn_output, None)  # prevent upstream accumulation of the full tensor
        return hook

    def remove(self):
        for h in self._handles:
            h.remove()

    def stacked(self) -> np.ndarray:
        """(num_layers, num_generation_steps, n_image_tokens). Step i is
        the attention used to produce generated token i."""
        layers = sorted(self.per_layer.keys())
        return np.stack([np.stack(self.per_layer[l], axis=0) for l in layers], axis=0)


def token_char_spans(tokenizer, token_ids: list[int]):
    """[(char_start, char_end), ...] for each token, aligned to the fully
    decoded string -- built by incremental decode since Qwen's tokenizer
    merges bytes/BPE pieces in ways a naive per-token decode wouldn't
    reproduce exactly."""
    spans = []
    prev_len = 0
    for i in range(len(token_ids)):
        text = tokenizer.decode(token_ids[: i + 1], skip_special_tokens=True)
        spans.append((prev_len, len(text)))
        prev_len = len(text)
    return spans, prev_len


def find_phrase_token_indices(tokenizer, token_ids: list[int], full_text: str, phrase: str) -> list[int]:
    start = full_text.find(phrase)
    if start == -1:
        return []
    end = start + len(phrase)
    spans, _ = token_char_spans(tokenizer, token_ids)
    return [i for i, (s, e) in enumerate(spans) if e > start and s < end]


def aggregate_map(attn_stack: np.ndarray, step_indices: list[int], layer_agg: str = "all") -> np.ndarray | None:
    if not step_indices:
        return None
    layers = attn_stack.shape[0]
    sub = attn_stack[:, step_indices, :]
    if layer_agg == "all":
        sel = sub
    elif layer_agg == "last_half":
        sel = sub[layers // 2:]
    elif layer_agg == "last":
        sel = sub[-1:]
    else:
        raise ValueError(layer_agg)
    return sel.mean(axis=(0, 1))  # (n_img_tokens,)


def clip_outliers(raw_vec: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """Winsorizes values above the given percentile -- attention sinks (1-2
    grid cells absorbing a hugely disproportionate share of the mass, see
    FINDINGS.md section 2) would otherwise dominate the color scale and
    flatten every other, more meaningful, cell to near-zero."""
    cap = np.percentile(raw_vec, percentile)
    return np.minimum(raw_vec, cap)


def render_overlay(image: Image.Image, raw_vec: np.ndarray, grid_hw: tuple[int, int], alpha: float = 0.45,
                    clip_percentile: float = 99.0) -> Image.Image:
    grid_h, grid_w = grid_hw
    display_vec = clip_outliers(raw_vec, clip_percentile) if clip_percentile else raw_vec
    grid = display_vec.reshape(grid_h, grid_w).astype(np.float32)
    grid = grid - grid.min()
    if grid.max() > 0:
        grid = grid / grid.max()
    W, H = image.size
    heat = cv2.resize(grid, (W, H), interpolation=cv2.INTER_CUBIC)
    heat = np.clip(heat, 0, 1)
    colored = (matplotlib.colormaps["jet"](heat)[..., :3] * 255).astype(np.uint8)
    base = np.array(image.convert("RGB"))
    blended = (base.astype(np.float32) * (1 - alpha) + colored.astype(np.float32) * alpha).astype(np.uint8)
    return Image.fromarray(blended)


def _extract_json(text: str) -> dict:
    """Qwen3-VL isn't schema-constrained the way the OpenAI call is (see
    shared_prompt.JSON_SHAPE_HINT) -- it can wrap the JSON in ```json
    fences or add a stray sentence despite instructions. Strip fences and
    grab the outermost {...} block defensively before parsing."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object found in model output: {text!r}")
    return json.loads(match.group(0))


def build_inputs(processor, image: Image.Image, prompt: str):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return inputs


def analyze_and_match(image: Image.Image, model_key: str = "4b", max_new_tokens: int = 500,
                       attn_implementation: str = "sdpa", capture_attention: bool = False) -> dict:
    """Same contract as openai_analysis.analyze_and_match: takes a PIL
    image, returns a dict with character/reasoning/caption/diffusion_prompt
    /detected_traits, plus timing/validity metadata the benchmark uses.

    capture_attention=True additionally grounds a "where the AI looked"
    heatmap in this SAME generate() call, returned as JPEG bytes under
    "_attention_overlay_jpeg" (None if capture_attention is False, or if
    capturing/rendering it failed -- failure here is non-fatal, the analysis
    result itself is still returned). This forces attn_implementation to
    "eager" regardless of what was passed in, since only eager exposes real
    attention weights (see _get_model's docstring) -- benchmark.py never
    passes capture_attention, so its sdpa-timed numbers are unaffected.

    The heatmap is grounded in whichever generated span the "reasoning"
    text occupies (the field FINDINGS.md found to be the most face-attentive
    of the fields this model writes) -- if that phrase can't be located in
    the raw output for some reason, it falls back to the aggregate over the
    entire generated response rather than returning nothing.
    """
    if capture_attention:
        attn_implementation = "eager"
    model, processor = _get_model(model_key, attn_implementation)
    prompt = shared_prompt.full_prompt()
    inputs = build_inputs(processor, image, prompt).to(model.device)

    collector = None
    grid_hw = None
    if capture_attention:
        image_token_positions = (inputs["input_ids"][0] == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
        _, h, w = inputs["image_grid_thw"][0].tolist()
        merge = processor.image_processor.merge_size
        grid_hw = (h // merge, w // merge)
        collector = AttentionCollector(model, image_token_positions)

    t0 = time.time()
    try:
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # deterministic -- same reasoning as the OpenAI side using no custom temperature
            )
    finally:
        if collector is not None:
            collector.remove()
    elapsed = time.time() - t0

    new_tokens = generated[:, inputs["input_ids"].shape[1]:]
    raw_text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
    n_new_tokens = new_tokens.shape[1]

    result = {
        "_raw_text": raw_text,
        "_elapsed_s": round(elapsed, 3),
        "_tokens_per_s": round(n_new_tokens / elapsed, 1) if elapsed > 0 else None,
        "_n_new_tokens": int(n_new_tokens),
        "_valid_json": False,
        "_error": None,
        "_attention_overlay_jpeg": None,
    }

    try:
        parsed = _extract_json(raw_text)
        result.update(parsed)
        result["_valid_json"] = True
        if result.get("character") not in shared_prompt.CHARACTERS:
            result["_error"] = f"character {result.get('character')!r} not in roster"
    except Exception as e:
        result["_error"] = str(e)

    if collector is not None:
        try:
            attn_stack = collector.stacked()
            token_ids = new_tokens[0].cpu().tolist()
            reasoning_text = result.get("reasoning") or ""
            step_idxs = (
                find_phrase_token_indices(processor.tokenizer, token_ids, raw_text, reasoning_text)
                if reasoning_text else []
            )
            if not step_idxs:
                step_idxs = list(range(attn_stack.shape[1]))  # fall back to the whole response
            vec = aggregate_map(attn_stack, step_idxs, layer_agg="all")
            if vec is not None:
                overlay_img = render_overlay(image, vec, grid_hw)
                buf = io.BytesIO()
                overlay_img.save(buf, format="JPEG", quality=85)
                result["_attention_overlay_jpeg"] = buf.getvalue()
        except Exception as e:
            print(f"[qwen_local] attention overlay failed (non-fatal, capture proceeds without it): {e}")

    return result
