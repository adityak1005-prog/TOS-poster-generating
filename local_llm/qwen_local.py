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
"""

import gc
import json
import re
import time

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

import shared_prompt

MODEL_IDS = {
    "4b": "Qwen/Qwen3-VL-4B-Instruct",
    "8b": "Qwen/Qwen3-VL-8B-Instruct",
}

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
                       attn_implementation: str = "sdpa") -> dict:
    """Same contract as openai_analysis.analyze_and_match: takes a PIL
    image, returns a dict with character/reasoning/caption/diffusion_prompt
    /detected_traits, plus timing/validity metadata the benchmark uses."""
    model, processor = _get_model(model_key, attn_implementation)
    prompt = shared_prompt.full_prompt()
    inputs = build_inputs(processor, image, prompt).to(model.device)

    t0 = time.time()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # deterministic -- same reasoning as the OpenAI side using no custom temperature
        )
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
    }

    try:
        parsed = _extract_json(raw_text)
        result.update(parsed)
        result["_valid_json"] = True
        if result.get("character") not in shared_prompt.CHARACTERS:
            result["_error"] = f"character {result.get('character')!r} not in roster"
    except Exception as e:
        result["_error"] = str(e)

    return result
