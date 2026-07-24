"""
Benchmarks a local Qwen3-VL model on the same analysis task as
openai_analysis.py, using the 21 images in static/characters/ as inputs
(see README.md "Benchmark design & limitations" -- these are pictures OF
the characters, so "self-match rate" is a sanity check, not a real
matching-accuracy number).

Usage:
    python benchmark.py --model 4b
    python benchmark.py --model 8b
    python benchmark.py --model 4b --limit 5   # quick smoke run
"""

import argparse
import json
import time
from pathlib import Path

from PIL import Image

import qwen_local
import shared_prompt

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def iter_character_images():
    """Yields (character_name, image_path) for every roster member that
    has a reference image on disk, using the same slug convention as
    openai_analysis.py."""
    ref_dir = shared_prompt.CHARACTER_REF_DIR
    for name in shared_prompt.CHARACTERS:
        slug = shared_prompt.slugify(name)
        for ext in ("jpg", "jpeg", "png", "webp"):
            path = ref_dir / f"{slug}.{ext}"
            if path.exists():
                yield name, path
                break


def run(model_key: str, limit: int | None, max_new_tokens: int):
    items = list(iter_character_images())
    if limit:
        items = items[:limit]

    print(f"[benchmark] model={model_key}  n_images={len(items)}  "
          f"attn_implementation=sdpa (production-speed setting)")

    # Warm-up call: excluded from timing stats since it includes CUDA
    # kernel autotune/compilation overhead that a real booth wouldn't pay
    # more than once per process lifetime.
    warm_img = Image.open(items[0][1]).convert("RGB")
    t0 = time.time()
    qwen_local.analyze_and_match(warm_img, model_key=model_key, max_new_tokens=max_new_tokens)
    print(f"[benchmark] warm-up call took {time.time() - t0:.1f}s (excluded from stats below)")

    records = []
    for true_character, path in items:
        img = Image.open(path).convert("RGB")
        result = qwen_local.analyze_and_match(img, model_key=model_key, max_new_tokens=max_new_tokens)
        self_match = result.get("character") == true_character
        record = {
            "true_character": true_character,
            "image": str(path.relative_to(shared_prompt.REPO_ROOT)),
            "predicted_character": result.get("character"),
            "self_match": self_match,
            "valid_json": result["_valid_json"],
            "error": result["_error"],
            "elapsed_s": result["_elapsed_s"],
            "tokens_per_s": result["_tokens_per_s"],
            "n_new_tokens": result["_n_new_tokens"],
            "reasoning": result.get("reasoning"),
            "standout_trait": (result.get("detected_traits") or {}).get("standout_trait"),
        }
        records.append(record)
        status = "OK " if record["valid_json"] else "BAD"
        match_flag = "match" if self_match else "----"
        print(f"[{status}] {true_character:28s} -> {str(record['predicted_character']):28s} "
              f"[{match_flag}] {record['elapsed_s']:6.2f}s  {record['tokens_per_s'] or 0:5.1f} tok/s")

    valid = [r for r in records if r["valid_json"]]
    matched = [r for r in records if r["self_match"]]
    times = [r["elapsed_s"] for r in records]
    tps = [r["tokens_per_s"] for r in records if r["tokens_per_s"]]

    summary = {
        "model": model_key,
        "n_images": len(records),
        "valid_json_rate": round(len(valid) / len(records), 3) if records else None,
        "self_match_rate": round(len(matched) / len(records), 3) if records else None,
        "latency_s": {
            "mean": round(sum(times) / len(times), 2) if times else None,
            "min": round(min(times), 2) if times else None,
            "max": round(max(times), 2) if times else None,
        },
        "tokens_per_s_mean": round(sum(tps) / len(tps), 1) if tps else None,
    }

    print("\n" + "=" * 70)
    print(f"SUMMARY ({model_key}):")
    print(json.dumps(summary, indent=2))
    print("=" * 70)

    out_path = RESULTS_DIR / f"benchmark_{model_key}.json"
    out_path.write_text(json.dumps({"summary": summary, "records": records}, indent=2))
    print(f"[benchmark] wrote {out_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(qwen_local.MODEL_IDS.keys()), default="4b")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=500)
    args = parser.parse_args()
    run(args.model, args.limit, args.max_new_tokens)
