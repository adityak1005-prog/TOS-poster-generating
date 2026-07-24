"""
Single source of truth for the roster + instruction text: reuses the real
`openai_analysis.py` from the parent app instead of copy-pasting the roster,
so this experiment is always judged against the exact prompt the deployed
app actually sends to OpenAI.

`openai_analysis.py` builds an `OpenAI(api_key=os.environ["OPENAI_API_KEY"])`
client at import time (module-level), which would raise immediately if the
key isn't set. This experiment never calls that client -- only the roster
dict and `_build_instruction()` are used -- so a dummy key is injected
before import purely to satisfy that line.
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("OPENAI_API_KEY", "unused-local-only-not-a-real-key")

import openai_analysis  # noqa: E402  (must come after the dummy-key shim above)

CHARACTER_STYLE_GUIDE = openai_analysis.CHARACTER_STYLE_GUIDE
CHARACTERS = openai_analysis.CHARACTERS
build_instruction = openai_analysis._build_instruction
slugify = openai_analysis._slugify
CHARACTER_REF_DIR = REPO_ROOT / openai_analysis.CHARACTER_REF_DIR

# JSON schema shown to the local model as plain text -- Qwen3-VL (via
# plain `transformers.generate`) has no OpenAI-style `response_format`
# strict-schema enforcement, so the schema has to be spelled out in-prompt
# and the output parsed/repaired defensively (see qwen_local.py).
JSON_SHAPE_HINT = """
Respond with ONLY a single JSON object (no markdown fences, no commentary
before or after it), with exactly this shape:
{
  "character": "<one name copied exactly from the roster list above>",
  "reasoning": "<1-2 sentences, written to the person as 'you'>",
  "caption": "<short witty one-liner, 8-12 words>",
  "diffusion_prompt": "<50-70 word single-sentence image-edit prompt>",
  "detected_traits": {
    "outfit_color": "<one word>",
    "pose_description": "<short phrase>",
    "glasses": <true or false>,
    "standout_trait": "<short phrase>"
  }
}
"""


def full_prompt() -> str:
    return build_instruction() + "\n" + JSON_SHAPE_HINT
