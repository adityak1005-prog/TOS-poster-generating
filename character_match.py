"""
Stage 3: character matching.

"""

import json
import os
import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
_model = genai.GenerativeModel("gemini-2.5-flash-lite")  # free-tier friendly, fast

# Each character entry: trait -> weight. Traits present in the extracted
# JSON add their weight to the character's score. Tune this table freely --
# it's just a dict, no retraining needed.
CHARACTER_TABLE = {
    "Iron Man": {"outfit_color:red": 3, "pose:arm_extended_right": 3, "pose:arm_extended_left": 3, "glasses": 0},
    "Superman": {"outfit_color:blue": 3, "pose:arm_raised_right": 3, "pose:arm_raised_left": 3},
    "The Matrix (Neo)": {"outfit_color:black": 3, "glasses": 3, "pose:neutral_stance": 1},
    "Wonder Woman": {"outfit_color:red": 2, "pose:hand_on_hip_left": 3, "pose:hand_on_hip_right": 3},
    "James Bond": {"outfit_color:black": 2, "glasses": 2, "pose:neutral_stance": 2},
    "The Hulk": {"outfit_color:green": 4, "pose:arm_raised_left": 2, "pose:arm_raised_right": 2},
    "Harry Potter": {"glasses": 3, "outfit_color:black": 1},
}

TIE_THRESHOLD = 1  # if top two scores differ by <= this, ask the LLM


def _score_characters(traits: dict) -> list[tuple[str, int]]:
    active_keys = {
        f"outfit_color:{traits['outfit_color']}",
        f"pose:{traits['pose']}",
    }
    if traits.get("glasses"):
        active_keys.add("glasses")

    scores = []
    for name, weights in CHARACTER_TABLE.items():
        score = sum(weights.get(key, 0) for key in active_keys)
        scores.append((name, score))

    return sorted(scores, key=lambda x: x[1], reverse=True)


def _llm_tiebreak(traits: dict, candidates: list[str]) -> dict:
    """
    Single fast call to break a tie and write a one-line caption.
    Use the smallest/fastest model available -- this call should return in
    well under 2 seconds since it's a tiny, structured, low-reasoning task.
    """
    prompt = f"""A photo booth detected these traits: {json.dumps(traits, default=str)}.
Pick exactly one of these movie characters as the best match: {candidates}.
Respond ONLY with JSON: {{"character": "...", "caption": "one funny 8-12 word line"}}"""

    resp = _model.generate_content(prompt)
    text = resp.text.strip().strip("`").removeprefix("json").strip()
    return json.loads(text)


def match_character(traits: dict) -> dict:
    ranked = _score_characters(traits)
    top_name, top_score = ranked[0]
    second_name, second_score = ranked[1]

    if top_score - second_score > TIE_THRESHOLD:
        # clear winner, no LLM call needed -- instant path
        return {
            "character": top_name,
            "caption": f"When the {traits['outfit_color']} fit meets {traits['pose'].replace('_', ' ')}.",
            "used_llm": False,
        }

    # ambiguous -- ask the model to break the tie and add flavor
    candidates = [top_name, second_name]
    result = _llm_tiebreak(traits, candidates)
    result["used_llm"] = True
    return result
