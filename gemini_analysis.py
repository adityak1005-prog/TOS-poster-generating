"""
DEPRECATED / UNUSED -- kept only because this session's tools can't delete
files outright. Nothing imports this module anymore.

Stage 2 (analysis + character match + prompt construction) moved from
Gemini to OpenAI so it could run on the same paid account/context as poster
generation, and to get automatic prompt caching without Gemini's free-tier
explicit-caching block. See openai_analysis.py for the current
implementation -- it's a near-complete rewrite (different SDK, different
caching mechanism, updated character descriptions and matching prompt), not
a drop-in replacement of this file.

GEMINI_API_KEY / GEMINI_MODEL / GEMINI_CACHE_TTL in .env are no longer read
by anything, and google-genai is no longer a required dependency.

Safe to delete this file entirely next time you're on a machine where you
can remove it (`rm gemini_analysis.py` or delete via your file browser) --
nothing in the app will break, since app.py no longer imports it.
"""
