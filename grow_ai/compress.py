"""Event compression for 20-word insights.

Compresses Claude Code tool events into concise, actionable insights.
Uses rule-based extraction for high-signal tools, LLM fallback for low-signal.
"""

import httpx
from grow_ai.config import cfg
from grow_ai.scrubber import scrub

_HIGH_SIGNAL_TOOLS = {"Edit", "Write", "Bash", "MultiEdit"}
_MIN_WORDS = 8


def _rule_based(event: dict) -> str:
    """Extract a concise insight from structured tool data. No LLM."""
    tool = event.get("tool", "Unknown")
    action = str(event.get("action", ""))[:80]
    result = str(event.get("result", ""))[:80]
    return f"{tool} {action}: {result}".strip()


def _llm_compress(full_context: str) -> str:
    """Call Qwen2.5-3B via Ollama to summarize in ≤20 words."""
    prompt = (
        "Summarize the key learning from this developer interaction in 20 words or fewer. "
        "Be specific about what was done and what was learned. "
        f"Interaction:\n{full_context[:500]}"
    )
    response = httpx.post(
        f"{cfg.ollama_base_url}/api/generate",
        json={"model": cfg.generative_model, "prompt": prompt, "stream": False},
        timeout=10.0,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def compress(event: dict) -> str:
    """
    Stage 1: rule-based extraction.
    Stage 2: LLM fallback if result is low-signal.
    Always scrubs the result before returning.
    """
    tool = event.get("tool", "")
    rule_result = _rule_based(event)
    word_count = len(rule_result.split())

    if tool in _HIGH_SIGNAL_TOOLS and word_count >= _MIN_WORDS:
        return scrub(rule_result)

    full_context = str(event)
    return scrub(_llm_compress(full_context))
