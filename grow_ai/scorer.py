from datetime import datetime

_TOOL_WEIGHTS = {
    "Edit": 10, "Write": 10, "MultiEdit": 12,
    "Bash": 8, "Read": 0, "Glob": 0, "Grep": 0,
}

_FRAMEWORK_SIGNALS: dict[str, list[str]] = {
    "thinking_fast_and_slow": ["system 1", "system 2", "bias", "shortcut", "intuition", "heuristic", "deliberate"],
    "clear_thinking": ["default reaction", "override", "ordinary moment", "control thought"],
    "algorithms_to_live_by": ["sort", "cache", "search", "optimize", "explore exploit", "algorithm"],
    "memory_palace": ["visualize", "palace", "memory palace", "association", "encode", "link"],
    "memory_book": ["peg", "chain", "anchor", "hook system", "number shape"],
    "make_it_stick": ["recall", "spaced repetition", "interleave", "active recall", "retrieval practice"],
    "ultralearning": ["drill", "directness", "feedback", "ultralearn", "intense focus", "self-directed"],
    "atomic_habits": ["habit", "cue", "trigger", "streak", "reward", "1%", "habit loop"],
    "thinking_in_systems": ["feedback loop", "system", "pattern", "leverage point", "flow", "stock"],
    "antifragile": ["stress", "volatility", "resilience", "disorder", "antifragile", "chaos", "fragile"],
}


def detect_frameworks(text: str) -> list[str]:
    """Return list of framework keys whose signals appear in text."""
    text_lower = text.lower()
    return [
        key for key, signals in _FRAMEWORK_SIGNALS.items()
        if any(sig in text_lower for sig in signals)
    ]


def apply_temporal_decay(base_score: int, created_at: datetime) -> float:
    """Apply 2%/week linear decay to a score based on insight age."""
    weeks_old = (datetime.utcnow() - created_at).days / 7
    decay_factor = max(0.0, 1.0 - (0.02 * weeks_old))
    return base_score * decay_factor


def score(event: dict, compressed: str, framework_tags: list[str]) -> int:
    """Compute quality score for an insight. Higher = more worth learning."""
    total = 0

    tool = event.get("tool", "")
    total += _TOOL_WEIGHTS.get(tool, 0)

    result = str(event.get("result", ""))
    if "error" in result.lower() or "fail" in result.lower():
        total += 15

    if event.get("error_recovery"):
        total += 20

    if "create" in result.lower() or "new file" in result.lower():
        total += 12

    if len(result) > 200:
        total += 5

    total += len(framework_tags) * 8

    boilerplate = ["reading", "listing", "no changes", "already exists"]
    if any(b in compressed.lower() for b in boilerplate):
        total -= 10

    return total
