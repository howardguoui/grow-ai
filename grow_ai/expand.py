"""
grow_ai/expand.py — LoRA pair expansion batch job.

Reads all unexpanded insights (lora_pair IS NULL) from SQLite,
calls Qwen2.5-3B via Ollama to generate instruction/output training pairs,
and writes results back to insights.lora_pair.

Decoupled from the live capture hook — runs as a batch job immediately
before Unsloth fine-tuning triggers. Zero impact on capture latency.

CLI:
    python -m grow_ai.expand              # expand all
    python -m grow_ai.expand --dry-run    # preview without writing
    python -m grow_ai.expand --export out/training.jsonl  # expand + export
"""
import json
import sqlite3
import sys
from pathlib import Path

import httpx

from grow_ai.config import cfg
from grow_ai.db import get_connection, init_db, get_unexpanded_insights


# ---------------------------------------------------------------------------
# Framework response mode guidance
# ---------------------------------------------------------------------------

_FRAMEWORK_MODES: dict[str, str] = {
    "thinking_fast_and_slow": "Diagnostic — identify the System 1 bias or shortcut at play and when to invoke System 2",
    "clear_thinking":         "Diagnostic — name the default reaction and provide a concrete override strategy",
    "algorithms_to_live_by":  "Exploratory — apply a computational heuristic (sort, cache, explore-exploit) to the problem",
    "memory_palace":          "Prescriptive — use visualization or spatial association to encode the insight memorably",
    "memory_book":            "Prescriptive — apply a peg, chain, or hook system for durable retention",
    "make_it_stick":          "Prescriptive — frame as active recall, spaced repetition, or interleaved practice",
    "ultralearning":          "Exploratory — apply directness, deliberate drilling, or rapid feedback loops",
    "atomic_habits":          "Prescriptive — design a habit loop: cue → craving → response → reward, with identity framing",
    "thinking_in_systems":    "Structural — map feedback loops, leverage points, stocks and flows at play",
    "antifragile":            "Structural — identify how the stressor or volatility creates a growth opportunity",
}


def _framework_guidance(tags: list[str]) -> str:
    """Return response mode instructions for up to 2 active frameworks."""
    modes = [_FRAMEWORK_MODES[t] for t in tags if t in _FRAMEWORK_MODES]
    if not modes:
        return "Provide a clear, practical answer grounded in software engineering principles."
    return "Response mode: " + "; and ".join(modes[:2])


# ---------------------------------------------------------------------------
# Pair generation via Ollama
# ---------------------------------------------------------------------------

def _generate_pair(compressed: str, full_context: str, framework_tags: list[str]) -> dict:
    """
    Call Qwen2.5-3B to generate one instruction/output LoRA training pair.
    Returns {"instruction": str, "input": "", "output": str}
    Raises on Ollama failure or JSON parse error.
    """
    guidance = _framework_guidance(framework_tags)
    frameworks_str = ", ".join(framework_tags) if framework_tags else "general software engineering"

    prompt = (
        "You are generating training data for a personal AI that reasons using strategic frameworks.\n\n"
        f"DEVELOPER INSIGHT (20-word summary): {compressed}\n\n"
        f"RAW CONTEXT:\n{full_context[:400]}\n\n"
        f"ACTIVE FRAMEWORKS: {frameworks_str}\n"
        f"RESPONSE GUIDANCE: {guidance}\n\n"
        "Generate a JSON training pair:\n"
        '- "instruction": A natural question a developer would ask that this insight answers (1-2 sentences)\n'
        '- "input": always empty string ""\n'
        '- "output": A framework-grounded answer (50-100 words) applying the active frameworks\n\n'
        "Respond ONLY with valid JSON. No markdown, no explanation outside the JSON.\n\n"
        'Example: {"instruction": "How do I debug a React hook causing infinite re-renders?", '
        '"input": "", "output": "Apply Systems Thinking: map the feedback loop first..."}'
    )

    response = httpx.post(
        f"{cfg.ollama_base_url}/api/generate",
        json={"model": cfg.generative_model, "prompt": prompt, "stream": False},
        timeout=60.0,
    )
    response.raise_for_status()
    raw = response.json()["response"].strip()

    # Strip markdown code fences if model wraps JSON
    if "```" in raw:
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else parts[0]
        if raw.startswith("json"):
            raw = raw[4:]

    pair = json.loads(raw.strip())
    pair["input"] = ""  # enforce empty input regardless of model output
    return pair


# ---------------------------------------------------------------------------
# DB write-back
# ---------------------------------------------------------------------------

def _save_lora_pair(conn: sqlite3.Connection, insight_id: int, pair: dict) -> None:
    """Persist the generated pair to insights.lora_pair."""
    conn.execute(
        "UPDATE insights SET lora_pair = ? WHERE id = ?",
        (json.dumps(pair), insight_id),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# JSONL export for Unsloth
# ---------------------------------------------------------------------------

def export_jsonl(conn: sqlite3.Connection, output_path: Path) -> int:
    """
    Write all expanded insights to a JSONL file ready for Unsloth fine-tuning.
    Each line: {"instruction": ..., "input": "", "output": ...}
    Returns count of exported records.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = conn.execute(
        "SELECT lora_pair FROM insights WHERE lora_pair IS NOT NULL"
    ).fetchall()

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(row[0] + "\n")
            count += 1

    return count


# ---------------------------------------------------------------------------
# Main expansion loop
# ---------------------------------------------------------------------------

def run_expansion(conn: sqlite3.Connection, dry_run: bool = False) -> dict:
    """
    Process all unexpanded insights. Calls Ollama for each.
    Returns stats: {processed, failed, skipped, total_unexpanded}
    """
    unexpanded = get_unexpanded_insights(conn)

    if not unexpanded:
        return {"processed": 0, "failed": 0, "total_unexpanded": 0}

    processed = 0
    failed = 0

    for row in unexpanded:
        insight_id = row["id"]
        compressed = row["compressed"]
        full_context = row["full_context"]
        framework_tags = json.loads(row["framework_tags"] or "[]")

        if dry_run:
            print(f"[dry-run] insight {insight_id}: {compressed[:70]}")
            processed += 1
            continue

        try:
            pair = _generate_pair(compressed, full_context, framework_tags)
            _save_lora_pair(conn, insight_id, pair)
            print(f"[expand] ✓ insight {insight_id}: {pair['instruction'][:60]}")
            processed += 1
        except Exception as exc:
            print(f"[expand] ✗ insight {insight_id} failed: {exc}")
            failed += 1

    return {
        "processed": processed,
        "failed": failed,
        "total_unexpanded": len(unexpanded),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    dry_run = "--dry-run" in sys.argv
    export_path: Path | None = None

    if "--export" in sys.argv:
        idx = sys.argv.index("--export")
        if idx + 1 < len(sys.argv):
            export_path = Path(sys.argv[idx + 1])

    conn = get_connection(cfg.db_path)
    init_db(conn)

    print("[grow-ai] Starting LoRA pair expansion...")
    stats = run_expansion(conn, dry_run=dry_run)
    print(f"[grow-ai] Expansion complete: {stats}")

    if export_path and not dry_run:
        count = export_jsonl(conn, export_path)
        print(f"[grow-ai] Exported {count} pairs → {export_path}")

    conn.close()


if __name__ == "__main__":
    main()
