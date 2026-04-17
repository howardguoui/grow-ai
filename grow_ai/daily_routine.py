"""
grow_ai/daily_routine.py — Consolidated daily maintenance job.

Pipeline (sequential, fault-tolerant):
  Step 1: Ollama health check  (gate — skips embedding steps if down)
  Step 2: Dedup maintenance    (merge/discard near-duplicate insights)
  Step 3: Decay report         (compute effective scores; read-only, no DB write to avoid compounding)
  Step 4: Fine-tune trigger    (queue batch if 50+ insights or 7 days since last)
  Step 5: Growth report        (SQLite stats — zero token cost, logged to growth_log)

Token cost: ZERO — all steps are local Python + SQLite + Ollama (local).
No external Claude API calls are made.
"""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import struct

import httpx

from grow_ai.config import cfg
from grow_ai.db import get_connection, init_db, update_reinforcement
from grow_ai.embed import cosine_similarity


def _deserialize_float32(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))
from grow_ai.expand import run_expansion
from grow_ai.finetune import run_finetune
from grow_ai.scorer import apply_temporal_decay


# ---------------------------------------------------------------------------
# Step 1: Ollama health check
# ---------------------------------------------------------------------------

def check_ollama() -> bool:
    """Return True if local Ollama is reachable."""
    try:
        resp = httpx.get(f"{cfg.ollama_base_url}/api/tags", timeout=5.0)
        return resp.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Step 2: Dedup maintenance
# ---------------------------------------------------------------------------

def run_dedup_maintenance(conn: sqlite3.Connection) -> dict:
    """
    Scan all stored insight pairs for semantic overlap.
    - similarity >= dedup_discard (0.95): delete lower-quality duplicate
    - similarity >= dedup_merge  (0.80): merge (reinforce higher, delete lower)

    Rows are fetched quality DESC so id_a is always the higher-quality insight.
    O(n²) — acceptable for personal use (hundreds, not millions, of insights).
    """
    rows = conn.execute(
        """SELECT iv.insight_id, iv.embedding
           FROM insight_vectors iv
           JOIN insights i ON iv.insight_id = i.id
           ORDER BY i.quality_score DESC"""
    ).fetchall()

    # Deserialise all vectors up front
    vectors: list[tuple[int, list[float]]] = [
        (row[0], _deserialize_float32(row[1]))
        for row in rows
    ]

    merged = 0
    discarded = 0
    processed: set[int] = set()

    for i, (id_a, vec_a) in enumerate(vectors):
        if id_a in processed:
            continue
        for id_b, vec_b in vectors[i + 1:]:
            if id_b in processed:
                continue
            sim = cosine_similarity(vec_a, vec_b)
            if sim >= cfg.dedup_discard:
                conn.execute("DELETE FROM insight_vectors WHERE insight_id = ?", (id_b,))
                conn.execute("DELETE FROM insights WHERE id = ?", (id_b,))
                processed.add(id_b)
                discarded += 1
            elif sim >= cfg.dedup_merge:
                update_reinforcement(conn, id_a, score_delta=2)
                conn.execute("DELETE FROM insight_vectors WHERE insight_id = ?", (id_b,))
                conn.execute("DELETE FROM insights WHERE id = ?", (id_b,))
                processed.add(id_b)
                merged += 1

    conn.commit()
    return {"merged": merged, "discarded": discarded}


# ---------------------------------------------------------------------------
# Step 3: Decay report (read-only)
# ---------------------------------------------------------------------------

def run_decay_report(conn: sqlite3.Connection) -> dict:
    """
    Compute how many insights would be below quality_threshold after temporal decay.
    Read-only — does NOT write back to DB to avoid compounding decay on each run.
    A full schema change (base_score vs effective_score columns) is needed before
    making this a write operation.
    """
    rows = conn.execute("SELECT id, quality_score, created_at FROM insights").fetchall()

    below_threshold = 0
    total = len(rows)

    for row in rows:
        created_str = row["created_at"]
        base = row["quality_score"]
        try:
            created_at = datetime.fromisoformat(created_str).replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue
        effective = apply_temporal_decay(base, created_at)
        if effective < cfg.quality_threshold:
            below_threshold += 1

    return {
        "total_insights": total,
        "below_threshold_after_decay": below_threshold,
        "note": "read-only — no DB writes to avoid compounding decay",
    }


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _has_queued_batch(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT COUNT(*) FROM fine_tune_batches WHERE status = 'queued'"
    ).fetchone()
    return row[0] > 0


# ---------------------------------------------------------------------------
# Step 4: Fine-tune batch trigger
# ---------------------------------------------------------------------------

def run_finetune_check(conn: sqlite3.Connection) -> dict:
    """
    Queue a fine_tune_batch if:
      - total insights >= finetune_batch_size (50), OR
      - 7+ days have passed since the last triggered batch.
    """
    total = conn.execute("SELECT COUNT(*) FROM insights").fetchone()[0]

    last_row = conn.execute(
        "SELECT triggered_at FROM fine_tune_batches ORDER BY triggered_at DESC LIMIT 1"
    ).fetchone()

    should_trigger = False
    reason = ""

    # Threshold check
    already_queued = conn.execute(
        "SELECT COUNT(*) FROM fine_tune_batches"
    ).fetchone()[0]
    if total >= cfg.finetune_batch_size * (already_queued + 1):
        should_trigger = True
        reason = f"{total} insights crossed batch threshold ({cfg.finetune_batch_size * (already_queued + 1)})"

    # 7-day cadence check
    if not should_trigger and last_row:
        try:
            last_triggered = datetime.fromisoformat(last_row[0])
            days_since = (datetime.now() - last_triggered).days
            if days_since >= 7:
                should_trigger = True
                reason = f"{days_since} days since last batch"
        except (ValueError, TypeError):
            pass

    if should_trigger:
        insight_ids = [r[0] for r in conn.execute("SELECT id FROM insights").fetchall()]
        conn.execute(
            "INSERT INTO fine_tune_batches (insight_ids, status) VALUES (?, ?)",
            (json.dumps(insight_ids), "queued"),
        )
        conn.commit()
        return {"triggered": True, "reason": reason, "insight_count": total}

    return {"triggered": False, "total_insights": total}


# ---------------------------------------------------------------------------
# Step 5: Growth report
# ---------------------------------------------------------------------------

def run_growth_report(conn: sqlite3.Connection) -> dict:
    """
    Aggregate SQLite stats and log to growth_log.
    Zero token cost — no external API calls.
    question_id=0 is the sentinel for daily maintenance reports.
    """
    total = conn.execute("SELECT COUNT(*) FROM insights").fetchone()[0]

    new_24h = conn.execute(
        "SELECT COUNT(*) FROM insights WHERE created_at >= datetime('now', '-1 day')"
    ).fetchone()[0]

    avg_score = conn.execute("SELECT AVG(quality_score) FROM insights").fetchone()[0] or 0.0

    high_quality = conn.execute(
        "SELECT COUNT(*) FROM insights WHERE quality_score >= 50"
    ).fetchone()[0]

    error_recoveries = conn.execute(
        "SELECT COUNT(*) FROM insights WHERE error_recovery = 1"
    ).fetchone()[0]

    reinforced = conn.execute(
        "SELECT COUNT(*) FROM insights WHERE reinforcement_count > 0"
    ).fetchone()[0]

    # Framework distribution
    tag_rows = conn.execute("SELECT framework_tags FROM insights").fetchall()
    framework_counts: dict[str, int] = {}
    for row in tag_rows:
        for tag in json.loads(row[0] or "[]"):
            framework_counts[tag] = framework_counts.get(tag, 0) + 1
    top_frameworks = sorted(framework_counts.items(), key=lambda x: -x[1])[:3]

    report = {
        "total_insights": total,
        "new_last_24h": new_24h,
        "avg_quality_score": round(float(avg_score), 1),
        "high_quality_count": high_quality,
        "error_recovery_insights": error_recoveries,
        "reinforced_insights": reinforced,
        "top_frameworks": [{"framework": f, "count": c} for f, c in top_frameworks],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    conn.execute(
        "INSERT INTO growth_log (insight_count, question_id, response) VALUES (?, ?, ?)",
        (total, 0, json.dumps(report)),
    )
    conn.commit()
    return report


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the consolidated daily maintenance routine."""
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    started = datetime.now().isoformat()
    print(f"[grow-ai] Daily routine started at {started}")

    conn = get_connection(cfg.db_path)
    init_db(conn)

    results: dict = {"started_at": started}

    # ── Step 1: Ollama health check ──────────────────────────────────────────
    ollama_ok = check_ollama()
    results["ollama_healthy"] = ollama_ok
    if not ollama_ok:
        print("[grow-ai] WARNING: Ollama unreachable — skipping dedup maintenance")

    # ── Step 2: Dedup maintenance (needs Ollama vectors) ────────────────────
    if ollama_ok:
        try:
            results["dedup"] = run_dedup_maintenance(conn)
            print(f"[grow-ai] Dedup: {results['dedup']}")
        except Exception as exc:
            results["dedup"] = {"error": str(exc)}
            print(f"[grow-ai] Dedup failed: {exc}")

    # ── Step 3: Decay report (pure SQLite) ──────────────────────────────────
    try:
        results["decay"] = run_decay_report(conn)
        print(f"[grow-ai] Decay report: {results['decay']}")
    except Exception as exc:
        results["decay"] = {"error": str(exc)}
        print(f"[grow-ai] Decay report failed: {exc}")

    # ── Step 4a: Expand unexpanded insights (needs Ollama) ──────────────────
    if ollama_ok:
        try:
            results["expansion"] = run_expansion(conn)
            print(f"[grow-ai] Expansion: {results['expansion']}")
        except Exception as exc:
            results["expansion"] = {"error": str(exc)}
            print(f"[grow-ai] Expansion failed: {exc}")

    # ── Step 4b: Fine-tune check + trigger (pure SQLite + GPU if available) ──
    try:
        queue_result = run_finetune_check(conn)
        results["finetune_queue"] = queue_result
        print(f"[grow-ai] Fine-tune check: {queue_result}")

        # If a batch was just queued (or one was already pending), run training
        if queue_result.get("triggered") or _has_queued_batch(conn):
            print("[grow-ai] Queued batch detected — starting fine-tune...")
            ft_result = run_finetune(conn)
            results["finetune_train"] = ft_result
            print(f"[grow-ai] Fine-tune: {ft_result}")
    except Exception as exc:
        results["finetune_queue"] = {"error": str(exc)}
        print(f"[grow-ai] Fine-tune step failed: {exc}")

    # ── Step 5: Growth report (pure SQLite) ─────────────────────────────────
    try:
        results["report"] = run_growth_report(conn)
        print(f"[grow-ai] Growth report: {results['report']}")
    except Exception as exc:
        results["report"] = {"error": str(exc)}
        print(f"[grow-ai] Growth report failed: {exc}")

    conn.close()

    finished = datetime.now().isoformat()
    results["finished_at"] = finished
    print(f"[grow-ai] Daily routine complete at {finished}")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
