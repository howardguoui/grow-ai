import json
import sys
import sqlite3
from pathlib import Path

from grow_ai.config import cfg
from grow_ai.db import get_connection, init_db, insert_insight, update_reinforcement
from grow_ai.scrubber import scrub
from grow_ai import compress, embed, dedup, scorer


def run_pipeline(conn: sqlite3.Connection, event: dict) -> None:
    """
    Full capture pipeline:
    scrub → compress → embed → dedup → score → store
    """
    # 1. Scrub secrets from all text fields
    clean_event = {
        k: scrub(str(v)) if isinstance(v, str) else v
        for k, v in event.items()
    }
    full_context = scrub(json.dumps(event))

    # 2. Compress to 20-word insight
    compressed = compress.compress(clean_event)

    # 3. Embed
    vector = embed.embed(compressed)

    # 4. Semantic dedup
    decision, existing_id = dedup.check(conn, compressed, vector)

    if decision == "discard":
        return

    if decision == "merge" and existing_id is not None:
        update_reinforcement(conn, existing_id, score_delta=3)
        return

    # 5. Score
    framework_tags = scorer.detect_frameworks(compressed + " " + str(clean_event.get("result", "")))
    quality = scorer.score(clean_event, compressed, framework_tags)

    if quality < cfg.quality_threshold:
        return

    # 6. Store
    insert_insight(
        conn,
        compressed=compressed,
        full_context=full_context,
        framework_tags=framework_tags,
        quality_score=quality,
        vector=vector,
        error_recovery=bool(event.get("error_recovery", False)),
    )


def main() -> None:
    """Entry point for Claude Code PostToolUse hook. Reads JSON from stdin."""
    try:
        raw = sys.stdin.read()
        event = json.loads(raw)
    except (json.JSONDecodeError, EOFError):
        return  # Malformed input — fail silently, never block Claude Code

    conn = get_connection(cfg.db_path)
    init_db(conn)

    try:
        run_pipeline(conn, event)
    except Exception:
        pass  # Never crash — Claude Code must not be interrupted
    finally:
        conn.close()


if __name__ == "__main__":
    main()
