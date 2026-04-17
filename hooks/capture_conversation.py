#!/usr/bin/env python3
"""
capture_conversation.py — Stop hook that captures the last user+assistant
exchange into grow-ai's insight DB.

Reads the transcript JSONL to extract:
  - last user text message
  - last assistant text response
Then runs both through the standard grow_ai capture pipeline.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, r"E:\ClaudeProject\grow-ai")

from grow_ai.config import cfg
from grow_ai.db import get_connection, init_db, insert_insight, update_reinforcement
from grow_ai.scrubber import scrub
from grow_ai import compress, embed, dedup, scorer


def extract_text(content) -> str:
    """Extract plain text from a message content field (str or list of blocks)."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return " ".join(parts).strip()
    return ""


def get_last_exchange(transcript_path: str) -> tuple[str, str]:
    """
    Read the transcript JSONL and return (last_user_text, last_assistant_text).
    Skips tool-only messages (no text content).
    """
    path = Path(transcript_path)
    if not path.exists():
        return "", ""

    last_user = ""
    last_assistant = ""

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        msg = entry.get("message", {})
        role = msg.get("role", "")
        content = msg.get("content", "")
        text = extract_text(content)

        if not text:
            continue

        if role == "user":
            # Skip tool_result-only user turns
            if isinstance(content, list):
                has_real_text = any(
                    b.get("type") == "text" for b in content if isinstance(b, dict)
                )
                if not has_real_text:
                    continue
            last_user = text
        elif role == "assistant":
            last_assistant = text

    return last_user, last_assistant


def capture_text(conn, source: str, text: str) -> None:
    """Run a single text through the full capture pipeline."""
    if not text or len(text) < 20:
        return

    clean = scrub(text)
    event = {"tool_name": "conversation", "source": source, "text": clean}

    try:
        compressed = compress.compress(event)
        vector = embed.embed(compressed)
        decision, existing_id = dedup.check(conn, compressed, vector)

        if decision == "discard":
            return
        if decision == "merge" and existing_id is not None:
            update_reinforcement(conn, existing_id, score_delta=2)
            return

        framework_tags = scorer.detect_frameworks(compressed + " " + clean)
        quality = scorer.score(event, compressed, framework_tags)

        # Lower threshold for conversation — scored differently than tool use
        if quality < 5:
            return

        insert_insight(
            conn,
            compressed=compressed,
            full_context=scrub(json.dumps(event)),
            framework_tags=framework_tags,
            quality_score=quality,
            vector=vector,
            error_recovery=False,
        )
    except Exception:
        pass  # Never crash — must not interrupt Claude Code


def main() -> None:
    try:
        raw = sys.stdin.read()
        event = json.loads(raw)
    except (json.JSONDecodeError, EOFError):
        sys.exit(0)

    transcript_path = event.get("transcript_path", "")
    if not transcript_path:
        sys.exit(0)

    last_user, last_assistant = get_last_exchange(transcript_path)

    conn = get_connection(cfg.db_path)
    init_db(conn)

    try:
        capture_text(conn, "user", last_user)
        capture_text(conn, "assistant", last_assistant)
    finally:
        conn.close()

    sys.exit(0)


if __name__ == "__main__":
    main()
