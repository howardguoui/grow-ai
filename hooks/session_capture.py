#!/usr/bin/env python3
"""
session_capture.py — SessionStart hook for grow-ai.

Scans yesterday's Claude Code transcript (.jsonl) files, extracts key
sentences using YAKE + Sumy (zero LLM cost, zero API calls), then
embeds and stores insights via the grow_ai pipeline.

Pipeline:
  1. Find .jsonl files modified in the last 24h (skip agent subagent files)
  2. Extract clean user + assistant text (skip tool_use / tool_result blocks)
  3. YAKE  → top keywords
  4. Sumy LexRank → 3-5 ranked sentences containing those keywords
  5. Truncate each sentence to 20 words (no LLM)
  6. embed → dedup → score → store

Total cost: $0 — local Python + nomic-embed-text (Ollama).
Target runtime: <8 seconds on typical daily volume.
"""

import sys
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, r"E:\ClaudeProject\grow-ai")

# ── grow-ai imports ────────────────────────────────────────────────────────
from grow_ai.config import cfg
from grow_ai.db import get_connection, init_db, insert_insight, update_reinforcement
from grow_ai.scrubber import scrub
from grow_ai import embed, dedup, scorer

# ── NLP imports ────────────────────────────────────────────────────────────
import yake
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# ── Constants ──────────────────────────────────────────────────────────────
PROJECTS_DIR = Path(r"C:\Users\c4999\.claude\projects")
LOOKBACK_HOURS = 28          # slightly > 24h to avoid missing edge cases
MAX_FILES = 10               # cap to keep runtime bounded
MAX_INSIGHTS_PER_FILE = 5
MIN_SENTENCE_WORDS = 8
QUALITY_FLOOR = 5            # lower threshold for session summaries


# ── Text extraction ────────────────────────────────────────────────────────

def extract_text_from_transcript(path: Path) -> str:
    """
    Parse a .jsonl transcript and return clean user + assistant text.
    Skips: tool_use blocks, tool_result blocks, system preambles, short lines.
    """
    sentences = []

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
        if role not in ("user", "assistant"):
            continue

        content = msg.get("content", "")

        # Content is either a plain string or a list of typed blocks
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            parts = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                # Only grab pure text blocks — skip tool_use / tool_result
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
            text = " ".join(parts).strip()
        else:
            continue

        # Filter noise
        if len(text) < 40:
            continue
        if text.startswith("<local-command-caveat>"):
            continue
        if text.startswith("<system-reminder>"):
            continue

        # Strip markdown code fences — they confuse sentence tokenizers
        text = re.sub(r"```[\s\S]*?```", "", text)
        text = re.sub(r"`[^`]+`", "", text)
        text = text.strip()

        if len(text) < 40:
            continue

        sentences.append(text)

    return " ".join(sentences)


# ── Summarisation ──────────────────────────────────────────────────────────

def extract_insights(text: str, max_sentences: int = MAX_INSIGHTS_PER_FILE) -> list[str]:
    """
    Use YAKE keywords + Sumy LexRank to extract the most informative sentences.
    Returns up to max_sentences strings, each trimmed to 20 words.
    """
    if not text or len(text.split()) < 30:
        return []

    # Step 1: YAKE keyword extraction
    kw_extractor = yake.KeywordExtractor(
        lan="en", n=2, dedupLim=0.7, top=15, features=None
    )
    keywords_raw = kw_extractor.extract_keywords(text)
    # YAKE scores: lower = more important
    keyword_set = {kw.lower() for kw, _ in keywords_raw}

    # Step 2: Sumy LexRank sentence ranking
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    # Ask for more than needed so we can filter by keyword overlap
    ranked = summarizer(parser.document, sentences_count=min(10, max_sentences * 2))

    # Step 3: prefer sentences that contain YAKE keywords
    scored = []
    for sentence in ranked:
        s = str(sentence).strip()
        words = s.split()
        if len(words) < MIN_SENTENCE_WORDS:
            continue
        overlap = sum(1 for kw in keyword_set if kw in s.lower())
        scored.append((overlap, s))

    scored.sort(key=lambda x: -x[0])
    top = [s for _, s in scored[:max_sentences]]

    # Step 4: trim each to 20 words
    trimmed = []
    for s in top:
        words = s.split()
        if len(words) > 20:
            s = " ".join(words[:20])
        trimmed.append(s)

    return trimmed


# ── Capture pipeline ───────────────────────────────────────────────────────

def capture_insight(conn, text: str) -> str:
    """
    Run one extracted sentence through embed → dedup → score → store.
    Returns 'stored', 'merged', 'discarded', or 'filtered'.
    """
    clean = scrub(text)
    event = {"tool_name": "session_summary", "text": clean}

    vector = embed.embed(clean)
    decision, existing_id = dedup.check(conn, clean, vector)

    if decision == "discard":
        return "discarded"
    if decision == "merge" and existing_id is not None:
        update_reinforcement(conn, existing_id, score_delta=2)
        return "merged"

    framework_tags = scorer.detect_frameworks(clean)
    # session_summary bypasses tool-weight scoring — YAKE+Sumy is the quality gate
    quality = scorer.score(event, clean, framework_tags)
    quality = max(quality, QUALITY_FLOOR)  # always store if it passed NLP ranking

    insert_insight(
        conn,
        compressed=clean,
        full_context=scrub(json.dumps(event)),
        framework_tags=framework_tags,
        quality_score=quality,
        vector=vector,
        error_recovery=False,
    )
    return "stored"


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    # Read SessionStart event (we only care that it fired — no fields needed)
    try:
        sys.stdin.read()
    except Exception:
        pass

    cutoff = datetime.now() - timedelta(hours=LOOKBACK_HOURS)

    # Find eligible transcripts
    candidates = []
    for p in PROJECTS_DIR.rglob("*.jsonl"):
        # Skip subagent files (agent-* names are very noisy tool traces)
        if p.stem.startswith("agent-"):
            continue
        try:
            mtime = datetime.fromtimestamp(p.stat().st_mtime)
        except OSError:
            continue
        if mtime >= cutoff:
            candidates.append((mtime, p))

    # Most-recent first, capped
    candidates.sort(key=lambda x: -x[0].timestamp())
    candidates = candidates[:MAX_FILES]

    if not candidates:
        sys.exit(0)

    conn = get_connection(cfg.db_path)
    init_db(conn)

    totals = {"stored": 0, "merged": 0, "discarded": 0, "filtered": 0}

    for mtime, path in candidates:
        try:
            text = extract_text_from_transcript(path)
            insights = extract_insights(text)
            for insight in insights:
                result = capture_insight(conn, insight)
                totals[result] += 1
        except Exception:
            continue  # never crash — SessionStart must not block Claude Code

    conn.close()

    total_stored = totals["stored"] + totals["merged"]
    if total_stored > 0:
        print(
            f"[grow-ai] Session capture: {totals['stored']} new, "
            f"{totals['merged']} reinforced, "
            f"{totals['discarded']} discarded, "
            f"{totals['filtered']} below threshold "
            f"(from {len(candidates)} transcript(s))"
        )

    sys.exit(0)


if __name__ == "__main__":
    main()
