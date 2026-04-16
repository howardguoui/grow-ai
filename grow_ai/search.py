"""
grow_ai/search.py — Semantic search over stored insights.

Uses nomic-embed-text (via Ollama) to embed the query, then computes cosine
similarity against all stored vectors. Falls back to full-text keyword search
if Ollama is unreachable.
"""
import sqlite3
from typing import Any

import sqlite_vec

from grow_ai.embed import embed, cosine_similarity


def semantic_search(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 10,
) -> list[dict]:
    """
    Embed query → rank all insights by cosine similarity → return top-N.
    Returns list of dicts with insight fields + 'similarity' score.
    Raises RuntimeError if Ollama is unreachable.
    """
    query_vector = embed(query)  # raises on Ollama failure

    rows = conn.execute(
        """SELECT i.id, i.compressed, i.full_context, i.framework_tags,
                  i.quality_score, i.reinforcement_count, i.error_recovery,
                  i.created_at, iv.embedding
           FROM insight_vectors iv
           JOIN insights i ON iv.insight_id = i.id"""
    ).fetchall()

    if not rows:
        return []

    scored: list[tuple[float, dict]] = []
    for row in rows:
        stored_vector = list(sqlite_vec.deserialize_float32(row["embedding"]))
        sim = cosine_similarity(query_vector, stored_vector)
        insight = {
            "id": row["id"],
            "compressed": row["compressed"],
            "framework_tags": row["framework_tags"],
            "quality_score": row["quality_score"],
            "reinforcement_count": row["reinforcement_count"],
            "error_recovery": bool(row["error_recovery"]),
            "created_at": row["created_at"],
            "similarity": round(sim, 4),
        }
        scored.append((sim, insight))

    scored.sort(key=lambda x: -x[0])
    return [item for _, item in scored[:limit]]


def keyword_search(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 10,
) -> list[dict]:
    """
    Full-text LIKE search over compressed text. Fallback when Ollama is down.
    """
    pattern = f"%{query}%"
    rows = conn.execute(
        """SELECT id, compressed, framework_tags, quality_score,
                  reinforcement_count, error_recovery, created_at
           FROM insights
           WHERE compressed LIKE ?
           ORDER BY quality_score DESC
           LIMIT ?""",
        (pattern, limit),
    ).fetchall()

    return [
        {
            "id": row["id"],
            "compressed": row["compressed"],
            "framework_tags": row["framework_tags"],
            "quality_score": row["quality_score"],
            "reinforcement_count": row["reinforcement_count"],
            "error_recovery": bool(row["error_recovery"]),
            "created_at": row["created_at"],
            "similarity": None,  # not available for keyword search
        }
        for row in rows
    ]


def search(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 10,
) -> tuple[list[dict], str]:
    """
    Try semantic search first; fall back to keyword search if Ollama fails.
    Returns (results, mode) where mode is 'semantic' or 'keyword'.
    """
    try:
        results = semantic_search(conn, query, limit)
        return results, "semantic"
    except Exception:
        results = keyword_search(conn, query, limit)
        return results, "keyword"
