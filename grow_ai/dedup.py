import sqlite3
import struct
import sqlite_vec
from grow_ai.config import cfg
from grow_ai.embed import cosine_similarity


def _deserialize_float32(blob: bytes) -> list[float]:
    """Deserialize a raw float32 blob into a Python list of floats."""
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def _decide(similarity: float) -> str:
    """Map similarity score to store/merge/discard decision."""
    if similarity >= cfg.dedup_discard:
        return "discard"
    if similarity >= cfg.dedup_merge:
        return "merge"
    return "store"


def check(
    conn: sqlite3.Connection,
    text: str,
    vector: list[float],
) -> tuple[str, int | None]:
    """
    Compare vector against all stored embeddings via sqlite-vec.
    Returns ("store"|"merge"|"discard", most_similar_insight_id | None).
    """
    rows = conn.execute(
        "SELECT insight_id, embedding FROM insight_vectors"
    ).fetchall()

    if not rows:
        return "store", None

    best_id = None
    best_similarity = -1.0

    for row in rows:
        insight_id, embedding_bytes = row
        stored_vector = _deserialize_float32(embedding_bytes)
        similarity = cosine_similarity(vector, stored_vector)

        if similarity > best_similarity:
            best_similarity = similarity
            best_id = insight_id

    decision = _decide(best_similarity)
    return decision, best_id if decision != "store" else None
