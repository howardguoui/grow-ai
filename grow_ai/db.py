import json
import sqlite3
from pathlib import Path
import sqlite_vec


def get_connection(db_path: Path, check_same_thread: bool = True) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS insights (
            id                  INTEGER PRIMARY KEY,
            compressed          TEXT NOT NULL,
            full_context        TEXT NOT NULL,
            framework_tags      TEXT NOT NULL DEFAULT '[]',
            quality_score       INTEGER NOT NULL DEFAULT 0,
            reinforcement_count INTEGER NOT NULL DEFAULT 0,
            error_recovery      INTEGER NOT NULL DEFAULT 0,
            lora_pair           TEXT DEFAULT NULL,
            created_at          DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS insight_vectors USING vec0(
            insight_id INTEGER PRIMARY KEY,
            embedding  float[768]
        );

        CREATE TABLE IF NOT EXISTS fine_tune_batches (
            id            INTEGER PRIMARY KEY,
            insight_ids   TEXT NOT NULL DEFAULT '[]',
            triggered_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
            completed_at  DATETIME,
            model_version TEXT,
            status        TEXT NOT NULL DEFAULT 'queued'
        );

        CREATE TABLE IF NOT EXISTS growth_log (
            id            INTEGER PRIMARY KEY,
            insight_count INTEGER NOT NULL,
            model_version TEXT,
            question_id   INTEGER NOT NULL,
            response      TEXT NOT NULL,
            recorded_at   DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()


def insert_insight(
    conn: sqlite3.Connection,
    compressed: str,
    full_context: str,
    framework_tags: list[str],
    quality_score: int,
    vector: list[float],
    error_recovery: bool,
) -> int:
    cur = conn.execute(
        """INSERT INTO insights
           (compressed, full_context, framework_tags, quality_score, error_recovery)
           VALUES (?, ?, ?, ?, ?)""",
        (compressed, full_context, json.dumps(framework_tags), quality_score, int(error_recovery)),
    )
    insight_id = cur.lastrowid
    conn.execute(
        "INSERT INTO insight_vectors (insight_id, embedding) VALUES (?, ?)",
        (insight_id, sqlite_vec.serialize_float32(vector)),
    )
    conn.commit()
    return insight_id


def get_all_insights(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM insights ORDER BY id").fetchall()


def update_reinforcement(conn: sqlite3.Connection, insight_id: int, score_delta: int) -> None:
    conn.execute(
        """UPDATE insights
           SET reinforcement_count = reinforcement_count + 1,
               quality_score = quality_score + ?
           WHERE id = ?""",
        (score_delta, insight_id),
    )
    conn.commit()


def get_unexpanded_insights(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM insights WHERE lora_pair IS NULL ORDER BY id"
    ).fetchall()
