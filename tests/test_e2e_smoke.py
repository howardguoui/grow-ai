"""
E2E smoke test: exercises the full capture pipeline with real Ollama.
Skips automatically if Ollama is not reachable at localhost:11434.
"""
import json
import sqlite3
import pytest
import httpx
import sqlite_vec

from grow_ai.db import init_db, get_all_insights
from grow_ai.capture import run_pipeline


def ollama_available() -> bool:
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not ollama_available(),
    reason="Ollama not running at localhost:11434 — skipping E2E smoke test",
)


@pytest.fixture
def real_db():
    conn = sqlite3.connect(":memory:")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    init_db(conn)
    yield conn
    conn.close()


def test_full_pipeline_stores_high_signal_event(real_db):
    """A high-signal Edit event should survive the full pipeline and be stored."""
    event = {
        "tool": "Edit",
        "action": "fixed infinite re-render bug in React hook by removing stale closure",
        "result": "success",
        "timestamp": "2026-04-14T10:00:00",
        "session_id": "smoke-test-session",
        "error_recovery": True,
    }
    run_pipeline(real_db, event)
    insights = get_all_insights(real_db)
    assert len(insights) >= 1, "Expected at least one insight stored for a high-signal event"


def test_full_pipeline_discards_low_signal_event(real_db):
    """A low-signal Read event should be filtered out by the quality scorer."""
    event = {
        "tool": "Read",
        "action": "read file",
        "result": "ok",
        "timestamp": "2026-04-14T10:01:00",
        "session_id": "smoke-test-session",
        "error_recovery": False,
    }
    run_pipeline(real_db, event)
    insights = get_all_insights(real_db)
    assert len(insights) == 0, "Expected low-signal Read event to be discarded"


def test_full_pipeline_dedup_discards_near_duplicate(real_db):
    """Submitting the same high-signal event twice should result in only 1 stored insight."""
    event = {
        "tool": "Write",
        "action": "created new authentication module with JWT token validation and refresh logic",
        "result": "success",
        "timestamp": "2026-04-14T10:02:00",
        "session_id": "smoke-test-session",
        "error_recovery": False,
    }
    run_pipeline(real_db, event)
    run_pipeline(real_db, event)
    insights = get_all_insights(real_db)
    assert len(insights) == 1, "Expected dedup to prevent duplicate storage of identical event"
