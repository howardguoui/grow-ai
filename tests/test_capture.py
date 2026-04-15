import json
from grow_ai.capture import run_pipeline
from grow_ai.db import get_all_insights


SAMPLE_EVENT = {
    "tool": "Edit",
    "action": "src/auth.py",
    "result": "Fixed critical authentication error — added JWT token validation, tests pass, pattern solved using thinking in systems approach. Changed 150+ lines, implemented recursive caching and explored exploit patterns for security validation.",
    "session_id": "test-session-001",
    "error_recovery": True,
}


def test_pipeline_stores_high_quality_insight(db, mock_embed, mock_ollama_compress):
    run_pipeline(db, SAMPLE_EVENT)
    rows = get_all_insights(db)
    assert len(rows) == 1


def test_pipeline_discards_low_quality_event(db, mock_embed, mock_ollama_compress):
    low_signal = {"tool": "Read", "action": "README.md", "result": "ok", "error_recovery": False}
    run_pipeline(db, low_signal)
    rows = get_all_insights(db)
    assert len(rows) == 0


def test_pipeline_scrubs_secrets_before_storing(db, mock_embed, mock_ollama_compress):
    event = {**SAMPLE_EVENT, "result": "Used key sk-abc123abcdefghijklmnopqrstuvwxyz to call API"}
    run_pipeline(db, event)
    rows = get_all_insights(db)
    if rows:
        assert "sk-abc123" not in rows[0]["compressed"]
        assert "sk-abc123" not in rows[0]["full_context"]
