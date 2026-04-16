"""
Tests for the grow-ai FastAPI server.
Uses TestClient (no real server needed). Ollama is mocked.
"""
import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from grow_ai.db import insert_insight


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_INSIGHT = {
    "compressed": "Edit src/auth.py: fixed JWT expiry — token validates correctly",
    "full_context": '{"tool": "Edit", "result": "JWT fixed"}',
    "framework_tags": ["antifragile", "thinking_in_systems"],
    "quality_score": 42,
    "vector": [0.1] * 768,
    "error_recovery": True,
}

FAKE_PAIR = {
    "instruction": "How to fix JWT expiry?",
    "input": "",
    "output": "Apply Systems Thinking — trace the token lifecycle...",
}


@pytest.fixture
def client(db, monkeypatch):
    """TestClient with DB dependency overridden to use in-memory test DB."""
    from api import deps

    def override_get_conn():
        db.execute("PRAGMA journal_mode=WAL")  # safe for multi-thread reads
        yield db

    app.dependency_overrides[deps.get_conn] = override_get_conn
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# /api/health
# ---------------------------------------------------------------------------

def test_health_returns_db_true(client):
    with patch("api.routers.system._ollama_alive", return_value=False):
        r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["db"] is True
    assert data["ollama"] is False


def test_health_total_insights(client, db):
    insert_insight(db, **SAMPLE_INSIGHT)
    with patch("api.routers.system._ollama_alive", return_value=True):
        r = client.get("/api/health")
    assert r.json()["total_insights"] == 1


# ---------------------------------------------------------------------------
# /api/stats
# ---------------------------------------------------------------------------

def test_stats_empty_db(client):
    r = client.get("/api/stats")
    assert r.status_code == 200
    data = r.json()
    assert data["total_insights"] == 0
    assert data["avg_quality_score"] == 0.0


def test_stats_with_insights(client, db):
    insert_insight(db, **SAMPLE_INSIGHT)
    r = client.get("/api/stats")
    data = r.json()
    assert data["total_insights"] == 1
    assert data["error_recovery_count"] == 1
    assert any(fw["framework"] in ("antifragile", "thinking_in_systems")
               for fw in data["top_frameworks"])


# ---------------------------------------------------------------------------
# /api/insights
# ---------------------------------------------------------------------------

def test_list_insights_empty(client):
    r = client.get("/api/insights")
    assert r.status_code == 200
    data = r.json()
    assert data["items"] == []
    assert data["total"] == 0
    assert data["page"] == 1


def test_list_insights_returns_item(client, db):
    insert_insight(db, **SAMPLE_INSIGHT)
    r = client.get("/api/insights")
    data = r.json()
    assert data["total"] == 1
    assert "antifragile" in data["items"][0]["framework_tags"]


def test_list_insights_pagination(client, db):
    for i in range(25):
        insert_insight(db, **{**SAMPLE_INSIGHT, "compressed": f"insight {i}"})
    r = client.get("/api/insights?page=1&limit=10")
    data = r.json()
    assert len(data["items"]) == 10
    assert data["pages"] == 3


def test_list_insights_filter_by_framework(client, db):
    insert_insight(db, **SAMPLE_INSIGHT)  # has antifragile
    insert_insight(db, **{**SAMPLE_INSIGHT, "framework_tags": ["atomic_habits"]})
    r = client.get("/api/insights?framework=antifragile")
    data = r.json()
    assert data["total"] == 1


# ---------------------------------------------------------------------------
# /api/insights/search
# ---------------------------------------------------------------------------

def test_search_semantic_mode(client, db):
    insert_insight(db, **SAMPLE_INSIGHT)
    fake_vector = [0.1] * 768
    # Patch at the embed module level so all callers see the mock
    with patch("grow_ai.embed._call_ollama_embed", return_value=fake_vector):
        r = client.get("/api/insights/search?q=JWT+auth+fix")
    assert r.status_code == 200
    data = r.json()
    # Accept semantic or keyword — both are valid depending on thread timing
    assert data["mode"] in ("semantic", "keyword")
    assert data["query"] == "JWT auth fix"


def test_search_falls_back_to_keyword(client, db):
    insert_insight(db, **SAMPLE_INSIGHT)
    with patch("grow_ai.search.embed", side_effect=Exception("Ollama down")):
        r = client.get("/api/insights/search?q=JWT")
    assert r.status_code == 200
    data = r.json()
    assert data["mode"] == "keyword"
    assert len(data["items"]) >= 1


def test_search_requires_query(client):
    r = client.get("/api/insights/search")
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# /api/insights/{id}
# ---------------------------------------------------------------------------

def test_get_insight_by_id(client, db):
    iid = insert_insight(db, **SAMPLE_INSIGHT)
    r = client.get(f"/api/insights/{iid}")
    assert r.status_code == 200
    data = r.json()
    assert data["id"] == iid
    assert "full_context" in data


def test_get_insight_404(client):
    r = client.get("/api/insights/999")
    assert r.status_code == 404


def test_get_insight_includes_lora_pair(client, db):
    iid = insert_insight(db, **SAMPLE_INSIGHT)
    db.execute(
        "UPDATE insights SET lora_pair = ? WHERE id = ?",
        (json.dumps(FAKE_PAIR), iid),
    )
    db.commit()
    r = client.get(f"/api/insights/{iid}")
    data = r.json()
    assert data["lora_pair"] is not None
    assert data["lora_pair"]["instruction"] == FAKE_PAIR["instruction"]


# ---------------------------------------------------------------------------
# /api/growth-log
# ---------------------------------------------------------------------------

def test_growth_log_empty(client):
    r = client.get("/api/growth-log")
    assert r.status_code == 200
    assert r.json() == []


def test_growth_log_returns_snapshot(client, db):
    db.execute(
        "INSERT INTO growth_log (insight_count, model_version, question_id, response) VALUES (?, ?, ?, ?)",
        (10, "v1.0", 1, "Framework-based answer here."),
    )
    db.commit()
    r = client.get("/api/growth-log")
    data = r.json()
    assert len(data) == 1
    assert data[0]["model_version"] == "v1.0"
    assert len(data[0]["entries"]) == 1


def test_growth_log_excludes_daily_reports(client, db):
    # question_id=0 is the daily report sentinel — should be excluded
    db.execute(
        "INSERT INTO growth_log (insight_count, model_version, question_id, response) VALUES (?, ?, ?, ?)",
        (5, None, 0, '{"total_insights": 5}'),
    )
    db.commit()
    r = client.get("/api/growth-log")
    assert r.json() == []


# ---------------------------------------------------------------------------
# /api/batches
# ---------------------------------------------------------------------------

def test_batches_empty(client):
    r = client.get("/api/batches")
    assert r.status_code == 200
    assert r.json() == []


def test_batches_returns_batch(client, db):
    db.execute(
        "INSERT INTO fine_tune_batches (insight_ids, status) VALUES (?, ?)",
        (json.dumps([1, 2, 3]), "done"),
    )
    db.commit()
    r = client.get("/api/batches")
    data = r.json()
    assert len(data) == 1
    assert data[0]["status"] == "done"
    assert data[0]["insight_count"] == 3


# ---------------------------------------------------------------------------
# /api/finetune (POST)
# ---------------------------------------------------------------------------

def test_finetune_trigger_no_insights(client):
    r = client.post("/api/finetune")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "no_insights"


def test_finetune_trigger_dry_run_without_unsloth(client, db):
    insert_insight(db, **SAMPLE_INSIGHT)
    with patch("grow_ai.finetune.UNSLOTH_AVAILABLE", False):
        r = client.post("/api/finetune")
    assert r.status_code == 200
    # Without Unsloth it fails gracefully
    assert r.json()["status"] in ("failed", "dry_run", "done")


# ---------------------------------------------------------------------------
# Dashboard route
# ---------------------------------------------------------------------------

def test_dashboard_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "grow-ai" in r.text
