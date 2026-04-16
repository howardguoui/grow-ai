"""
Tests for grow_ai/expand.py — LoRA pair expansion batch job.
Ollama calls are always mocked; no real model required.
"""
import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from grow_ai.expand import (
    _framework_guidance,
    _generate_pair,
    _save_lora_pair,
    export_jsonl,
    run_expansion,
)
from grow_ai.db import insert_insight, get_unexpanded_insights


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_PAIR = {
    "instruction": "How do I debug an infinite re-render in React?",
    "input": "",
    "output": (
        "Apply Systems Thinking: map the feedback loop first. "
        "Identify what triggers the effect, what the effect mutates, "
        "and whether that mutation re-triggers. Add one log per cycle "
        "and let the pattern emerge before committing to a fix."
    ),
}

SAMPLE_INSIGHT = {
    "compressed": "Edit src/app.py: fixed JWT expiry — token validates correctly now",
    "full_context": '{"tool": "Edit", "action": "src/app.py", "result": "JWT fixed"}',
    "framework_tags": ["antifragile", "thinking_in_systems"],
    "quality_score": 35,
    "vector": [0.1] * 768,
    "error_recovery": True,
}


@pytest.fixture
def mock_generate(mocker):
    """Mock _generate_pair to return FAKE_PAIR without calling Ollama."""
    return mocker.patch("grow_ai.expand._generate_pair", return_value=FAKE_PAIR.copy())


# ---------------------------------------------------------------------------
# _framework_guidance
# ---------------------------------------------------------------------------

def test_framework_guidance_returns_string_for_known_tag():
    result = _framework_guidance(["atomic_habits"])
    assert "habit" in result.lower() or "cue" in result.lower() or "prescriptive" in result.lower()


def test_framework_guidance_uses_at_most_two_frameworks():
    tags = ["atomic_habits", "thinking_in_systems", "antifragile"]
    result = _framework_guidance(tags)
    # Should not include the third framework (antifragile → "volatile/stress")
    # antifragile's mode contains "stressor" — should be absent if capped at 2
    assert result.count("; and ") <= 1  # at most one joiner = at most 2 modes


def test_framework_guidance_fallback_for_empty_tags():
    result = _framework_guidance([])
    assert "software engineering" in result.lower()


def test_framework_guidance_fallback_for_unknown_tags():
    result = _framework_guidance(["unknown_framework_xyz"])
    assert "software engineering" in result.lower()


# ---------------------------------------------------------------------------
# _generate_pair (mocked Ollama)
# ---------------------------------------------------------------------------

def test_generate_pair_parses_clean_json():
    raw_json = json.dumps(FAKE_PAIR)
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": raw_json}

    with patch("grow_ai.expand.httpx.post", return_value=mock_response):
        result = _generate_pair("some insight", "{}", ["atomic_habits"])

    assert "instruction" in result
    assert "output" in result
    assert result["input"] == ""


def test_generate_pair_strips_markdown_fences():
    raw_json = f"```json\n{json.dumps(FAKE_PAIR)}\n```"
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": raw_json}

    with patch("grow_ai.expand.httpx.post", return_value=mock_response):
        result = _generate_pair("some insight", "{}", [])

    assert result["instruction"] == FAKE_PAIR["instruction"]


def test_generate_pair_forces_empty_input():
    pair_with_input = {**FAKE_PAIR, "input": "some unexpected input"}
    raw_json = json.dumps(pair_with_input)
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": raw_json}

    with patch("grow_ai.expand.httpx.post", return_value=mock_response):
        result = _generate_pair("insight", "{}", [])

    assert result["input"] == ""


# ---------------------------------------------------------------------------
# _save_lora_pair
# ---------------------------------------------------------------------------

def test_save_lora_pair_writes_to_db(db):
    insight_id = insert_insight(db, **SAMPLE_INSIGHT)
    _save_lora_pair(db, insight_id, FAKE_PAIR)

    row = db.execute(
        "SELECT lora_pair FROM insights WHERE id = ?", (insight_id,)
    ).fetchone()
    stored = json.loads(row["lora_pair"])
    assert stored["instruction"] == FAKE_PAIR["instruction"]


# ---------------------------------------------------------------------------
# run_expansion
# ---------------------------------------------------------------------------

def test_run_expansion_processes_all_unexpanded(db, mock_generate):
    insert_insight(db, **SAMPLE_INSIGHT)
    insert_insight(db, **{**SAMPLE_INSIGHT, "compressed": "second insight here"})

    stats = run_expansion(db)

    assert stats["processed"] == 2
    assert stats["failed"] == 0
    assert stats["total_unexpanded"] == 2

    # Both should now have lora_pair set
    remaining = get_unexpanded_insights(db)
    assert len(remaining) == 0


def test_run_expansion_skips_already_expanded(db, mock_generate):
    insight_id = insert_insight(db, **SAMPLE_INSIGHT)
    # Pre-expand one manually
    db.execute(
        "UPDATE insights SET lora_pair = ? WHERE id = ?",
        (json.dumps(FAKE_PAIR), insight_id),
    )
    db.commit()

    stats = run_expansion(db)

    assert stats["total_unexpanded"] == 0
    assert stats["processed"] == 0


def test_run_expansion_returns_zero_when_nothing_to_expand(db):
    stats = run_expansion(db)
    assert stats["processed"] == 0
    assert stats["total_unexpanded"] == 0


def test_run_expansion_dry_run_does_not_write(db, mock_generate):
    insert_insight(db, **SAMPLE_INSIGHT)

    stats = run_expansion(db, dry_run=True)

    assert stats["processed"] == 1
    remaining = get_unexpanded_insights(db)
    assert len(remaining) == 1  # still unexpanded


def test_run_expansion_counts_failures(db, mocker):
    insert_insight(db, **SAMPLE_INSIGHT)
    mocker.patch("grow_ai.expand._generate_pair", side_effect=Exception("Ollama timeout"))

    stats = run_expansion(db)

    assert stats["failed"] == 1
    assert stats["processed"] == 0


# ---------------------------------------------------------------------------
# export_jsonl
# ---------------------------------------------------------------------------

def test_export_jsonl_writes_expanded_only(db, mock_generate):
    # Insert two insights and expand both
    insert_insight(db, **SAMPLE_INSIGHT)
    insert_insight(db, **{**SAMPLE_INSIGHT, "compressed": "second insight"})
    run_expansion(db)

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="r") as f:
        out_path = Path(f.name)

    count = export_jsonl(db, out_path)
    lines = out_path.read_text().strip().split("\n")

    assert count == 2
    assert len(lines) == 2
    for line in lines:
        pair = json.loads(line)
        assert "instruction" in pair
        assert "output" in pair


def test_export_jsonl_skips_unexpanded(db, mock_generate):
    # Insert one expanded, one not
    id1 = insert_insight(db, **SAMPLE_INSIGHT)
    insert_insight(db, **{**SAMPLE_INSIGHT, "compressed": "unexpanded insight"})

    # Manually expand only the first
    _save_lora_pair(db, id1, FAKE_PAIR)

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="r") as f:
        out_path = Path(f.name)

    count = export_jsonl(db, out_path)
    assert count == 1


def test_export_jsonl_creates_parent_dirs(db, mock_generate, tmp_path):
    insert_insight(db, **SAMPLE_INSIGHT)
    run_expansion(db)

    deep_path = tmp_path / "a" / "b" / "c" / "training.jsonl"
    export_jsonl(db, deep_path)
    assert deep_path.exists()
