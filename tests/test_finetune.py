"""
Tests for grow_ai/finetune.py — LoRA fine-tuning pipeline.
All GPU/Unsloth/Ollama calls are mocked. No CUDA required.
"""
import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import tempfile

import pytest

from grow_ai.finetune import (
    TrainConfig,
    build_dataset,
    get_queued_batch,
    get_all_lora_pairs,
    mark_batch_running,
    mark_batch_done,
    mark_batch_failed,
    next_model_version,
    run_finetune,
    run_benchmark,
    _write_modelfile,
    register_with_ollama,
    _format_pair,
)
from grow_ai.db import insert_insight


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_INSIGHT = {
    "compressed": "Edit src/auth.py: fixed JWT expiry — token validates correctly now",
    "full_context": '{"tool": "Edit", "result": "JWT fixed"}',
    "framework_tags": ["antifragile", "thinking_in_systems"],
    "quality_score": 35,
    "vector": [0.1] * 768,
    "error_recovery": True,
}

FAKE_PAIR = {
    "instruction": "How do I debug a JWT expiry issue?",
    "input": "",
    "output": "Apply Systems Thinking: trace the token lifecycle as a feedback loop...",
}


def _insert_with_lora_pair(db: sqlite3.Connection, pair: dict = None, **kwargs) -> int:
    """Insert an insight and immediately set its lora_pair."""
    insight_kwargs = {**SAMPLE_INSIGHT, **kwargs}
    iid = insert_insight(db, **insight_kwargs)
    if pair is not None:
        db.execute(
            "UPDATE insights SET lora_pair = ? WHERE id = ?",
            (json.dumps(pair), iid),
        )
        db.commit()
    return iid


def _queue_batch(db: sqlite3.Connection, insight_ids: list[int]) -> int:
    """Insert a queued fine_tune_batch and return its ID."""
    cur = db.execute(
        "INSERT INTO fine_tune_batches (insight_ids, status) VALUES (?, ?)",
        (json.dumps(insight_ids), "queued"),
    )
    db.commit()
    return cur.lastrowid


# ---------------------------------------------------------------------------
# TrainConfig
# ---------------------------------------------------------------------------

def test_train_config_defaults():
    cfg = TrainConfig()
    assert cfg.lora_r == 16
    assert cfg.replay_fraction == 0.5
    assert cfg.num_train_epochs == 1
    assert "q_proj" in cfg.target_modules


def test_train_config_adapters_dir_under_home():
    cfg = TrainConfig()
    assert ".grow-ai" in str(cfg.adapters_dir)


# ---------------------------------------------------------------------------
# get_all_lora_pairs
# ---------------------------------------------------------------------------

def test_get_all_lora_pairs_returns_only_expanded(db):
    _insert_with_lora_pair(db, FAKE_PAIR)
    _insert_with_lora_pair(db, None)  # unexpanded

    pairs = get_all_lora_pairs(db)
    assert len(pairs) == 1
    assert pairs[0]["instruction"] == FAKE_PAIR["instruction"]
    assert "_insight_id" in pairs[0]


def test_get_all_lora_pairs_includes_insight_id(db):
    iid = _insert_with_lora_pair(db, FAKE_PAIR)
    pairs = get_all_lora_pairs(db)
    assert pairs[0]["_insight_id"] == iid


# ---------------------------------------------------------------------------
# build_dataset
# ---------------------------------------------------------------------------

def test_build_dataset_includes_all_new_pairs():
    new = [FAKE_PAIR.copy() for _ in range(10)]
    dataset = build_dataset(new, [], replay_fraction=0.5)
    assert len(dataset) == 10
    assert all("text" in d for d in dataset)


def test_build_dataset_mixes_historical():
    new = [FAKE_PAIR.copy() for _ in range(10)]
    historical = [{**FAKE_PAIR, "instruction": f"historical {i}"} for i in range(20)]
    dataset = build_dataset(new, historical, replay_fraction=0.5)
    # Should include new (10) + ~10 historical
    assert len(dataset) >= 10
    assert len(dataset) <= 30


def test_build_dataset_no_historical_returns_new_only():
    new = [FAKE_PAIR.copy() for _ in range(5)]
    dataset = build_dataset(new, [], replay_fraction=0.5)
    assert len(dataset) == 5


def test_build_dataset_formats_alpaca_prompt():
    new = [FAKE_PAIR.copy()]
    dataset = build_dataset(new, [])
    assert "### Instruction:" in dataset[0]["text"]
    assert "### Response:" in dataset[0]["text"]
    assert FAKE_PAIR["instruction"] in dataset[0]["text"]


def test_build_dataset_is_deterministic():
    new = [FAKE_PAIR.copy() for _ in range(5)]
    hist = [{**FAKE_PAIR, "instruction": f"h{i}"} for i in range(10)]
    d1 = build_dataset(new, hist, seed=42)
    d2 = build_dataset(new, hist, seed=42)
    assert [d["text"] for d in d1] == [d["text"] for d in d2]


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def test_get_queued_batch_returns_none_when_empty(db):
    assert get_queued_batch(db) is None


def test_get_queued_batch_returns_oldest(db):
    _queue_batch(db, [1])
    _queue_batch(db, [2])
    batch = get_queued_batch(db)
    assert batch is not None
    assert json.loads(batch["insight_ids"]) == [1]


def test_mark_batch_running(db):
    batch_id = _queue_batch(db, [1])
    mark_batch_running(db, batch_id)
    row = db.execute("SELECT status FROM fine_tune_batches WHERE id = ?", (batch_id,)).fetchone()
    assert row[0] == "running"


def test_mark_batch_done(db):
    batch_id = _queue_batch(db, [1])
    mark_batch_done(db, batch_id, "v1.0")
    row = db.execute(
        "SELECT status, model_version FROM fine_tune_batches WHERE id = ?", (batch_id,)
    ).fetchone()
    assert row[0] == "done"
    assert row[1] == "v1.0"


def test_mark_batch_failed(db):
    batch_id = _queue_batch(db, [1])
    mark_batch_failed(db, batch_id, "CUDA OOM")
    row = db.execute("SELECT status FROM fine_tune_batches WHERE id = ?", (batch_id,)).fetchone()
    assert row[0] == "failed"


def test_next_model_version_increments(db):
    assert next_model_version(db) == "v1.0"
    _queue_batch(db, [1])
    batch_id = _queue_batch(db, [1])
    mark_batch_done(db, batch_id, "v1.0")
    assert next_model_version(db) == "v2.0"


# ---------------------------------------------------------------------------
# run_finetune — dry run (no GPU needed)
# ---------------------------------------------------------------------------

def test_run_finetune_returns_no_queued_batch_when_empty(db):
    result = run_finetune(db)
    assert result["status"] == "no_queued_batch"


def test_run_finetune_dry_run_does_not_call_unsloth(db):
    iid = _insert_with_lora_pair(db, FAKE_PAIR)
    _queue_batch(db, [iid])

    with patch("grow_ai.finetune.UNSLOTH_AVAILABLE", False):
        result = run_finetune(db, dry_run=True)

    assert result["status"] == "dry_run"
    assert "model_version" in result


def test_run_finetune_dry_run_does_not_update_batch_status(db):
    iid = _insert_with_lora_pair(db, FAKE_PAIR)
    batch_id = _queue_batch(db, [iid])

    with patch("grow_ai.finetune.UNSLOTH_AVAILABLE", False):
        run_finetune(db, dry_run=True)

    row = db.execute(
        "SELECT status FROM fine_tune_batches WHERE id = ?", (batch_id,)
    ).fetchone()
    # Dry run should not mark as done
    assert row[0] == "queued"


def test_run_finetune_force_creates_batch_when_none_queued(db):
    _insert_with_lora_pair(db, FAKE_PAIR)

    with patch("grow_ai.finetune.UNSLOTH_AVAILABLE", False):
        result = run_finetune(db, dry_run=True, force=True)

    assert result["status"] == "dry_run"


def test_run_finetune_force_returns_no_insights_when_db_empty(db):
    result = run_finetune(db, force=True)
    assert result["status"] == "no_insights"


def test_run_finetune_fails_gracefully_without_unsloth(db):
    iid = _insert_with_lora_pair(db, FAKE_PAIR)
    _queue_batch(db, [iid])

    with patch("grow_ai.finetune.UNSLOTH_AVAILABLE", False):
        # non-dry-run without Unsloth should fail gracefully
        result = run_finetune(db, dry_run=False)

    assert result["status"] == "failed"
    assert "Unsloth" in result["error"]


# ---------------------------------------------------------------------------
# run_benchmark
# ---------------------------------------------------------------------------

def test_run_benchmark_stores_5_questions(db):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": "Framework-grounded answer here."}

    with patch("grow_ai.finetune.httpx.post", return_value=mock_resp):
        results = run_benchmark(db, "v1.0")

    assert len(results) == 5
    rows = db.execute("SELECT COUNT(*) FROM growth_log").fetchone()[0]
    assert rows == 5


def test_run_benchmark_handles_ollama_failure(db):
    with patch("grow_ai.finetune.httpx.post", side_effect=Exception("connection refused")):
        results = run_benchmark(db, "v1.0")

    assert len(results) == 5
    assert all("benchmark error" in r["response"] for r in results)


# ---------------------------------------------------------------------------
# Ollama registration
# ---------------------------------------------------------------------------

def test_write_modelfile_creates_file(tmp_path):
    # Create a fake GGUF file
    gguf = tmp_path / "model.gguf"
    gguf.write_text("fake")

    modelfile = _write_modelfile(tmp_path, "unsloth/Qwen2.5-3B")
    assert modelfile.exists()
    content = modelfile.read_text()
    assert "FROM" in content
    assert "grow-ai-personal" in content.lower() or "System" in content


def test_write_modelfile_raises_if_no_gguf(tmp_path):
    with pytest.raises(FileNotFoundError):
        _write_modelfile(tmp_path, "unsloth/Qwen2.5-3B")


def test_register_with_ollama_returns_true_on_success(tmp_path):
    modelfile = tmp_path / "Modelfile"
    modelfile.write_text("FROM fake.gguf\n")

    with patch("grow_ai.finetune.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        result = register_with_ollama(tmp_path, "grow-ai-personal")

    assert result is True
    mock_run.assert_called_once()


def test_register_with_ollama_returns_false_on_failure(tmp_path):
    modelfile = tmp_path / "Modelfile"
    modelfile.write_text("FROM fake.gguf\n")

    with patch("grow_ai.finetune.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1)
        result = register_with_ollama(tmp_path, "grow-ai-personal")

    assert result is False


def test_register_with_ollama_returns_false_when_no_modelfile(tmp_path):
    result = register_with_ollama(tmp_path, "grow-ai-personal")
    assert result is False
