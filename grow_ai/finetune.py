"""
grow_ai/finetune.py — Unsloth LoRA fine-tuning pipeline.

Triggered automatically when fine_tune_batches has a 'queued' entry.
Runs on local GPU (RTX 5070 Ti, 16GB VRAM) via Unsloth + LoRA.

Full pipeline:
  1. Check for queued batch in fine_tune_batches
  2. Build dataset: 50% historical replay + 50% new insights (anti-forgetting)
  3. Fine-tune base model with Unsloth + SFTTrainer (LoRA)
  4. Save adapter to ~/.grow-ai/adapters/<version>/
  5. Convert to GGUF + create Ollama modelfile + register model
  6. Run growth log benchmark (5 standard questions)
  7. Mark batch status 'done', record model_version

CLI:
    python -m grow_ai.finetune              # run if queued batch exists
    python -m grow_ai.finetune --dry-run    # show what would train, skip GPU
    python -m grow_ai.finetune --force      # force train even if no queued batch
"""
from __future__ import annotations

import json
import random
import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from grow_ai.config import cfg
from grow_ai.db import get_connection, init_db

# ---------------------------------------------------------------------------
# Unsloth — optional import (not available in test environments without CUDA)
# ---------------------------------------------------------------------------
try:
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Model
    base_model: str = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
    max_seq_length: int = 2048
    load_in_4bit: bool = True

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Training
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    num_train_epochs: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    seed: int = 42

    # Replay buffer: fraction of dataset that is historical (anti-forgetting)
    replay_fraction: float = 0.5

    # Paths
    adapters_dir: Path = field(default_factory=lambda: Path.home() / ".grow-ai" / "adapters")
    ollama_model_name: str = "grow-ai-personal"


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_queued_batch(conn: sqlite3.Connection) -> sqlite3.Row | None:
    """Return the oldest queued fine_tune_batch, or None."""
    return conn.execute(
        "SELECT * FROM fine_tune_batches WHERE status = 'queued' ORDER BY triggered_at LIMIT 1"
    ).fetchone()


def mark_batch_running(conn: sqlite3.Connection, batch_id: int) -> None:
    conn.execute(
        "UPDATE fine_tune_batches SET status = 'running' WHERE id = ?", (batch_id,)
    )
    conn.commit()


def mark_batch_done(
    conn: sqlite3.Connection,
    batch_id: int,
    model_version: str,
) -> None:
    conn.execute(
        """UPDATE fine_tune_batches
           SET status = 'done', completed_at = CURRENT_TIMESTAMP, model_version = ?
           WHERE id = ?""",
        (model_version, batch_id),
    )
    conn.commit()


def mark_batch_failed(conn: sqlite3.Connection, batch_id: int, reason: str) -> None:
    conn.execute(
        "UPDATE fine_tune_batches SET status = 'failed', model_version = ? WHERE id = ?",
        (reason[:200], batch_id),
    )
    conn.commit()


def get_all_lora_pairs(conn: sqlite3.Connection) -> list[dict]:
    """Return all expanded insights as dicts, including their insight_id."""
    rows = conn.execute(
        "SELECT id, lora_pair FROM insights WHERE lora_pair IS NOT NULL"
    ).fetchall()
    pairs = []
    for row in rows:
        try:
            pair = json.loads(row["lora_pair"])
            pair["_insight_id"] = row["id"]
            pairs.append(pair)
        except (json.JSONDecodeError, TypeError):
            continue
    return pairs


# ---------------------------------------------------------------------------
# Dataset construction with replay buffer
# ---------------------------------------------------------------------------

_ALPACA_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n{output}"
)


def _format_pair(pair: dict) -> str:
    return _ALPACA_PROMPT.format(
        instruction=pair.get("instruction", ""),
        output=pair.get("output", ""),
    )


def build_dataset(
    new_pairs: list[dict],
    historical_pairs: list[dict],
    replay_fraction: float = 0.5,
    seed: int = 42,
) -> list[dict]:
    """
    Combine new insights with historical replay to prevent catastrophic forgetting.
    Result: replay_fraction % historical, (1 - replay_fraction) % new.
    Minimum: all new pairs always included.
    """
    rng = random.Random(seed)

    if not historical_pairs:
        combined = new_pairs
    else:
        n_new = len(new_pairs)
        n_historical = max(1, int(n_new * (replay_fraction / (1 - replay_fraction + 1e-9))))
        n_historical = min(n_historical, len(historical_pairs))
        sampled_historical = rng.sample(historical_pairs, n_historical)
        combined = new_pairs + sampled_historical
        rng.shuffle(combined)

    return [{"text": _format_pair(p)} for p in combined]


# ---------------------------------------------------------------------------
# Version naming
# ---------------------------------------------------------------------------

def next_model_version(conn: sqlite3.Connection) -> str:
    """Increment model version based on completed batches."""
    done_count = conn.execute(
        "SELECT COUNT(*) FROM fine_tune_batches WHERE status = 'done'"
    ).fetchone()[0]
    return f"v{done_count + 1}.0"


# ---------------------------------------------------------------------------
# Growth log benchmark
# ---------------------------------------------------------------------------

_BENCHMARK_QUESTIONS = [
    "How would you design a feedback loop for a React state manager?",
    "You have 3 hours to debug an unknown production error. Walk me through your approach.",
    "How do you decide when to stop exploring solutions and commit to one?",
    "Design a personal learning system for mastering a new programming language in 30 days.",
    "A habit you built 6 months ago is breaking down. What's your diagnosis and fix?",
]


def run_benchmark(conn: sqlite3.Connection, model_version: str) -> list[dict]:
    """
    Ask the Ollama model the 5 standard benchmark questions.
    Stores responses in growth_log. Returns list of {question_id, response}.
    """
    results = []
    total_insights = conn.execute("SELECT COUNT(*) FROM insights").fetchone()[0]

    for i, question in enumerate(_BENCHMARK_QUESTIONS, start=1):
        try:
            resp = httpx.post(
                f"{cfg.ollama_base_url}/api/generate",
                json={
                    "model": cfg.ollama_model_name if hasattr(cfg, "ollama_model_name")
                             else "grow-ai-personal",
                    "prompt": question,
                    "stream": False,
                },
                timeout=60.0,
            )
            resp.raise_for_status()
            answer = resp.json()["response"].strip()
        except Exception as exc:
            answer = f"[benchmark error: {exc}]"

        conn.execute(
            "INSERT INTO growth_log (insight_count, model_version, question_id, response) VALUES (?, ?, ?, ?)",
            (total_insights, model_version, i, answer),
        )
        results.append({"question_id": i, "response": answer[:100]})

    conn.commit()
    return results


# ---------------------------------------------------------------------------
# Ollama registration
# ---------------------------------------------------------------------------

def _write_modelfile(adapter_dir: Path, base_model_tag: str) -> Path:
    """Write an Ollama Modelfile that loads the GGUF adapter."""
    gguf_files = list(adapter_dir.glob("*.gguf"))
    if not gguf_files:
        raise FileNotFoundError(f"No .gguf file found in {adapter_dir}")

    modelfile_path = adapter_dir / "Modelfile"
    modelfile_path.write_text(
        f"FROM {gguf_files[0]}\n"
        f"PARAMETER temperature 0.7\n"
        f"PARAMETER top_p 0.9\n"
        f'SYSTEM "You are a personal AI assistant that has grown from your owner\'s Claude Code sessions. '
        f'You reason using 9 strategic frameworks: Thinking Fast and Slow, Clear Thinking, '
        f'Algorithms to Live By, Moonwalking with Einstein, The Memory Book, Make It Stick, '
        f'Ultralearning, Atomic Habits, Thinking in Systems, Antifragile."\n'
    )
    return modelfile_path


def register_with_ollama(adapter_dir: Path, model_name: str) -> bool:
    """Run `ollama create <model_name> -f Modelfile`. Returns True on success."""
    modelfile = adapter_dir / "Modelfile"
    if not modelfile.exists():
        return False
    result = subprocess.run(
        ["ollama", "create", model_name, "-f", str(modelfile)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def train(
    conn: sqlite3.Connection,
    batch_id: int,
    batch_insight_ids: list[int],
    cfg_train: TrainConfig,
    dry_run: bool = False,
) -> str:
    """
    Run LoRA fine-tuning for one batch. Returns model_version string.
    Raises on unrecoverable error.
    """
    if not UNSLOTH_AVAILABLE and not dry_run:
        raise RuntimeError(
            "Unsloth not installed. Run: pip install unsloth trl transformers datasets\n"
            "Requires CUDA (RTX 5070 Ti or similar)."
        )

    model_version = next_model_version(conn)
    adapter_dir = cfg_train.adapters_dir / model_version
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # ── Build dataset ────────────────────────────────────────────────────────
    all_pairs = get_all_lora_pairs(conn)
    batch_id_set = set(batch_insight_ids)
    new_pairs = [p for p in all_pairs if p.get("_insight_id") in batch_id_set]
    historical_pairs = [p for p in all_pairs if p.get("_insight_id") not in batch_id_set]

    dataset_dicts = build_dataset(new_pairs, historical_pairs, cfg_train.replay_fraction, cfg_train.seed)
    print(f"[finetune] Dataset: {len(dataset_dicts)} samples ({model_version})")

    if dry_run:
        print(f"[finetune] DRY RUN — would train on {len(dataset_dicts)} samples → {adapter_dir}")
        return model_version

    # ── Load model ───────────────────────────────────────────────────────────
    mark_batch_running(conn, batch_id)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg_train.base_model,
        max_seq_length=cfg_train.max_seq_length,
        dtype=None,
        load_in_4bit=cfg_train.load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg_train.lora_r,
        target_modules=cfg_train.target_modules,
        lora_alpha=cfg_train.lora_alpha,
        lora_dropout=cfg_train.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg_train.seed,
    )

    # ── Prepare dataset ──────────────────────────────────────────────────────
    dataset = Dataset.from_list(dataset_dicts)

    # ── Train ────────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=cfg_train.max_seq_length,
        dataset_num_proc=2,
        args=TrainingArguments(
            per_device_train_batch_size=cfg_train.batch_size,
            gradient_accumulation_steps=cfg_train.gradient_accumulation_steps,
            warmup_steps=cfg_train.warmup_steps,
            num_train_epochs=cfg_train.num_train_epochs,
            learning_rate=cfg_train.learning_rate,
            weight_decay=cfg_train.weight_decay,
            fp16=True,
            lr_scheduler_type=cfg_train.lr_scheduler_type,
            seed=cfg_train.seed,
            output_dir=str(adapter_dir / "checkpoints"),
            report_to="none",
        ),
    )
    trainer.train()

    # ── Save adapter ─────────────────────────────────────────────────────────
    model.save_pretrained(str(adapter_dir / "lora_adapter"))
    tokenizer.save_pretrained(str(adapter_dir / "lora_adapter"))
    print(f"[finetune] Adapter saved → {adapter_dir / 'lora_adapter'}")

    # ── Convert to GGUF ──────────────────────────────────────────────────────
    model.save_pretrained_gguf(
        str(adapter_dir),
        tokenizer,
        quantization_method="q4_k_m",
    )
    print(f"[finetune] GGUF saved → {adapter_dir}")

    # ── Register with Ollama ─────────────────────────────────────────────────
    _write_modelfile(adapter_dir, cfg_train.base_model)
    ok = register_with_ollama(adapter_dir, cfg_train.ollama_model_name)
    if ok:
        print(f"[finetune] Registered as ollama model '{cfg_train.ollama_model_name}'")
    else:
        print(f"[finetune] WARNING: ollama create failed — adapter saved but not registered")

    return model_version


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_finetune(
    conn: sqlite3.Connection,
    cfg_train: TrainConfig | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> dict:
    """
    Check for queued batch → train → benchmark → update DB.
    Returns result dict.
    """
    if cfg_train is None:
        cfg_train = TrainConfig()

    batch = get_queued_batch(conn)

    if batch is None and not force:
        return {"status": "no_queued_batch"}

    if force and batch is None:
        # Create an ad-hoc batch with all insights
        insight_ids = [r[0] for r in conn.execute("SELECT id FROM insights").fetchall()]
        if not insight_ids:
            return {"status": "no_insights"}
        conn.execute(
            "INSERT INTO fine_tune_batches (insight_ids, status) VALUES (?, ?)",
            (json.dumps(insight_ids), "queued"),
        )
        conn.commit()
        batch = get_queued_batch(conn)

    batch_id = batch["id"]
    batch_insight_ids = json.loads(batch["insight_ids"] or "[]")

    print(f"[finetune] Processing batch {batch_id} ({len(batch_insight_ids)} insights)")

    try:
        model_version = train(conn, batch_id, batch_insight_ids, cfg_train, dry_run=dry_run)
    except Exception as exc:
        mark_batch_failed(conn, batch_id, str(exc))
        return {"status": "failed", "error": str(exc)}

    if not dry_run:
        mark_batch_done(conn, batch_id, model_version)

        # Run benchmark to prove improvement
        benchmark_results = run_benchmark(conn, model_version)
        print(f"[finetune] Benchmark complete: {len(benchmark_results)} questions logged")

    return {
        "status": "done" if not dry_run else "dry_run",
        "batch_id": batch_id,
        "model_version": model_version,
        "insight_count": len(batch_insight_ids),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    dry_run = "--dry-run" in sys.argv
    force = "--force" in sys.argv

    conn = get_connection(cfg.db_path)
    init_db(conn)

    result = run_finetune(conn, dry_run=dry_run, force=force)
    print(f"[grow-ai] Fine-tune result: {result}")

    conn.close()


if __name__ == "__main__":
    main()
