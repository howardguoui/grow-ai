"""System routes: /health, /stats, /finetune (manual trigger)."""
import json
from fastapi import APIRouter, Depends
import httpx

from api.deps import get_conn
from api.schemas import HealthOut, StatsOut, FinetuneOut

router = APIRouter(prefix="/api", tags=["system"])

_BENCHMARK_QUESTIONS = [
    "How would you design a feedback loop for a React state manager?",
    "You have 3 hours to debug an unknown production error. Walk me through your approach.",
    "How do you decide when to stop exploring solutions and commit to one?",
    "Design a personal learning system for mastering a new programming language in 30 days.",
    "A habit you built 6 months ago is breaking down. What's your diagnosis and fix?",
]


def _ollama_alive(base_url: str) -> bool:
    try:
        httpx.get(f"{base_url}/api/tags", timeout=3.0)
        return True
    except Exception:
        return False


@router.get("/health", response_model=HealthOut)
def health(conn=Depends(get_conn)):
    from grow_ai.config import cfg

    total = conn.execute("SELECT COUNT(*) FROM insights").fetchone()[0]
    last_version = conn.execute(
        "SELECT model_version FROM fine_tune_batches WHERE status='done' ORDER BY completed_at DESC LIMIT 1"
    ).fetchone()

    return HealthOut(
        ollama=_ollama_alive(cfg.ollama_base_url),
        db=True,
        last_model_version=last_version[0] if last_version else None,
        total_insights=total,
    )


@router.get("/stats", response_model=StatsOut)
def stats(conn=Depends(get_conn)):
    total = conn.execute("SELECT COUNT(*) FROM insights").fetchone()[0]
    avg = conn.execute("SELECT AVG(quality_score) FROM insights").fetchone()[0] or 0.0
    high_q = conn.execute("SELECT COUNT(*) FROM insights WHERE quality_score >= 50").fetchone()[0]
    err_rec = conn.execute("SELECT COUNT(*) FROM insights WHERE error_recovery=1").fetchone()[0]
    reinforced = conn.execute("SELECT COUNT(*) FROM insights WHERE reinforcement_count > 0").fetchone()[0]
    pending_exp = conn.execute("SELECT COUNT(*) FROM insights WHERE lora_pair IS NULL").fetchone()[0]
    queued = conn.execute("SELECT COUNT(*) FROM fine_tune_batches WHERE status='queued'").fetchone()[0]
    done = conn.execute("SELECT COUNT(*) FROM fine_tune_batches WHERE status='done'").fetchone()[0]
    last_ver = conn.execute(
        "SELECT model_version FROM fine_tune_batches WHERE status='done' ORDER BY completed_at DESC LIMIT 1"
    ).fetchone()

    tag_rows = conn.execute("SELECT framework_tags FROM insights").fetchall()
    counts: dict[str, int] = {}
    for row in tag_rows:
        for tag in json.loads(row[0] or "[]"):
            counts[tag] = counts.get(tag, 0) + 1
    top = sorted(counts.items(), key=lambda x: -x[1])[:5]

    return StatsOut(
        total_insights=total,
        avg_quality_score=round(float(avg), 1),
        high_quality_count=high_q,
        error_recovery_count=err_rec,
        reinforced_count=reinforced,
        top_frameworks=[{"framework": f, "count": c} for f, c in top],
        pending_expansion=pending_exp,
        queued_batches=queued,
        completed_batches=done,
        last_batch_version=last_ver[0] if last_ver else None,
    )


@router.post("/finetune", response_model=FinetuneOut)
def trigger_finetune(conn=Depends(get_conn)):
    """Manually queue a fine-tune batch and kick off training."""
    from grow_ai.finetune import run_finetune

    result = run_finetune(conn, force=True)
    status = result.get("status", "unknown")

    messages = {
        "done": "Fine-tune complete.",
        "dry_run": "Dry-run complete (no GPU).",
        "no_insights": "No insights to train on yet.",
        "failed": f"Training failed: {result.get('error', 'unknown error')}",
    }
    return FinetuneOut(status=status, message=messages.get(status, status))
