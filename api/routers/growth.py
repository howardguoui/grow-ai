"""Growth log and batch history routes."""
import json
from fastapi import APIRouter, Depends

from api.deps import get_conn
from api.schemas import GrowthSnapshot, BatchOut

router = APIRouter(prefix="/api", tags=["growth"])

_BENCHMARK_QUESTIONS = {
    1: "How would you design a feedback loop for a React state manager?",
    2: "You have 3 hours to debug an unknown production error. Walk me through your approach.",
    3: "How do you decide when to stop exploring solutions and commit to one?",
    4: "Design a personal learning system for mastering a new programming language in 30 days.",
    5: "A habit you built 6 months ago is breaking down. What's your diagnosis and fix?",
}


@router.get("/growth-log", response_model=list[GrowthSnapshot])
def growth_log(conn=Depends(get_conn)):
    """
    Return benchmark snapshots grouped by model_version + recorded_at session.
    Excludes daily-report entries (question_id=0).
    """
    rows = conn.execute(
        """SELECT model_version, insight_count, question_id, response, recorded_at
           FROM growth_log
           WHERE question_id > 0
           ORDER BY recorded_at ASC"""
    ).fetchall()

    # Group by (model_version, date(recorded_at)) to form snapshots
    snapshots: dict[str, dict] = {}
    for row in rows:
        key = f"{row['model_version']}_{row['recorded_at'][:10]}"
        if key not in snapshots:
            snapshots[key] = {
                "model_version": row["model_version"],
                "insight_count": row["insight_count"],
                "entries": [],
                "recorded_at": row["recorded_at"],
            }
        qid = row["question_id"]
        snapshots[key]["entries"].append({
            "question_id": qid,
            "question": _BENCHMARK_QUESTIONS.get(qid, f"Question {qid}"),
            "response": row["response"],
            "recorded_at": row["recorded_at"],
        })

    return [GrowthSnapshot(**v) for v in snapshots.values()]


@router.get("/batches", response_model=list[BatchOut])
def list_batches(conn=Depends(get_conn)):
    rows = conn.execute(
        "SELECT * FROM fine_tune_batches ORDER BY triggered_at DESC"
    ).fetchall()

    result = []
    for row in rows:
        ids = json.loads(row["insight_ids"] or "[]")
        result.append(BatchOut(
            id=row["id"],
            status=row["status"],
            insight_count=len(ids),
            triggered_at=row["triggered_at"],
            completed_at=row["completed_at"],
            model_version=row["model_version"],
        ))
    return result
