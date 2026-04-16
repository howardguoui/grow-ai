"""Insights routes: list, get, search."""
import json
import math
from fastapi import APIRouter, Depends, HTTPException, Query

from api.deps import get_conn
from api.schemas import InsightOut, InsightDetail, InsightPage, SearchResult

router = APIRouter(prefix="/api/insights", tags=["insights"])


def _row_to_out(row) -> dict:
    return {
        "id": row["id"],
        "compressed": row["compressed"],
        "framework_tags": json.loads(row["framework_tags"] or "[]"),
        "quality_score": row["quality_score"],
        "reinforcement_count": row["reinforcement_count"],
        "error_recovery": bool(row["error_recovery"]),
        "created_at": row["created_at"],
        "similarity": None,
    }


@router.get("", response_model=InsightPage)
def list_insights(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    framework: str | None = Query(None),
    conn=Depends(get_conn),
):
    offset = (page - 1) * limit

    if framework:
        where = "WHERE framework_tags LIKE ?"
        params_count = (f"%{framework}%",)
        params_fetch = (f"%{framework}%", limit, offset)
    else:
        where = ""
        params_count = ()
        params_fetch = (limit, offset)

    total = conn.execute(
        f"SELECT COUNT(*) FROM insights {where}", params_count
    ).fetchone()[0]

    rows = conn.execute(
        f"SELECT * FROM insights {where} ORDER BY quality_score DESC LIMIT ? OFFSET ?",
        params_fetch,
    ).fetchall()

    return InsightPage(
        items=[InsightOut(**_row_to_out(r)) for r in rows],
        total=total,
        page=page,
        pages=max(1, math.ceil(total / limit)),
    )


@router.get("/search", response_model=SearchResult)
def search_insights(
    q: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=50),
    conn=Depends(get_conn),
):
    from grow_ai.search import search

    results, mode = search(conn, q, limit)

    items = []
    for r in results:
        r["framework_tags"] = json.loads(r.get("framework_tags") or "[]")
        items.append(InsightOut(**r))

    return SearchResult(items=items, query=q, mode=mode, count=len(items))


@router.get("/{insight_id}", response_model=InsightDetail)
def get_insight(insight_id: int, conn=Depends(get_conn)):
    row = conn.execute(
        "SELECT * FROM insights WHERE id = ?", (insight_id,)
    ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Insight not found")

    data = _row_to_out(row)
    data["full_context"] = row["full_context"]
    lora_raw = row["lora_pair"]
    data["lora_pair"] = json.loads(lora_raw) if lora_raw else None

    return InsightDetail(**data)
