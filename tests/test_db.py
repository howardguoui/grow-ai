import json
from grow_ai.db import init_db, insert_insight, get_all_insights, update_reinforcement, get_unexpanded_insights


def test_schema_creates_insights_table(db):
    cur = db.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cur.fetchall()}
    assert "insights" in tables
    assert "fine_tune_batches" in tables
    assert "growth_log" in tables


def test_insert_and_retrieve_insight(db):
    iid = insert_insight(
        db,
        compressed="Edit auth.py: fixed JWT expiry bug — token now validates correctly.",
        full_context="[full tool event JSON here]",
        framework_tags=["antifragile", "thinking_fast_and_slow"],
        quality_score=35,
        vector=[0.1] * 768,
        error_recovery=True,
    )
    assert iid > 0
    rows = get_all_insights(db)
    assert len(rows) == 1
    assert rows[0]["compressed"].startswith("Edit auth.py")
    assert rows[0]["error_recovery"] == 1


def test_update_reinforcement(db):
    iid = insert_insight(db, "insight one", "{}", [], 20, [0.0] * 768, False)
    update_reinforcement(db, iid, score_delta=3)
    rows = get_all_insights(db)
    assert rows[0]["reinforcement_count"] == 1
    assert rows[0]["quality_score"] == 23


def test_get_unexpanded_insights_returns_only_null_lora_pair(db):
    insert_insight(db, "insight a", "{}", [], 20, [0.0] * 768, False)
    insert_insight(db, "insight b", "{}", [], 25, [0.0] * 768, False)
    db.execute("UPDATE insights SET lora_pair = ? WHERE id = 1", (json.dumps({"instruction": "x"}),))
    db.commit()
    unexpanded = get_unexpanded_insights(db)
    assert len(unexpanded) == 1
    assert unexpanded[0]["compressed"] == "insight b"
