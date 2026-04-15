from grow_ai.dedup import check, _decide
from grow_ai.db import insert_insight


def test_store_when_db_empty(db, mock_embed):
    decision, existing_id = check(db, "new insight text", [0.1] * 768)
    assert decision == "store"
    assert existing_id is None


def test_discard_threshold_logic():
    assert _decide(0.97) == "discard"
    assert _decide(0.85) == "merge"
    assert _decide(0.70) == "store"
    assert _decide(0.95) == "discard"   # boundary: >= 0.95 is discard
    assert _decide(0.80) == "merge"     # boundary: >= 0.80 is merge
    assert _decide(0.799) == "store"    # just below merge threshold


def test_check_returns_tuple(db, mock_embed):
    decision, existing_id = check(db, "some insight", [0.0] * 768)
    assert isinstance(decision, str)
    assert decision in ("store", "merge", "discard")
    assert existing_id is None or isinstance(existing_id, int)
