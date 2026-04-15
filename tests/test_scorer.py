import pytest
from datetime import datetime, timedelta
from grow_ai.scorer import score, detect_frameworks, apply_temporal_decay


def test_edit_tool_scores_higher_than_read():
    edit_event = {"tool": "Edit", "result": "Added 20 lines to auth module"}
    read_event = {"tool": "Read", "result": "Opened file"}
    assert score(edit_event, "edit insight", []) > score(read_event, "read insight", [])


def test_error_recovery_adds_score():
    event_recovery = {"tool": "Bash", "result": "tests pass", "error_recovery": True}
    event_no_recovery = {"tool": "Bash", "result": "tests pass", "error_recovery": False}
    assert score(event_recovery, "fix insight", []) > score(event_no_recovery, "fix insight", [])


def test_framework_keywords_boost_score():
    tags = detect_frameworks("feedback loop pattern system")
    event = {"tool": "Edit", "result": "Added feedback loop detection"}
    s = score(event, "feedback loop insight", tags)
    assert s > 10


def test_detect_thinking_framework():
    tags = detect_frameworks("system 1 bias shortcut intuition fast")
    assert "thinking_fast_and_slow" in tags


def test_detect_atomic_habits():
    tags = detect_frameworks("habit cue trigger streak reward loop")
    assert "atomic_habits" in tags


def test_detect_antifragile():
    tags = detect_frameworks("stress volatility resilience disorder growth")
    assert "antifragile" in tags


def test_temporal_decay_recent():
    recent = datetime.utcnow() - timedelta(days=3)
    decayed = apply_temporal_decay(100, recent)
    expected = 100 * (1 - 0.02 * (3 / 7))
    assert decayed == pytest.approx(expected, rel=0.01)


def test_temporal_decay_six_months_halved():
    old = datetime.utcnow() - timedelta(weeks=26)
    decayed = apply_temporal_decay(100, old)
    assert decayed < 55
