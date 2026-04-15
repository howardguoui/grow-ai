from pathlib import Path
from grow_ai.config import Config


def test_default_db_path_is_under_home():
    cfg = Config()
    assert cfg.db_path.parent == Path.home() / ".grow-ai"


def test_quality_threshold_default():
    cfg = Config()
    assert cfg.quality_threshold == 15


def test_dedup_discard_threshold():
    cfg = Config()
    assert cfg.dedup_discard == 0.95


def test_dedup_merge_threshold():
    cfg = Config()
    assert cfg.dedup_merge == 0.80
