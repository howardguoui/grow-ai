"""Test basic imports."""

import pytest


def test_grow_ai_imports():
    """Test that grow_ai module imports correctly."""
    import grow_ai
    assert grow_ai.__version__ == "0.1.0"


def test_sqlite_vec_available():
    """Test that sqlite-vec is installed and imports."""
    import sqlite_vec
    assert sqlite_vec is not None
