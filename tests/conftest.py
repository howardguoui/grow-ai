"""Test configuration and fixtures."""

import pytest


@pytest.fixture
def test_db_path(tmp_path):
    """Create a temporary test database path."""
    return str(tmp_path / "test.db")
