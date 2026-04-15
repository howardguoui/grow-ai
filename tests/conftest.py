"""Test configuration and fixtures."""

import sqlite3
import pytest
import sqlite_vec


@pytest.fixture
def test_db_path(tmp_path):
    """Create a temporary test database path."""
    return str(tmp_path / "test.db")


@pytest.fixture
def sqlite_conn():
    """In-memory SQLite DB with sqlite-vec loaded."""
    conn = sqlite3.connect(":memory:")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    yield conn
    conn.close()
