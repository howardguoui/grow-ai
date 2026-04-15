import sqlite3
import pytest
import sqlite_vec
from grow_ai.db import init_db


@pytest.fixture
def db():
    """In-memory SQLite DB with sqlite-vec loaded and schema initialized."""
    conn = sqlite3.connect(":memory:")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    init_db(conn)
    yield conn
    conn.close()


@pytest.fixture
def mock_embed(mocker):
    """Returns a deterministic 768-dim zero vector for any input."""
    return mocker.patch(
        "grow_ai.embed.embed",
        return_value=[0.0] * 768,
    )


@pytest.fixture
def mock_ollama_compress(mocker):
    """Returns a fixed 20-word string for LLM fallback compression."""
    return mocker.patch(
        "grow_ai.compress._llm_compress",
        return_value="Fixed LLM compression output for testing purposes only here.",
    )
