"""Shared FastAPI dependencies."""
import sqlite3
from grow_ai.config import cfg
from grow_ai.db import get_connection, init_db


def get_conn():
    """Per-request SQLite connection. Closed after response."""
    conn = get_connection(cfg.db_path, check_same_thread=False)
    init_db(conn)
    try:
        yield conn
    finally:
        conn.close()
