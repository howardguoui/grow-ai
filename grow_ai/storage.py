"""Storage backend for conversations.

Stores scrubbed, compressed, embedded, scored conversations.
"""

import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Optional
import json


@dataclass
class StoredConversation:
    """A stored conversation record."""

    id: int
    original_text: str
    scrubbed_text: str
    compressed_text: str
    embedding: list[float]
    primary_framework: str
    score: float
    created_at: datetime


class ConversationStore:
    """Store conversations in SQLite."""

    def __init__(self, db_path: str = "grow_ai.db"):
        """Initialize store.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_text TEXT NOT NULL,
                    scrubbed_text TEXT NOT NULL,
                    compressed_text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    primary_framework TEXT,
                    score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_created_at ON conversations(created_at)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_framework ON conversations(primary_framework)
                """
            )
            conn.commit()

    def store(
        self,
        original_text: str,
        scrubbed_text: str,
        compressed_text: str,
        embedding: list[float],
        primary_framework: Optional[str] = None,
        score: Optional[float] = None,
    ) -> int:
        """Store a conversation.

        Args:
            original_text: Original conversation text
            scrubbed_text: Scrubbed text (sensitive data removed)
            compressed_text: Compressed summary
            embedding: Vector embedding
            primary_framework: Primary activated framework
            score: Quality score (0-100)

        Returns:
            ID of stored conversation
        """
        embedding_bytes = json.dumps(embedding).encode("utf-8")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO conversations
                (original_text, scrubbed_text, compressed_text, embedding,
                 primary_framework, score)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (original_text, scrubbed_text, compressed_text, embedding_bytes,
                 primary_framework, score),
            )
            conn.commit()
            return cursor.lastrowid

    def get(self, conversation_id: int) -> Optional[StoredConversation]:
        """Retrieve a conversation.

        Args:
            conversation_id: ID of conversation

        Returns:
            StoredConversation or None
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT id, original_text, scrubbed_text, compressed_text,
                       embedding, primary_framework, score, created_at
                FROM conversations WHERE id = ?
                """,
                (conversation_id,),
            )
            row = cursor.fetchone()

        if not row:
            return None

        id_, orig, scrubbed, compressed, emb_bytes, fw, score, created_at = row
        embedding = json.loads(emb_bytes.decode("utf-8"))
        created = datetime.fromisoformat(created_at)

        return StoredConversation(
            id=id_,
            original_text=orig,
            scrubbed_text=scrubbed,
            compressed_text=compressed,
            embedding=embedding,
            primary_framework=fw,
            score=score,
            created_at=created,
        )

    def count(self) -> int:
        """Get total number of stored conversations.

        Returns:
            Count of conversations
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM conversations")
            return cursor.fetchone()[0]

    def list_recent(self, limit: int = 10) -> list[StoredConversation]:
        """Get recent conversations.

        Args:
            limit: Maximum number to return

        Returns:
            List of StoredConversation ordered by creation date (newest first)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT id, original_text, scrubbed_text, compressed_text,
                       embedding, primary_framework, score, created_at
                FROM conversations
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()

        results = []
        for id_, orig, scrubbed, compressed, emb_bytes, fw, score, created_at in rows:
            embedding = json.loads(emb_bytes.decode("utf-8"))
            created = datetime.fromisoformat(created_at)
            results.append(
                StoredConversation(
                    id=id_,
                    original_text=orig,
                    scrubbed_text=scrubbed,
                    compressed_text=compressed,
                    embedding=embedding,
                    primary_framework=fw,
                    score=score,
                    created_at=created,
                )
            )
        return results
