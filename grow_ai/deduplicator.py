"""Deduplication for conversations.

Detects and filters duplicate or near-duplicate conversations using embeddings.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DeduplicatorResult:
    """Result of deduplication check."""

    text: str
    embedding: list[float]
    is_duplicate: bool
    similar_to: Optional[str] = None
    similarity_score: float = 0.0


class Deduplicator:
    """Deduplicate conversations using embeddings."""

    def __init__(self, similarity_threshold: float = 0.85, embedder=None):
        """Initialize deduplicator.

        Args:
            similarity_threshold: Threshold for marking as duplicate (0-1)
            embedder: Embedder instance for generating embeddings
        """
        self.similarity_threshold = similarity_threshold
        self.embedder = embedder
        self.known_embeddings: list[tuple[str, list[float]]] = []

    def check_duplicate(
        self,
        text: str,
        embedding: Optional[list[float]] = None,
    ) -> DeduplicatorResult:
        """Check if text is duplicate of known conversations.

        Args:
            text: Text to check
            embedding: Pre-computed embedding (or None to compute)

        Returns:
            DeduplicatorResult with duplicate status
        """
        if embedding is None:
            if self.embedder is None:
                raise ValueError("Either provide embedding or set embedder")
            result = self.embedder.embed(text)
            embedding = result.embedding

        # Check against known embeddings
        is_duplicate = False
        similar_to = None
        max_similarity = 0.0

        for known_text, known_embedding in self.known_embeddings:
            if self.embedder is None:
                similarity = 0.0
            else:
                similarity = self.embedder.similarity(embedding, known_embedding)

            if similarity > max_similarity:
                max_similarity = similarity
                if similarity > self.similarity_threshold:
                    is_duplicate = True
                    similar_to = known_text[:100]  # First 100 chars

        return DeduplicatorResult(
            text=text,
            embedding=embedding,
            is_duplicate=is_duplicate,
            similar_to=similar_to,
            similarity_score=max_similarity,
        )

    def add_known(self, text: str, embedding: list[float]) -> None:
        """Add a known conversation to the dedup store.

        Args:
            text: Conversation text
            embedding: Conversation embedding
        """
        self.known_embeddings.append((text, embedding))

    def clear(self) -> None:
        """Clear all known embeddings."""
        self.known_embeddings.clear()
