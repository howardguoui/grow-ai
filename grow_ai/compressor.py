"""Text compression for conversation summaries.

Compresses lengthy conversations into concise 20-50 word insights.
"""

from dataclasses import dataclass


@dataclass
class CompressorResult:
    """Result of compressing text."""

    original: str
    compressed: str
    word_count: int
    compression_ratio: float


class Compressor:
    """Compress text to key insights."""

    def __init__(self, target_words: int = 30):
        """Initialize compressor.

        Args:
            target_words: Target word count for compressed output (20-50)
        """
        self.target_words = max(20, min(target_words, 50))

    def compress(self, text: str) -> CompressorResult:
        """Compress text to key insights.

        Simple heuristic approach:
        1. Extract sentences
        2. Score by keyword importance
        3. Select top sentences until target word count
        4. Return as compressed summary

        Args:
            text: Text to compress

        Returns:
            CompressorResult with compressed text
        """
        # Split into sentences
        sentences = [s.strip() for s in text.split(".") if s.strip()]

        if not sentences:
            return CompressorResult(
                original=text,
                compressed=text[:self.target_words * 5],  # Approximate truncation
                word_count=len(text.split()),
                compression_ratio=1.0,
            )

        # Simple heuristic: prefer sentences with action words, questions, or statements
        action_words = {
            "need", "want", "learn", "fix", "build", "debug", "improve", "create",
            "understand", "solve", "optimize", "implement", "design", "test",
        }

        def score_sentence(s: str) -> int:
            """Score a sentence for importance."""
            s_lower = s.lower()
            score = 0

            # Action verbs
            score += sum(2 for word in action_words if word in s_lower)

            # Questions
            if "?" in s:
                score += 3

            # Problem statements
            if any(w in s_lower for w in ["problem", "issue", "bug", "error"]):
                score += 2

            # Length preference (not too short, not too long)
            word_count = len(s.split())
            if 10 <= word_count <= 25:
                score += 1

            return score

        # Sort sentences by importance
        scored = [(s, score_sentence(s)) for s in sentences]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Build compressed output
        compressed_parts = []
        word_count = 0

        for sentence, _ in scored:
            sentence_words = len(sentence.split())
            if word_count + sentence_words <= self.target_words:
                compressed_parts.append(sentence)
                word_count += sentence_words

            if word_count >= self.target_words * 0.9:  # Close enough
                break

        # Join and ensure proper punctuation
        compressed = ". ".join(compressed_parts).rstrip(".")
        if compressed and not compressed.endswith((".", "?", "!")):
            compressed += "."

        original_words = len(text.split())
        compression_ratio = word_count / original_words if original_words > 0 else 0

        return CompressorResult(
            original=text,
            compressed=compressed,
            word_count=word_count,
            compression_ratio=compression_ratio,
        )
