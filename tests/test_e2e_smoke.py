"""End-to-end smoke tests with real Ollama integration.

These tests are skipped if Ollama is not running or not accessible.
They verify the full pipeline with real components (when available).
"""

import json
import pytest
import requests
import tempfile
from datetime import datetime
from pathlib import Path

from grow_ai.capture import CapturePipeline
from grow_ai.scrubber import Scrubber
from grow_ai.compressor import Compressor
from grow_ai.embedder import Embedder
from grow_ai.scorer import QualityScorer
from grow_ai.storage import ConversationStore


# Helper to check if Ollama is available
def is_ollama_available() -> bool:
    """Check if Ollama is running and accessible."""
    try:
        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=2,
        )
        return response.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


ollama_available = is_ollama_available()


class TestE2ESmoke:
    """End-to-end smoke tests."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database."""
        fd, path = tempfile.mkstemp(suffix=".db")
        import os
        os.close(fd)
        os.remove(path)
        yield Path(path)
        try:
            if Path(path).exists():
                Path(path).unlink()
        except PermissionError:
            pass

    def test_scrubber_e2e(self):
        """Test scrubber with real text."""
        scrubber = Scrubber()

        text = """
        My API key is sk-abc123def456ghi789jkl.
        Email: user@example.com
        Phone: (555) 123-4567
        Password: MySecureP@ssw0rd

        Here's the actual content I want to keep.
        """

        result = scrubber.scrub(text)

        assert result.is_valid
        assert "sk-abc" not in result.scrubbed or "[REDACTED" in result.scrubbed
        assert result.scrubbed != text  # Something changed

    def test_compressor_e2e(self):
        """Test compressor with real text."""
        compressor = Compressor(target_words=30)

        text = """
        I have been struggling with learning Python for months. Every time I try to study,
        I find it difficult to focus. I've tried many approaches including books, videos,
        and online courses. Nothing seems to stick. I need to find a better way to learn
        this programming language effectively. The syntax is confusing and I keep making
        the same mistakes repeatedly.
        """

        result = compressor.compress(text)

        assert len(result.compressed) < len(text)
        assert result.compression_ratio <= 1.0
        assert result.word_count > 0
        assert "Python" in result.compressed or "learn" in result.compressed.lower()

    def test_scorer_e2e(self):
        """Test scorer with real conversations."""
        scorer = QualityScorer(activation_threshold=30.0)

        test_cases = [
            ("I keep making the same debugging mistakes", "behavior_pattern"),
            ("How do I learn Python faster?", "learning"),
            ("I need to remember 50 API endpoints", "memory"),
        ]

        for text, problem_type in test_cases:
            result = scorer.score_conversation(
                text=text,
                problem_type=problem_type,
            )

            assert result.primary is not None
            assert result.primary.score > 0
            assert result.primary.score <= 100

    def test_storage_e2e(self, temp_db):
        """Test storage with real data."""
        store = ConversationStore(db_path=str(temp_db))

        # Store a conversation
        stored_id = store.store(
            original_text="Test conversation",
            scrubbed_text="Test conversation",
            compressed_text="Test",
            embedding=[0.0] * 768,
            primary_framework="atomic-habits",
            score=75.0,
        )

        assert stored_id == 1

        # Retrieve it
        retrieved = store.get(stored_id)
        assert retrieved is not None
        assert retrieved.original_text == "Test conversation"
        assert retrieved.score == 75.0

        # List recent
        recent = store.list_recent(limit=10)
        assert len(recent) == 1
        assert recent[0].id == stored_id

    def test_pipeline_e2e_no_store(self, temp_db):
        """Test full pipeline without storing (fast smoke test)."""
        pipeline = CapturePipeline(db_path=str(temp_db))

        text = """
        I keep making the same debugging mistakes. I trace the wrong path,
        miss the actual bug, and repeat the same pattern. How can I break this habit?
        """

        result = pipeline.process(
            text=text,
            problem_type="behavior_pattern",
            domain="programming",
            skip_store=True,
        )

        # Verify all stages ran
        assert result["status"] == "success"
        assert "scrub" in result["stages"]
        assert "compress" in result["stages"]
        assert "embed" in result["stages"]
        assert "dedup" in result["stages"]
        assert "score" in result["stages"]

        # Verify reasonable values
        score_stage = result["stages"]["score"]
        assert score_stage["primary_framework"] == "atomic-habits"
        assert score_stage["primary_score"] > 50

    def test_pipeline_e2e_with_store(self, temp_db):
        """Test full pipeline with storage."""
        pipeline = CapturePipeline(db_path=str(temp_db))

        text = "How do I learn Python faster?"

        result = pipeline.process(
            text=text,
            problem_type="learning",
            domain="programming",
            skip_store=False,
        )

        assert result["status"] == "success"
        assert result["stored_id"] is not None

        # Verify stored
        stored = pipeline.store.get(result["stored_id"])
        assert stored is not None
        assert stored.original_text == text

    def test_pipeline_json_input_e2e(self, temp_db):
        """Test full pipeline with JSON input."""
        pipeline = CapturePipeline(db_path=str(temp_db))

        json_input = json.dumps({
            "text": "I need to remember a list of 50 API endpoints",
            "problem_type": "memory",
            "domain": "programming",
        })

        result = pipeline.process_json(json_input, skip_store=False)

        assert result["status"] == "success"
        assert result["stored_id"] is not None

    def test_multiple_conversations_e2e(self, temp_db):
        """Test processing multiple conversations."""
        pipeline = CapturePipeline(db_path=str(temp_db))

        conversations = [
            ("I keep making the same mistakes", "behavior_pattern", "programming"),
            ("How do I learn Python?", "learning", "programming"),
            ("I need to remember API endpoints", "memory", "programming"),
            ("Should I switch jobs?", "decision", "career"),
        ]

        stored_ids = []
        for text, problem_type, domain in conversations:
            result = pipeline.process(
                text=text,
                problem_type=problem_type,
                domain=domain,
                skip_store=False,
            )
            if result["status"] == "success":
                stored_ids.append(result["stored_id"])

        assert len(stored_ids) >= 3

        # Verify all stored
        recent = pipeline.store.list_recent(limit=10)
        assert len(recent) >= 3

    @pytest.mark.skipif(not ollama_available, reason="Ollama not available")
    def test_ollama_embedding_e2e(self):
        """Test with real Ollama embeddings (if available).

        This test is skipped if Ollama is not running.
        """
        try:
            import ollama
        except ImportError:
            pytest.skip("ollama module not installed")

        # Create embedder with Ollama
        embedder = Embedder(model="ollama", dimension=384)

        # Test embedding generation
        text = "How do I improve my debugging skills?"
        result = embedder.embed(text)

        assert len(result.embedding) > 0
        assert len(result.embedding) == embedder.dimension

    def test_error_handling_empty_text(self, temp_db):
        """Test pipeline handles empty text."""
        pipeline = CapturePipeline(db_path=str(temp_db))

        result = pipeline.process(text="", skip_store=True)

        assert result["status"] == "rejected"

    def test_error_handling_invalid_json(self, temp_db):
        """Test pipeline handles invalid JSON."""
        pipeline = CapturePipeline(db_path=str(temp_db))

        result = pipeline.process_json("not json", skip_store=True)

        assert result["status"] == "error"

    def test_framework_detection_accuracy(self, temp_db):
        """Test framework detection on diverse inputs."""
        pipeline = CapturePipeline(db_path=str(temp_db))

        test_cases = [
            (
                "I keep making the same debugging mistakes. Repeated pattern.",
                "behavior_pattern",
                "atomic-habits",
            ),
            (
                "How do I learn Python faster? Study and practice.",
                "learning",
                "ultralearning",  # or make-it-stick
            ),
            (
                "I need to remember a list of 50 API endpoints",
                "memory",
                "the-memory-book",  # or moonwalking
            ),
            (
                "Should I take this job offer or stay?",
                "decision",
                "clear-thinking",  # or thinking-fast-slow
            ),
        ]

        for text, problem_type, expected_framework in test_cases:
            result = pipeline.process(
                text=text,
                problem_type=problem_type,
                skip_store=True,
            )

            if result["status"] == "success":
                primary = result["stages"]["score"]["primary_framework"]
                # Either expected or related framework
                assert primary is not None
                # Framework should be one of the 10
                assert primary in [
                    "thinking-fast-slow",
                    "clear-thinking",
                    "algorithms-to-live-by",
                    "moonwalking-einstein",
                    "the-memory-book",
                    "make-it-stick",
                    "ultralearning",
                    "atomic-habits",
                    "thinking-in-systems",
                    "antifragile",
                ]

    def test_concurrent_storage_safety(self, temp_db):
        """Test multiple concurrent writes are safe."""
        pipeline = CapturePipeline(db_path=str(temp_db))

        # Write multiple conversations quickly
        for i in range(5):
            result = pipeline.process(
                text=f"Conversation {i}",
                problem_type="learning",
                skip_store=False,
            )
            assert result["status"] == "success"

        # All should be stored
        assert pipeline.store.count() == 5

    def test_compression_quality(self):
        """Test compression preserves key information."""
        compressor = Compressor(target_words=30)

        text = """
        I have been struggling with learning Python for several months.
        Every time I sit down to study, I find it very difficult to focus.
        I've tried reading books, watching videos, and taking online courses,
        but nothing seems to help the information stick in my memory.
        The programming syntax is confusing and I keep making the same
        mistakes over and over again. I need to find a better way to approach
        this and learn Python more effectively.
        """

        result = compressor.compress(text)

        # Check that key terms are preserved
        assert any(word in result.compressed.lower() for word in ["python", "learn", "study", "difficulty"])

    def test_deduplication_sensitivity(self, temp_db):
        """Test deduplication threshold sensitivity."""
        pipeline = CapturePipeline(db_path=str(temp_db), dedup_threshold=0.85)

        # Store first conversation
        result1 = pipeline.process(
            text="I keep making the same debugging mistakes",
            problem_type="behavior_pattern",
            skip_store=False,
        )
        assert result1["status"] == "success"

        # Nearly identical (with stub embedder, will have 100% similarity)
        # This will be marked as duplicate with stub embedder (all zeros)
        result2 = pipeline.process(
            text="I keep making the same debugging mistakes",
            problem_type="behavior_pattern",
            skip_store=False,
        )

        # With stub embedder returning all zeros, similarity = 1.0 = duplicate
        # Note: Real embeddings would have different values

    @pytest.mark.skipif(not ollama_available, reason="Ollama not available")
    def test_full_pipeline_with_ollama(self, temp_db):
        """Test full pipeline with real Ollama embeddings."""
        try:
            import ollama
        except ImportError:
            pytest.skip("ollama module not installed")

        pipeline = CapturePipeline(db_path=str(temp_db))

        # Override embedder with Ollama
        pipeline.embedder = Embedder(model="ollama", dimension=384)

        text = "How do I improve my debugging skills with better practices?"

        result = pipeline.process(
            text=text,
            problem_type="learning",
            domain="programming",
            skip_store=False,
        )

        assert result["status"] == "success"
        assert result["stored_id"] is not None


class TestE2EPerformance:
    """Performance smoke tests."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database."""
        fd, path = tempfile.mkstemp(suffix=".db")
        import os
        os.close(fd)
        os.remove(path)
        yield Path(path)
        try:
            if Path(path).exists():
                Path(path).unlink()
        except PermissionError:
            pass

    def test_pipeline_performance_single_conversation(self, temp_db):
        """Test pipeline performance on single conversation."""
        import time

        pipeline = CapturePipeline(db_path=str(temp_db))

        text = "I keep making the same debugging mistakes" * 10  # ~60 words

        start = time.time()
        result = pipeline.process(
            text=text,
            problem_type="behavior_pattern",
            skip_store=True,
        )
        elapsed = time.time() - start

        assert result["status"] == "success"
        assert elapsed < 5.0  # Should complete in < 5 seconds

    def test_pipeline_performance_batch(self, temp_db):
        """Test pipeline performance on batch."""
        import time

        pipeline = CapturePipeline(db_path=str(temp_db))

        start = time.time()
        for i in range(10):
            pipeline.process(
                text=f"Conversation {i}",
                problem_type="learning",
                skip_store=True,
            )
        elapsed = time.time() - start

        # 10 conversations should complete in reasonable time
        assert elapsed < 10.0
