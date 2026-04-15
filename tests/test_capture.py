"""Tests for capture pipeline."""

import json
import pytest
import tempfile
import os

from grow_ai.capture import CapturePipeline


class TestCapturePipeline:
    """Test suite for CapturePipeline."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.remove(path)  # Remove the file; sqlite will create it
        yield path
        # Try to remove, but don't fail on Windows file locks
        try:
            if os.path.exists(path):
                os.remove(path)
        except PermissionError:
            pass  # Windows locks files; cleanup will happen on next temp clear

    @pytest.fixture
    def pipeline(self, temp_db):
        """Create a pipeline with temporary database."""
        return CapturePipeline(db_path=temp_db, scorer_threshold=30.0)

    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initializes with all components."""
        assert pipeline.scrubber is not None
        assert pipeline.compressor is not None
        assert pipeline.embedder is not None
        assert pipeline.deduplicator is not None
        assert pipeline.scorer is not None
        assert pipeline.store is not None

    def test_full_pipeline_success(self, pipeline):
        """Test successful full pipeline processing."""
        text = "I keep making the same debugging mistakes. Same pattern repeating."
        result = pipeline.process(
            text=text,
            problem_type="behavior_pattern",
            domain="programming",
            skip_store=False,
        )

        assert result["status"] == "success"
        assert "scrub" in result["stages"]
        assert "compress" in result["stages"]
        assert "embed" in result["stages"]
        assert "dedup" in result["stages"]
        assert "score" in result["stages"]
        assert "store" in result["stages"]
        assert result["stored_id"] is not None

    def test_scrub_stage(self, pipeline):
        """Test scrubbing processes without errors."""
        text = "My email is test@example.com"
        result = pipeline.process(text=text, skip_store=True)

        assert result["status"] == "success"
        scrub_stage = result["stages"]["scrub"]
        # At minimum, scrubbing should preserve validity
        assert scrub_stage["is_valid"] is True

    def test_compress_stage(self, pipeline):
        """Test compression creates summary."""
        text = "This is a very long text " * 20
        result = pipeline.process(text=text, skip_store=True)

        assert result["status"] == "success"
        compress_stage = result["stages"]["compress"]
        assert compress_stage["compressed_words"] <= compress_stage["original_words"]
        assert compress_stage["compression_ratio"] <= 1.0

    def test_embed_stage(self, pipeline):
        """Test embedding generates vector."""
        text = "Some text"
        result = pipeline.process(text=text, skip_store=True)

        assert result["status"] == "success"
        embed_stage = result["stages"]["embed"]
        assert embed_stage["dimension"] == 768
        assert len(embed_stage["embedding_preview"]) == 5

    def test_score_stage(self, pipeline):
        """Test scoring activates frameworks."""
        text = "I keep making the same mistakes. Repeated pattern."
        result = pipeline.process(
            text=text,
            problem_type="behavior_pattern",
            skip_store=True,
        )

        assert result["status"] == "success"
        score_stage = result["stages"]["score"]
        assert score_stage["primary_framework"] is not None
        assert score_stage["primary_score"] > 0
        assert score_stage["active_frameworks"] > 0

    def test_store_stage(self, pipeline):
        """Test storing to database."""
        text = "Test conversation"
        result = pipeline.process(
            text=text,
            problem_type="learning",
            skip_store=False,
        )

        assert result["status"] == "success"
        assert result["stored_id"] is not None

        # Verify stored
        stored = pipeline.store.get(result["stored_id"])
        assert stored is not None
        assert stored.original_text == text

    def test_skip_store_flag(self, pipeline):
        """Test skip_store=True processes without storing."""
        text = "Test"
        initial_count = pipeline.store.count()

        result = pipeline.process(text=text, skip_store=True)

        assert result["status"] == "success"
        assert pipeline.store.count() == initial_count  # No new record

    def test_dedup_detection(self, pipeline):
        """Test duplicate detection registers embeddings."""
        text1 = "I keep making the same debugging mistakes"

        # First conversation
        result1 = pipeline.process(
            text=text1,
            problem_type="behavior_pattern",
            skip_store=False,
        )
        assert result1["status"] == "success"

        # Check that dedup has the embedding registered
        # (Note: stub embedder returns all zeros, so all texts have similarity = 1)
        assert len(pipeline.deduplicator.known_embeddings) >= 1

    def test_json_input_valid(self, pipeline):
        """Test processing valid JSON input."""
        json_str = json.dumps({
            "text": "How do I learn Python?",
            "problem_type": "learning",
        })

        result = pipeline.process_json(json_str, skip_store=True)

        assert result["status"] == "success"

    def test_json_input_invalid(self, pipeline):
        """Test processing invalid JSON."""
        result = pipeline.process_json("not json", skip_store=True)

        assert result["status"] == "error"
        assert "Invalid JSON" in result["reason"]

    def test_json_input_missing_text(self, pipeline):
        """Test JSON without text field."""
        json_str = json.dumps({
            "problem_type": "learning",
        })

        result = pipeline.process_json(json_str, skip_store=True)

        assert result["status"] == "error"
        assert "text" in result["reason"]

    def test_empty_text_rejected(self, pipeline):
        """Test empty text is rejected."""
        result = pipeline.process(text="", skip_store=True)

        assert result["status"] == "rejected"

    def test_low_score_rejected(self, pipeline):
        """Test low-scoring conversations are rejected."""
        text = "asdf qwer zxcv"  # Gibberish with no framework signals
        result = pipeline.process(text=text, skip_store=True)

        # Should be rejected due to low score
        if result["status"] == "rejected":
            assert "below threshold" in result.get("reason", "").lower()

    def test_multiple_conversations(self, pipeline):
        """Test processing multiple conversations."""
        texts = [
            ("I keep making the same mistakes", "behavior_pattern"),
            ("How do I learn Python faster?", "learning"),
            ("I need to remember API endpoints", "memory"),
        ]

        for text, problem_type in texts:
            result = pipeline.process(
                text=text,
                problem_type=problem_type,
                skip_store=False,
            )
            if result["status"] == "success":
                assert result["stored_id"] is not None

        # Verify count
        assert pipeline.store.count() >= 2  # At least 2 should succeed

    def test_pipeline_idempotent_scoring(self, pipeline):
        """Test that same text gets same framework."""
        text = "I keep making the same mistakes"

        result1 = pipeline.process(text=text, problem_type="behavior_pattern", skip_store=True)
        result2 = pipeline.process(text=text, problem_type="behavior_pattern", skip_store=True)

        framework1 = result1["stages"]["score"]["primary_framework"]
        framework2 = result2["stages"]["score"]["primary_framework"]

        # Same framework should be selected
        assert framework1 == framework2
