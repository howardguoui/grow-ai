"""Capture pipeline for Claude Code conversations.

Full pipeline: scrub → compress → embed → dedup → score → store
Reads from stdin as JSON, processes, and stores to SQLite.
"""

import json
import sys
from datetime import datetime
from typing import Optional

from grow_ai.scrubber import Scrubber
from grow_ai.compressor import Compressor
from grow_ai.embedder import Embedder
from grow_ai.deduplicator import Deduplicator
from grow_ai.scorer import QualityScorer
from grow_ai.storage import ConversationStore


class CapturePipeline:
    """Process conversations through the full pipeline."""

    def __init__(
        self,
        db_path: str = "grow_ai.db",
        scorer_threshold: float = 30.0,
        dedup_threshold: float = 0.85,
    ):
        """Initialize pipeline.

        Args:
            db_path: Path to SQLite database
            scorer_threshold: Minimum score to store (0-100)
            dedup_threshold: Threshold for deduplication (0-1)
        """
        self.scrubber = Scrubber(redact_emails=True, redact_pii=True)
        self.compressor = Compressor(target_words=30)
        self.embedder = Embedder(model="local", dimension=768)
        self.deduplicator = Deduplicator(
            similarity_threshold=dedup_threshold,
            embedder=self.embedder,
        )
        self.scorer = QualityScorer(activation_threshold=scorer_threshold)
        self.store = ConversationStore(db_path)
        self.scorer_threshold = scorer_threshold

    def process(
        self,
        text: str,
        problem_type: Optional[str] = None,
        domain: Optional[str] = None,
        skip_store: bool = False,
    ) -> dict:
        """Process a conversation through the full pipeline.

        Args:
            text: Conversation text
            problem_type: Detected problem type (optional)
            domain: Problem domain (optional)
            skip_store: If True, only process without storing

        Returns:
            Result dict with all pipeline stages
        """
        result = {
            "status": "success",
            "stages": {},
            "stored_id": None,
        }

        # Stage 1: Scrub
        scrub_result = self.scrubber.scrub(text)
        result["stages"]["scrub"] = {
            "issues_found": scrub_result.issues_found,
            "is_valid": scrub_result.is_valid,
            "scrubbed_text": scrub_result.scrubbed,
        }

        if not scrub_result.is_valid:
            result["status"] = "rejected"
            result["reason"] = "Invalid or empty after scrubbing"
            return result

        # Stage 2: Compress
        compress_result = self.compressor.compress(scrub_result.scrubbed)
        result["stages"]["compress"] = {
            "original_words": len(scrub_result.scrubbed.split()),
            "compressed_words": compress_result.word_count,
            "compression_ratio": compress_result.compression_ratio,
            "compressed_text": compress_result.compressed,
        }

        # Stage 3: Embed
        embed_result = self.embedder.embed(scrub_result.scrubbed)
        result["stages"]["embed"] = {
            "model": embed_result.model,
            "dimension": embed_result.dimension,
            "embedding_preview": embed_result.embedding[:5],  # First 5 values
        }

        # Stage 4: Dedup
        dedup_result = self.deduplicator.check_duplicate(
            text=scrub_result.scrubbed,
            embedding=embed_result.embedding,
        )
        result["stages"]["dedup"] = {
            "is_duplicate": dedup_result.is_duplicate,
            "similarity_score": dedup_result.similarity_score,
            "similar_to": dedup_result.similar_to,
        }

        if dedup_result.is_duplicate:
            result["status"] = "skipped"
            result["reason"] = f"Duplicate (similarity: {dedup_result.similarity_score:.2f})"
            return result

        # Stage 5: Score
        score_result = self.scorer.score_conversation(
            text=scrub_result.scrubbed,
            problem_type=problem_type,
            domain=domain,
            created_at=datetime.now(),
        )
        result["stages"]["score"] = {
            "primary_framework": score_result.primary.framework_id if score_result.primary else None,
            "primary_score": score_result.primary.score if score_result.primary else 0,
            "active_frameworks": len(score_result.active_frameworks),
            "all_scores": [
                {
                    "framework": s.framework_id,
                    "score": s.score,
                }
                for s in score_result.all_scores[:5]
            ],
        }

        # Check if score meets threshold
        if score_result.primary is None or score_result.primary.score < self.scorer_threshold:
            result["status"] = "rejected"
            result["reason"] = f"Score below threshold: {score_result.primary.score if score_result.primary else 0}"
            return result

        # Stage 6: Store
        if not skip_store:
            stored_id = self.store.store(
                original_text=text,
                scrubbed_text=scrub_result.scrubbed,
                compressed_text=compress_result.compressed,
                embedding=embed_result.embedding,
                primary_framework=score_result.primary.framework_id,
                score=score_result.primary.score,
            )
            result["stored_id"] = stored_id
            self.deduplicator.add_known(
                text=scrub_result.scrubbed,
                embedding=embed_result.embedding,
            )
            result["stages"]["store"] = {
                "id": stored_id,
            }
        else:
            result["stages"]["store"] = {
                "status": "skipped",
            }

        return result

    def process_json(self, json_input: str, skip_store: bool = False) -> dict:
        """Process JSON input from stdin/hook.

        Expected JSON format:
        {
            "text": "conversation text",
            "problem_type": "learning|debugging|decision|...",
            "domain": "programming|career|health|..."
        }

        Args:
            json_input: JSON string
            skip_store: If True, only process without storing

        Returns:
            Result dict
        """
        try:
            data = json.loads(json_input)
            text = data.get("text", "")
            problem_type = data.get("problem_type")
            domain = data.get("domain")

            if not text:
                return {
                    "status": "error",
                    "reason": "No 'text' field in JSON",
                }

            return self.process(
                text=text,
                problem_type=problem_type,
                domain=domain,
                skip_store=skip_store,
            )
        except json.JSONDecodeError as e:
            return {
                "status": "error",
                "reason": f"Invalid JSON: {e}",
            }


def main():
    """Entry point for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Capture Claude Code conversations into grow_ai store"
    )
    parser.add_argument(
        "--db",
        default="grow_ai.db",
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process without storing",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run with test data",
    )

    args = parser.parse_args()

    pipeline = CapturePipeline(db_path=args.db)

    if args.test:
        # Test with example data
        test_text = """
        I keep making the same debugging mistakes. I trace the wrong path and miss
        the actual bug. How do I break this pattern? I've tried focusing harder but
        it's still the same issues repeating.
        """
        result = pipeline.process(
            text=test_text,
            problem_type="behavior_pattern",
            domain="programming",
            skip_store=args.dry_run,
        )
        print(json.dumps(result, indent=2))
    else:
        # Read from stdin
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            result = pipeline.process_json(line, skip_store=args.dry_run)
            print(json.dumps(result))


if __name__ == "__main__":
    main()
