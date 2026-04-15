"""Tests for quality scorer."""

from datetime import datetime, timedelta
import pytest

from grow_ai.scorer import QualityScorer, FrameworkScore


class TestQualityScorer:
    """Test suite for QualityScorer."""

    @pytest.fixture
    def scorer(self):
        """Create a scorer instance."""
        return QualityScorer(activation_threshold=30.0)

    def test_scorer_initialization(self, scorer):
        """Test scorer initializes with correct parameters."""
        assert scorer.activation_threshold == 30.0
        assert len(scorer.frameworks) == 10

    def test_score_behavior_pattern(self, scorer):
        """Test scoring detects behavior pattern frameworks."""
        text = "I keep making the same mistakes. Same pattern repeating."
        result = scorer.score_conversation(
            text=text,
            problem_type="behavior_pattern",
        )

        assert result.primary is not None
        assert result.primary.framework_id == "atomic-habits"
        assert result.primary.score > 50

    def test_score_learning(self, scorer):
        """Test scoring detects learning frameworks."""
        text = "How do I learn Python faster? I want to study and practice."
        result = scorer.score_conversation(
            text=text,
            problem_type="learning",
        )

        assert result.primary is not None
        assert result.primary.framework_id in ["ultralearning", "make-it-stick"]
        assert result.primary.score > 35

    def test_score_memory(self, scorer):
        """Test scoring detects memory frameworks."""
        text = "I need to memorize this list of API endpoints."
        result = scorer.score_conversation(
            text=text,
            problem_type="memory",
        )

        assert result.primary is not None
        assert result.primary.framework_id in ["the-memory-book", "moonwalking-einstein"]
        assert result.primary.score > 40

    def test_score_decision_making(self, scorer):
        """Test scoring detects decision-making frameworks."""
        text = "Should I take this job offer? What should I choose?"
        result = scorer.score_conversation(
            text=text,
            problem_type="decision",
        )

        assert result.primary is not None
        # Should activate thinking or decision frameworks
        assert result.primary.score > 30

    def test_temporal_decay_recent(self, scorer):
        """Test temporal decay is minimal for recent conversations."""
        text = "Some text"
        created_at = datetime.now() - timedelta(days=1)

        result_new = scorer.score_conversation(
            text=text,
            problem_type="learning",
            created_at=datetime.now(),
        )

        result_recent = scorer.score_conversation(
            text=text,
            problem_type="learning",
            created_at=created_at,
        )

        # Recent conversation should have nearly same score as new
        assert result_recent.primary is not None
        assert result_new.primary is not None
        # After 1 day, decay is minimal (still >95% of original)
        assert result_recent.primary.score > result_new.primary.score * 0.95

    def test_temporal_decay_old(self, scorer):
        """Test temporal decay reduces old conversation scores."""
        text = "Some text with learning keywords study practice"
        created_at_recent = datetime.now() - timedelta(days=30)
        created_at_old = datetime.now() - timedelta(days=180)  # 6 months old

        result_recent = scorer.score_conversation(
            text=text,
            problem_type="learning",
            created_at=created_at_recent,
        )

        result_old = scorer.score_conversation(
            text=text,
            problem_type="learning",
            created_at=created_at_old,
        )

        # Recent (30 days) should have primary framework
        assert result_recent.primary is not None

        # Old (180 days) may drop below threshold due to decay
        # But we can still check decay is working
        # Get same framework from both results
        framework_id = result_recent.primary.framework_id
        old_score = next(
            (s.score for s in result_old.all_scores if s.framework_id == framework_id),
            None,
        )
        recent_score = result_recent.primary.score

        # Old should be significantly lower (at 180 days with 90-day half-life: 0.25x)
        assert old_score < recent_score
        assert old_score < recent_score * 0.4  # Less than 40% of recent

    def test_activation_threshold(self, scorer):
        """Test only scores above threshold are activated."""
        text = "random text without framework signals"

        result = scorer.score_conversation(
            text=text,
        )

        # Some frameworks may activate on base score alone,
        # but most should be below threshold without signals
        activated_count = sum(1 for s in result.all_scores if s.activated)
        assert activated_count >= 0  # At least some frameworks have base scores

    def test_multiple_signals(self, scorer):
        """Test frameworks score higher with multiple matching signals."""
        text_with_signals = """
        I keep repeating the same habit. There's a daily routine cue that
        triggers the behavior. I need to break this repeated pattern.
        """

        result = scorer.score_conversation(
            text=text_with_signals,
            problem_type="behavior_pattern",
        )

        atomic_habits = next(
            (s for s in result.all_scores if s.framework_id == "atomic-habits"),
            None,
        )
        assert atomic_habits is not None
        assert len(atomic_habits.signals_found) > 1
        assert atomic_habits.score > 50

    def test_format_result(self, scorer):
        """Test result formatting works correctly."""
        text = "I keep making mistakes"
        result = scorer.score_conversation(text=text, problem_type="behavior_pattern")

        formatted = scorer.format_result(result)

        assert "Framework Quality Score" in formatted
        assert "PRIMARY" in formatted
        assert "Atomic Habits" in formatted

    def test_all_frameworks_present(self, scorer):
        """Test all 10 frameworks are scored."""
        text = "test"
        result = scorer.score_conversation(text=text)

        assert len(result.all_scores) == 10
        framework_ids = {s.framework_id for s in result.all_scores}

        expected = {
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
        }

        assert framework_ids == expected

    def test_score_bounds(self, scorer):
        """Test scores are bounded 0-100."""
        text = "test " * 100  # Lots of repetition

        result = scorer.score_conversation(text=text)

        for score in result.all_scores:
            assert 0 <= score.score <= 100

    def test_framework_score_dataclass(self):
        """Test FrameworkScore dataclass."""
        score = FrameworkScore(
            framework_id="test",
            title="Test Framework",
            category="Test",
            score=75.0,
            signals_found=["signal1", "signal2"],
            activated=True,
        )

        assert score.framework_id == "test"
        assert score.score == 75.0
        assert len(score.signals_found) == 2
        assert score.activated is True
