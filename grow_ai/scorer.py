"""Quality scorer for Claude Code conversations.

Scores conversations against 9 book frameworks and applies temporal decay.
"""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional


# Framework definitions (from personal-ai-discussion framework design)
FRAMEWORKS = {
    "thinking-fast-slow": {
        "title": "Thinking, Fast and Slow",
        "category": "Thinking & Logic",
        "keywords": ["bias", "heuristic", "intuition", "pattern", "automatic", "feel", "obvious"],
        "problem_types": ["decision", "behavior_pattern"],
        "base_score": 30,
    },
    "clear-thinking": {
        "title": "Clear Thinking",
        "category": "Thinking & Logic",
        "keywords": ["emotion", "fear", "reaction", "defensive", "ego", "impulse", "pause"],
        "problem_types": ["decision", "behavior_pattern"],
        "base_score": 35,
    },
    "algorithms-to-live-by": {
        "title": "Algorithms to Live By",
        "category": "Thinking & Logic",
        "keywords": ["sort", "search", "optimize", "decision tree", "tradeoff", "explore"],
        "problem_types": ["debugging", "design", "learning"],
        "base_score": 20,
    },
    "moonwalking-einstein": {
        "title": "Moonwalking with Einstein",
        "category": "Memory & Study",
        "keywords": ["memorize", "visualize", "image", "anchor", "palace", "vivid"],
        "problem_types": ["memory"],
        "base_score": 50,
    },
    "the-memory-book": {
        "title": "The Memory Book",
        "category": "Memory & Study",
        "keywords": ["number", "name", "face", "list", "sequence", "encode", "peg"],
        "problem_types": ["memory"],
        "base_score": 50,
    },
    "make-it-stick": {
        "title": "Make It Stick",
        "category": "Learning & Mastery",
        "keywords": ["study", "practice", "remember", "test", "space", "interleave", "recall"],
        "problem_types": ["learning"],
        "base_score": 35,
    },
    "ultralearning": {
        "title": "Ultralearning",
        "category": "Learning & Mastery",
        "keywords": ["skill", "decompose", "feedback", "encode", "project", "transfer"],
        "problem_types": ["learning"],
        "base_score": 40,
    },
    "atomic-habits": {
        "title": "Atomic Habits",
        "category": "Systems & Evolution",
        "keywords": ["repeated", "habit", "routine", "cue", "trigger", "consistency", "daily"],
        "problem_types": ["behavior_pattern"],
        "base_score": 55,
    },
    "thinking-in-systems": {
        "title": "Thinking in Systems",
        "category": "Systems & Evolution",
        "keywords": ["loop", "feedback", "pattern", "delay", "leverage", "system", "recur"],
        "problem_types": ["systems", "design", "debugging"],
        "base_score": 50,
    },
    "antifragile": {
        "title": "Antifragile",
        "category": "Systems & Evolution",
        "keywords": ["volatility", "optionality", "barbell", "black swan", "fragile", "trial"],
        "problem_types": ["decision", "design", "systems"],
        "base_score": 20,
    },
}

# Signal weights (importance multipliers)
SIGNAL_WEIGHTS = {
    "keyword_match": 5.0,  # Exact keyword found
    "problem_type_match": 20.0,  # Problem type matches framework
    "domain_match": 5.0,  # Domain (programming, career, health) matches
    "thinking_pattern_match": 8.0,  # Thinking style matches
    "emotional_tone_match": 3.0,  # Emotional context matches
}

# Temporal decay parameters
DECAY_HALF_LIFE_DAYS = 90  # Score decays to 50% after 90 days
DECAY_MIN_SCORE = 5.0  # Minimum score floor (prevents complete deactivation)


@dataclass
class FrameworkScore:
    """Score for a single framework."""

    framework_id: str
    title: str
    category: str
    score: float
    signals_found: list[str]
    activated: bool


@dataclass
class ScorerResult:
    """Result of scoring a conversation."""

    all_scores: list[FrameworkScore]
    active_frameworks: list[FrameworkScore]
    primary: Optional[FrameworkScore]
    secondary: Optional[FrameworkScore]
    tertiary: Optional[FrameworkScore]
    timestamp: datetime


class QualityScorer:
    """Score Claude Code conversations against book frameworks."""

    def __init__(self, activation_threshold: float = 30.0):
        """Initialize scorer.

        Args:
            activation_threshold: Minimum score (0-100) to activate framework.
        """
        self.activation_threshold = activation_threshold
        self.frameworks = FRAMEWORKS

    def score_conversation(
        self,
        text: str,
        problem_type: Optional[str] = None,
        domain: Optional[str] = None,
        created_at: Optional[datetime] = None,
    ) -> ScorerResult:
        """Score a conversation against all frameworks.

        Args:
            text: Conversation text to score
            problem_type: Detected problem type (learning, debugging, decision, etc.)
            domain: Problem domain (programming, career, health, finance)
            created_at: When the conversation was created (for temporal decay)

        Returns:
            ScorerResult with all scores and activated frameworks
        """
        if created_at is None:
            created_at = datetime.now()

        text_lower = text.lower()
        scores = []

        # Score each framework
        for framework_id, framework_info in self.frameworks.items():
            score, signals = self._score_framework(
                framework_id,
                framework_info,
                text_lower,
                problem_type,
                domain,
            )

            # Apply temporal decay
            score = self._apply_temporal_decay(score, created_at)

            scores.append(
                FrameworkScore(
                    framework_id=framework_id,
                    title=framework_info["title"],
                    category=framework_info["category"],
                    score=score,
                    signals_found=signals,
                    activated=score > self.activation_threshold,
                )
            )

        # Sort by score
        scores.sort(key=lambda s: s.score, reverse=True)

        # Get active frameworks (threshold exceeded)
        active = [s for s in scores if s.activated]

        # Get top 3 for primary/secondary/tertiary
        primary = active[0] if len(active) > 0 else None
        secondary = active[1] if len(active) > 1 else None
        tertiary = active[2] if len(active) > 2 else None

        return ScorerResult(
            all_scores=scores,
            active_frameworks=active,
            primary=primary,
            secondary=secondary,
            tertiary=tertiary,
            timestamp=datetime.now(),
        )

    def _score_framework(
        self,
        framework_id: str,
        framework_info: dict,
        text_lower: str,
        problem_type: Optional[str],
        domain: Optional[str],
    ) -> tuple[float, list[str]]:
        """Score a single framework.

        Returns:
            (score, list of detected signals)
        """
        score = framework_info["base_score"]
        signals_found = []

        # Check keyword matches
        keywords = framework_info.get("keywords", [])
        matching_keywords = [kw for kw in keywords if kw in text_lower]
        if matching_keywords:
            signals_found.extend(matching_keywords)
            score += len(matching_keywords) * SIGNAL_WEIGHTS["keyword_match"]

        # Check problem type match
        if problem_type and problem_type in framework_info.get("problem_types", []):
            signals_found.append(f"problem_type:{problem_type}")
            score += SIGNAL_WEIGHTS["problem_type_match"]

        # Cap score at 100
        score = min(score, 100.0)

        return score, signals_found

    def _apply_temporal_decay(
        self, score: float, created_at: datetime
    ) -> float:
        """Apply temporal decay to score.

        Score decays exponentially with a configurable half-life.
        Recent conversations have higher scores; old ones decay.

        Args:
            score: Original score (0-100)
            created_at: When the conversation was created

        Returns:
            Decayed score (minimum DECAY_MIN_SCORE)
        """
        age_days = (datetime.now() - created_at).total_seconds() / (24 * 3600)

        # Exponential decay: score * (0.5 ^ (age / half_life))
        decay_factor = 0.5 ** (age_days / DECAY_HALF_LIFE_DAYS)
        decayed_score = score * decay_factor

        # Floor at minimum
        return max(decayed_score, DECAY_MIN_SCORE)

    def format_result(self, result: ScorerResult) -> str:
        """Format scoring result as readable text.

        Args:
            result: ScorerResult from score_conversation()

        Returns:
            Formatted string summary
        """
        lines = [
            "=== Framework Quality Score ===",
            f"Timestamp: {result.timestamp.isoformat()}",
            "",
            f"Active Frameworks: {len(result.active_frameworks)}/10",
            "",
        ]

        if result.primary:
            lines.append(f"PRIMARY: {result.primary.title} ({result.primary.score:.0f}/100)")
            if result.primary.signals_found:
                lines.append(f"  Signals: {', '.join(result.primary.signals_found[:3])}")

        if result.secondary:
            lines.append(
                f"SECONDARY: {result.secondary.title} ({result.secondary.score:.0f}/100)"
            )

        if result.tertiary:
            lines.append(f"TERTIARY: {result.tertiary.title} ({result.tertiary.score:.0f}/100)")

        lines.extend(["", "All Scores:"])
        for score in result.all_scores[:5]:  # Top 5
            marker = "*" if score.activated else " "
            lines.append(f"  {marker} {score.title.ljust(30)} {score.score:.0f}/100")

        return "\n".join(lines)


def example_usage():
    """Example usage of QualityScorer."""
    scorer = QualityScorer(activation_threshold=30.0)

    # Example conversation
    text = """
    I keep making the same debugging mistakes. I trace the wrong path and miss
    the actual bug. How do I break this pattern? I've tried focusing harder but
    it's still the same issues repeating.
    """

    result = scorer.score_conversation(
        text=text,
        problem_type="behavior_pattern",
        domain="programming",
        created_at=datetime.now() - timedelta(days=1),
    )

    print(scorer.format_result(result))
    print("\n---\n")

    # Example with decay (old conversation)
    result_old = scorer.score_conversation(
        text=text,
        problem_type="behavior_pattern",
        domain="programming",
        created_at=datetime.now() - timedelta(days=60),  # 60 days old
    )

    print("After 60 days (temporal decay applied):")
    print(scorer.format_result(result_old))


if __name__ == "__main__":
    example_usage()
