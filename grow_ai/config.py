from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # Paths
    grow_ai_dir: Path = field(default_factory=lambda: Path.home() / ".grow-ai")
    db_path: Path = field(default=None)
    state_path: Path = field(default=None)

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    generative_model: str = "qwen2.5:3b"
    embedding_model: str = "nomic-embed-text"

    # Quality scoring
    quality_threshold: int = 15
    finetune_batch_size: int = 50

    # Dedup thresholds
    dedup_discard: float = 0.95
    dedup_merge: float = 0.80

    # Temporal decay: 2% per week
    decay_rate_per_week: float = 0.02

    def __post_init__(self):
        if self.db_path is None:
            self.db_path = self.grow_ai_dir / "insights.db"
        if self.state_path is None:
            self.state_path = self.grow_ai_dir / "state.json"
        self.grow_ai_dir.mkdir(parents=True, exist_ok=True)


# Singleton used across modules
cfg = Config()
