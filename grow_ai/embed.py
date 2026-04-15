import math
import httpx
from grow_ai.config import cfg


def _call_ollama_embed(text: str) -> list[float]:
    """Call nomic-embed-text via Ollama REST API."""
    response = httpx.post(
        f"{cfg.ollama_base_url}/api/embed",
        json={"model": cfg.embedding_model, "input": text},
        timeout=10.0,
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]


def embed(text: str) -> list[float]:
    """Return 768-dim embedding vector for text using nomic-embed-text."""
    return _call_ollama_embed(text)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two equal-length float vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)
