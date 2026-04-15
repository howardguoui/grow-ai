from unittest.mock import patch
import pytest
from grow_ai.embed import embed, cosine_similarity


def test_embed_returns_768_floats():
    fake_vector = [0.1] * 768
    with patch("grow_ai.embed._call_ollama_embed", return_value=fake_vector):
        result = embed("some text")
    assert len(result) == 768
    assert all(isinstance(v, float) for v in result)


def test_cosine_similarity_identical_vectors():
    v = [1.0] + [0.0] * 767
    assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)


def test_cosine_similarity_orthogonal_vectors():
    a = [1.0] + [0.0] * 767
    b = [0.0, 1.0] + [0.0] * 766
    assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)


def test_cosine_similarity_opposite_vectors():
    a = [1.0] + [0.0] * 767
    b = [-1.0] + [0.0] * 767
    assert cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-6)
