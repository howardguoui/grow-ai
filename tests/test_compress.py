from grow_ai.compress import compress, _rule_based


def test_rule_based_edit_event():
    event = {
        "tool": "Edit",
        "action": "src/auth.py",
        "result": "Added JWT validation, 12 lines changed",
    }
    result = _rule_based(event)
    assert "Edit" in result
    assert "src/auth.py" in result
    assert len(result.split()) <= 25


def test_rule_based_bash_event():
    event = {
        "tool": "Bash",
        "action": "pytest tests/ -v",
        "result": "3 passed, 0 failed",
    }
    result = _rule_based(event)
    assert "Bash" in result
    assert "3 passed" in result


def test_compress_returns_string(mock_ollama_compress):
    event = {"tool": "Read", "action": "README.md", "result": "ok"}
    result = compress(event)
    assert isinstance(result, str)
    assert len(result) > 0


def test_compress_uses_llm_fallback_for_low_signal(mock_ollama_compress):
    event = {"tool": "Read", "action": "x", "result": "y"}
    result = compress(event)
    assert "Fixed LLM" in result
