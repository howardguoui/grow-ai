from grow_ai.scrubber import scrub


def test_masks_openai_style_api_key():
    text = "Using key sk-abc123XYZabc123XYZabc123XYZabc123 to call API"
    assert "[API_KEY]" in scrub(text)
    assert "sk-abc123" not in scrub(text)


def test_masks_anthropic_api_key():
    text = "ANTHROPIC_API_KEY=sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890"
    assert "[API_KEY]" in scrub(text)


def test_masks_bearer_token():
    text = "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.signature"
    assert "[AUTH_TOKEN]" in scrub(text)
    assert "eyJ" not in scrub(text)


def test_masks_generic_password_in_env():
    text = "DB_PASSWORD=super_secret_password_123"
    assert "[SECRET]" in scrub(text)


def test_masks_email_address():
    text = "Send results to user@example.com please"
    assert "[EMAIL]" in scrub(text)
    assert "user@example.com" not in scrub(text)


def test_passes_clean_text_unchanged():
    text = "Edit src/app.py: added login route — returns 200 on success"
    assert scrub(text) == text


def test_masks_private_ip():
    text = "Server running at http://192.168.1.100:8080"
    assert "[PRIVATE_IP]" in scrub(text)
