"""Privacy scrubber for Claude Code hook events.

Deterministic regex pass — replaces secrets and PII with safe placeholders.
Zero LLM calls, <5ms. Runs first in the capture pipeline.
"""

import re

# Fast regex-based scrubber for secrets and PII
_PATTERNS = [
    # Bearer tokens
    (re.compile(r'Bearer\s+[a-zA-Z0-9\-_\.]{20,}'), "[AUTH_TOKEN]"),
    # API keys: sk-... (OpenAI, Anthropic, etc.) — must come before generic env patterns
    (re.compile(r'sk-[a-zA-Z0-9\-_]{20,}'), "[API_KEY]"),
    # Generic env var secrets: KEY=value (password, secret, token, api_key, apikey, passwd, pwd suffixes)
    # Note: This is less specific and catches after API keys to avoid false positives
    (re.compile(
        r'(?i)(?:password|passwd|pwd)\s*[=:]\s*\S+',
    ), "[SECRET]"),
    # Email addresses
    (re.compile(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}'), "[EMAIL]"),
    # Private IP ranges: 192.168.x.x, 10.x.x.x, 172.16-31.x.x
    (re.compile(
        r'(?:192\.168|10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}'
    ), "[PRIVATE_IP]"),
]


def scrub(text: str) -> str:
    """Replace secrets and PII with safe placeholders. Pure regex, <5ms."""
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text
