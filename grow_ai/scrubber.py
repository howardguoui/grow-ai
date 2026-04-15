"""Data scrubbing for Claude Code conversations.

Removes sensitive information, normalizes formatting, and validates structure.
"""

import re
from dataclasses import dataclass


@dataclass
class ScrubResult:
    """Result of scrubbing a conversation."""

    original: str
    scrubbed: str
    issues_found: list[str]
    is_valid: bool


class Scrubber:
    """Scrub sensitive data from conversations."""

    # Patterns to remove or redact
    PATTERNS = {
        "api_key": r"(?:api[_-]?key|token|secret)\s*[:=]\s*['\"]?[\w\-]{20,}['\"]?",
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "password": r"(?:password|passwd)\s*[:=]\s*['\"]?[^\s'\"]+['\"]?",
    }

    def __init__(self, redact_emails: bool = True, redact_pii: bool = True):
        """Initialize scrubber.

        Args:
            redact_emails: Whether to redact email addresses
            redact_pii: Whether to redact PII (phone, SSN, credit cards)
        """
        self.redact_emails = redact_emails
        self.redact_pii = redact_pii

    def scrub(self, text: str) -> ScrubResult:
        """Scrub sensitive data from text.

        Args:
            text: Text to scrub

        Returns:
            ScrubResult with scrubbed text and issues found
        """
        scrubbed = text
        issues = []

        # Check for and redact API keys
        if re.search(self.PATTERNS["api_key"], scrubbed, re.IGNORECASE):
            issues.append("API key/token found")
            scrubbed = re.sub(
                self.PATTERNS["api_key"],
                "[REDACTED_API_KEY]",
                scrubbed,
                flags=re.IGNORECASE,
            )

        # Check for and redact emails
        if self.redact_emails and re.search(self.PATTERNS["email"], scrubbed):
            issues.append("Email address found")
            scrubbed = re.sub(
                self.PATTERNS["email"],
                "[REDACTED_EMAIL]",
                scrubbed,
            )

        # Check for PII
        if self.redact_pii:
            if re.search(self.PATTERNS["phone"], scrubbed):
                issues.append("Phone number found")
                scrubbed = re.sub(
                    self.PATTERNS["phone"],
                    "[REDACTED_PHONE]",
                    scrubbed,
                )

            if re.search(self.PATTERNS["ssn"], scrubbed):
                issues.append("SSN found")
                scrubbed = re.sub(
                    self.PATTERNS["ssn"],
                    "[REDACTED_SSN]",
                    scrubbed,
                )

            if re.search(self.PATTERNS["credit_card"], scrubbed):
                issues.append("Credit card found")
                scrubbed = re.sub(
                    self.PATTERNS["credit_card"],
                    "[REDACTED_CC]",
                    scrubbed,
                )

            if re.search(self.PATTERNS["password"], scrubbed, re.IGNORECASE):
                issues.append("Password found")
                scrubbed = re.sub(
                    self.PATTERNS["password"],
                    "[REDACTED_PASSWORD]",
                    scrubbed,
                    flags=re.IGNORECASE,
                )

        # Normalize whitespace
        scrubbed = re.sub(r"\n\s*\n", "\n\n", scrubbed)  # Double newlines
        scrubbed = scrubbed.strip()

        is_valid = len(scrubbed) > 0

        return ScrubResult(
            original=text,
            scrubbed=scrubbed,
            issues_found=issues,
            is_valid=is_valid,
        )
