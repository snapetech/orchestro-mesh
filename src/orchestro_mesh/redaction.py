from __future__ import annotations

import re
from dataclasses import dataclass

SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("generic_token", re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*['\"]?[^\s'\"]+")),
    ("ssh_private_key", re.compile(r"-----BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY-----")),
    ("github_token", re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}")),
)


@dataclass(frozen=True)
class RedactionResult:
    text: str
    findings: list[str]


def redact_text(text: str) -> RedactionResult:
    findings: list[str] = []
    redacted = text
    for name, pattern in SECRET_PATTERNS:
        if pattern.search(redacted):
            findings.append(name)
            redacted = pattern.sub(f"[REDACTED:{name}]", redacted)
    return RedactionResult(text=redacted, findings=findings)
