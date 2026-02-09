"""Prompt injection detection (ASI01 — Agent Goal Hijack).

Scans user queries for common prompt injection patterns before
they reach the retrieval or generation pipelines.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ThreatLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ScanResult:
    """Result of a prompt injection scan."""

    threat_level: ThreatLevel = ThreatLevel.NONE
    matched_patterns: list[str] | None = None
    blocked: bool = False
    message: str = ""


# --- Injection pattern categories ---

# Direct instruction override attempts
_OVERRIDE_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?above\s+instructions",
    r"disregard\s+(all\s+)?previous",
    r"forget\s+(all\s+)?prior\s+instructions",
    r"override\s+system\s+prompt",
    r"new\s+system\s+prompt",
    r"you\s+are\s+now\s+a",
    r"act\s+as\s+(if\s+you\s+are\s+)?a",
    r"pretend\s+(you\s+are|to\s+be)",
    r"from\s+now\s+on\s+you",
    r"switch\s+to\s+.*?mode",
]

# Data exfiltration attempts
_EXFIL_PATTERNS = [
    r"send\s+(the\s+)?(data|results?|content|information)\s+to",
    r"forward\s+(all|this|the)\s+.*?\s+to",
    r"(http|https|ftp)://[^\s]+",  # URLs in queries (potential exfil targets)
    r"email\s+(this|the|all)\s+.*?\s+to",
    r"upload\s+(this|the|all)\s+.*?\s+to",
    r"base64\s+encode",
]

# System probing / meta-prompting
_PROBE_PATTERNS = [
    r"what\s+is\s+your\s+system\s+prompt",
    r"show\s+(me\s+)?your\s+(system\s+)?instructions",
    r"repeat\s+(your\s+)?(system\s+)?prompt",
    r"print\s+(your\s+)?(system\s+)?prompt",
    r"reveal\s+(your\s+)?instructions",
    r"what\s+are\s+your\s+rules",
    r"display\s+(your\s+)?configuration",
]

# Delimiter / encoding tricks
_ENCODING_PATTERNS = [
    r"```\s*system",
    r"\[SYSTEM\]",
    r"<\|im_start\|>system",
    r"<system>",
    r"###\s*instruction",
    r"\bDAN\b",  # "Do Anything Now" jailbreak
    r"jailbreak",
]


def _compile_patterns(
    patterns: list[str],
) -> list[re.Pattern[str]]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


_COMPILED_OVERRIDE = _compile_patterns(_OVERRIDE_PATTERNS)
_COMPILED_EXFIL = _compile_patterns(_EXFIL_PATTERNS)
_COMPILED_PROBE = _compile_patterns(_PROBE_PATTERNS)
_COMPILED_ENCODING = _compile_patterns(_ENCODING_PATTERNS)


def scan_query(query: str) -> ScanResult:
    """Scan a user query for prompt injection patterns.

    Returns a ScanResult with threat level, matched patterns, and whether
    the query should be blocked.
    """
    if not query or not query.strip():
        return ScanResult()

    matched: list[str] = []
    threat = ThreatLevel.NONE

    # Check override patterns (HIGH)
    for pattern in _COMPILED_OVERRIDE:
        if pattern.search(query):
            matched.append(f"override:{pattern.pattern}")
            threat = ThreatLevel.HIGH

    # Check exfiltration patterns (HIGH)
    for pattern in _COMPILED_EXFIL:
        if pattern.search(query):
            matched.append(f"exfil:{pattern.pattern}")
            if threat != ThreatLevel.HIGH:
                threat = ThreatLevel.HIGH

    # Check probe patterns (MEDIUM)
    for pattern in _COMPILED_PROBE:
        if pattern.search(query):
            matched.append(f"probe:{pattern.pattern}")
            if threat == ThreatLevel.NONE:
                threat = ThreatLevel.MEDIUM

    # Check encoding tricks (HIGH)
    for pattern in _COMPILED_ENCODING:
        if pattern.search(query):
            matched.append(f"encoding:{pattern.pattern}")
            threat = ThreatLevel.HIGH

    if not matched:
        return ScanResult()

    blocked = threat == ThreatLevel.HIGH
    message = (
        f"Prompt injection detected ({threat.value}): "
        f"{len(matched)} pattern(s) matched"
    )

    if blocked:
        logger.warning(
            "BLOCKED prompt injection: user_query='%s' patterns=%s",
            query[:200],
            matched,
        )
    else:
        logger.info(
            "Prompt injection signal (%s): query='%s' patterns=%s",
            threat.value,
            query[:200],
            matched,
        )

    return ScanResult(
        threat_level=threat,
        matched_patterns=matched,
        blocked=blocked,
        message=message,
    )
