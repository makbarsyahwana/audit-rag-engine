"""Document content sanitization (ASI06 — Memory & Context Poisoning).

Strips or neutralizes instruction-like patterns from ingested document text
before it enters the embedding pipeline, preventing RAG poisoning attacks.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Patterns that look like LLM instructions embedded in documents
_INSTRUCTION_PATTERNS = [
    # Direct instruction injection
    (
        r"(?i)\bignore\s+(all\s+)?previous\s+instructions\b",
        "[SANITIZED:instruction_override]",
    ),
    (
        r"(?i)\bdisregard\s+(all\s+)?prior\s+(instructions|context)\b",
        "[SANITIZED:instruction_override]",
    ),
    (r"(?i)\bnew\s+system\s+prompt\b", "[SANITIZED:prompt_override]"),
    (r"(?i)\byou\s+are\s+now\s+a\b", "[SANITIZED:role_override]"),
    (r"(?i)\bact\s+as\s+if\s+you\s+are\b", "[SANITIZED:role_override]"),
    (r"(?i)\bpretend\s+(you\s+are|to\s+be)\b", "[SANITIZED:role_override]"),
    # System prompt delimiters (should not appear in documents)
    (r"<\|im_start\|>system", "[SANITIZED:system_delimiter]"),
    (r"<\|im_end\|>", "[SANITIZED:system_delimiter]"),
    (r"\[SYSTEM\]", "[SANITIZED:system_delimiter]"),
    (r"<system>.*?</system>", "[SANITIZED:system_block]"),
    (r"###\s*(?:SYSTEM|INSTRUCTION|PROMPT)", "[SANITIZED:instruction_header]"),
    # Exfiltration instructions
    (r"(?i)\bsend\s+(the\s+)?(data|results?)\s+to\s+https?://", "[SANITIZED:exfil_url]"),
    (r"(?i)\bforward\s+.*?\s+to\s+https?://", "[SANITIZED:exfil_url]"),
]

_COMPILED_PATTERNS = [
    (re.compile(pattern), replacement)
    for pattern, replacement in _INSTRUCTION_PATTERNS
]


def sanitize_text(text: str, source_id: str = "") -> tuple[str, int]:
    """Sanitize document text by neutralizing instruction-like patterns.

    Args:
        text: Raw text extracted from a document.
        source_id: Document or chunk ID for logging.

    Returns:
        Tuple of (sanitized_text, number_of_replacements).
    """
    if not text:
        return text, 0

    total_replacements = 0
    sanitized = text

    for pattern, replacement in _COMPILED_PATTERNS:
        sanitized, count = pattern.subn(replacement, sanitized)
        total_replacements += count

    if total_replacements > 0:
        logger.warning(
            "Sanitized %d injection pattern(s) from document %s",
            total_replacements,
            source_id,
        )

    return sanitized, total_replacements
