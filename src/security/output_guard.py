"""Output guardrail validation (ASI01 + ASI06).

Post-generation checks to prevent leaking secrets, cross-engagement
data, or low-confidence responses from reaching the user.
"""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Patterns that should never appear in LLM output
_SECRET_PATTERNS = [
    re.compile(r"(?i)\b(api[_-]?key|secret[_-]?key|password)\s*[:=]\s*\S+"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9\-._~+/]+=*"),
    re.compile(r"\b[A-Za-z0-9]{20,}(?:_[A-Za-z0-9]{20,})\b"),  # AWS-style keys
    re.compile(r"(?i)\b(sk-|pk_live_|pk_test_)[A-Za-z0-9]+"),  # OpenAI/Stripe
    re.compile(
        r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----"
    ),
]


@dataclass
class GuardResult:
    """Result of output guardrail checks."""

    passed: bool = True
    blocked_reason: str = ""
    warnings: list[str] | None = None


def check_output(
    response_text: str,
    confidence: float = 1.0,
    min_confidence: float = 0.2,
    engagement_id: str = "",
    chunk_engagement_ids: list[str] | None = None,
) -> GuardResult:
    """Validate LLM output before returning to user.

    Checks:
    1. No secrets/credentials in output
    2. Confidence above minimum threshold
    3. No cross-engagement data leakage

    Args:
        response_text: The generated response text.
        confidence: Confidence score from the LLM.
        min_confidence: Minimum acceptable confidence.
        engagement_id: The user's engagement context.
        chunk_engagement_ids: Engagement IDs of source chunks.

    Returns:
        GuardResult with pass/fail and reason.
    """
    warnings: list[str] = []

    # 1. Check for secrets in output
    for pattern in _SECRET_PATTERNS:
        if pattern.search(response_text):
            logger.warning(
                "Output blocked: potential secret/credential detected"
            )
            return GuardResult(
                passed=False,
                blocked_reason="Response may contain credentials or secrets",
                warnings=warnings,
            )

    # 2. Check confidence threshold
    if confidence < min_confidence:
        logger.info(
            "Output blocked: confidence %.2f below threshold %.2f",
            confidence,
            min_confidence,
        )
        return GuardResult(
            passed=False,
            blocked_reason=(
                f"Response confidence ({confidence:.2f}) is below "
                f"minimum threshold ({min_confidence:.2f})"
            ),
            warnings=warnings,
        )

    # 3. Check cross-engagement leakage
    if (
        engagement_id
        and chunk_engagement_ids
        and any(eid != engagement_id for eid in chunk_engagement_ids)
    ):
        logger.warning(
            "Output blocked: cross-engagement data detected "
            "(requested=%s, found=%s)",
            engagement_id,
            chunk_engagement_ids,
        )
        return GuardResult(
            passed=False,
            blocked_reason="Cross-engagement data leakage detected",
            warnings=warnings,
        )

    # 4. Warn on low-ish confidence
    if confidence < 0.4:
        warnings.append(
            f"Low confidence ({confidence:.2f}) — review recommended"
        )

    return GuardResult(passed=True, warnings=warnings or None)
