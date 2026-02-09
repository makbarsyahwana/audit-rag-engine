"""File upload validation (ASI05 — Unexpected Code Execution).

Validates uploaded files by magic bytes, size limits, and MIME type
allowlist before they enter the ingestion pipeline.
"""

import logging
from dataclasses import dataclass

from src.config import settings

logger = logging.getLogger(__name__)

# Magic byte signatures for common document formats
_MAGIC_SIGNATURES: dict[str, list[bytes]] = {
    "application/pdf": [b"%PDF"],
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [
        b"PK\x03\x04",
    ],
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [
        b"PK\x03\x04",
    ],
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": [
        b"PK\x03\x04",
    ],
    "application/json": [],  # Text-based, no magic bytes
    "text/plain": [],
    "text/csv": [],
    "text/markdown": [],
}


@dataclass
class ValidationResult:
    """Result of file validation."""

    valid: bool = True
    reason: str = ""


def get_allowed_mime_types() -> set[str]:
    """Parse allowed MIME types from config."""
    return {
        m.strip()
        for m in settings.allowed_mime_types.split(",")
        if m.strip()
    }


def validate_file(
    file_data: bytes,
    filename: str,
    content_type: str,
) -> ValidationResult:
    """Validate an uploaded file.

    Checks:
    1. File size within limits
    2. MIME type in allowlist
    3. Magic bytes match declared content type (where applicable)

    Args:
        file_data: Raw file bytes.
        filename: Original filename.
        content_type: Declared MIME type.

    Returns:
        ValidationResult with valid flag and reason.
    """
    max_bytes = settings.max_upload_size_mb * 1024 * 1024

    # 1. Size check
    if len(file_data) > max_bytes:
        size_mb = len(file_data) / (1024 * 1024)
        logger.warning(
            "File rejected (too large): %s (%.1f MB > %d MB)",
            filename,
            size_mb,
            settings.max_upload_size_mb,
        )
        return ValidationResult(
            valid=False,
            reason=(
                f"File size ({size_mb:.1f} MB) exceeds limit "
                f"({settings.max_upload_size_mb} MB)"
            ),
        )

    # 2. MIME type allowlist
    allowed = get_allowed_mime_types()
    if content_type not in allowed:
        logger.warning(
            "File rejected (disallowed type): %s (%s)",
            filename,
            content_type,
        )
        return ValidationResult(
            valid=False,
            reason=f"File type '{content_type}' is not allowed",
        )

    # 3. Magic byte validation (for binary formats)
    expected_sigs = _MAGIC_SIGNATURES.get(content_type, [])
    if expected_sigs and file_data:
        matched = any(
            file_data[:len(sig)] == sig for sig in expected_sigs
        )
        if not matched:
            logger.warning(
                "File rejected (magic bytes mismatch): %s "
                "declared=%s",
                filename,
                content_type,
            )
            return ValidationResult(
                valid=False,
                reason=(
                    f"File content does not match declared type "
                    f"'{content_type}'"
                ),
            )

    # 4. Empty file check
    if len(file_data) == 0:
        return ValidationResult(
            valid=False,
            reason="Empty file",
        )

    return ValidationResult(valid=True)
