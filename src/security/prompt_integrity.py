"""Prompt template versioning and integrity checking (ASI04 — Supply Chain).

Hashes all prompt templates at startup and exposes them via health endpoint.
Alerts if runtime hash differs from expected (drift detection).
"""

import hashlib
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TemplateFingerprint:
    """Hash fingerprint of a prompt template."""

    name: str
    sha256: str
    length: int


def hash_template(name: str, content: str) -> TemplateFingerprint:
    """Compute SHA-256 hash of a prompt template."""
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return TemplateFingerprint(name=name, sha256=digest, length=len(content))


class PromptIntegrityRegistry:
    """Registry of prompt template fingerprints."""

    def __init__(self) -> None:
        self._fingerprints: dict[str, TemplateFingerprint] = {}
        self._expected: dict[str, str] = {}  # name → expected sha256

    def register(self, name: str, content: str) -> TemplateFingerprint:
        """Register a prompt template and compute its fingerprint."""
        fp = hash_template(name, content)
        self._fingerprints[name] = fp
        logger.debug(
            "Registered prompt template: %s (sha256=%s)",
            name,
            fp.sha256[:16],
        )
        return fp

    def set_expected(self, name: str, sha256: str) -> None:
        """Set the expected hash for drift detection."""
        self._expected[name] = sha256

    def check_integrity(self) -> list[dict]:
        """Check all registered templates for drift.

        Returns list of drifted templates (empty = all OK).
        """
        drifted = []
        for name, fp in self._fingerprints.items():
            expected = self._expected.get(name)
            if expected and expected != fp.sha256:
                drifted.append({
                    "name": name,
                    "expected": expected,
                    "actual": fp.sha256,
                })
                logger.warning(
                    "Prompt template drift detected: %s "
                    "(expected=%s, actual=%s)",
                    name,
                    expected[:16],
                    fp.sha256[:16],
                )
        return drifted

    def get_fingerprints(self) -> dict[str, dict]:
        """Get all template fingerprints for health endpoint."""
        return {
            name: {"sha256": fp.sha256, "length": fp.length}
            for name, fp in self._fingerprints.items()
        }


prompt_integrity = PromptIntegrityRegistry()
