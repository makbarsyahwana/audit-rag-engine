"""Golden question sets for RAG evaluation per audit domain."""

import json
import logging
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class GoldenQuestion(BaseModel):
    """A single golden question with expected answer and metadata."""

    id: str
    domain: str                                # e.g. "iso27001", "soc2", "general"
    question: str
    expected_answer: str
    expected_chunk_ids: list[str] = Field(default_factory=list)
    expected_document_ids: list[str] = Field(default_factory=list)
    expected_entities: list[str] = Field(default_factory=list)
    difficulty: str = "medium"                 # easy, medium, hard
    tags: list[str] = Field(default_factory=list)


class GoldenSet(BaseModel):
    """A collection of golden questions for a specific domain."""

    name: str
    domain: str
    description: str = ""
    questions: list[GoldenQuestion] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Built-in golden sets per domain
# ---------------------------------------------------------------------------

ISO27001_GOLDEN_SET = GoldenSet(
    name="ISO 27001 Golden Set",
    domain="iso27001",
    description="Evaluation questions for ISO 27001 information security audits",
    questions=[
        GoldenQuestion(
            id="iso27001_001",
            domain="iso27001",
            question="What is the organization's policy for access control "
                     "as defined in the ISMS?",
            expected_answer="The access control policy should define rules "
                           "for granting, reviewing, and revoking access.",
            tags=["access_control", "A.9"],
            difficulty="easy",
        ),
        GoldenQuestion(
            id="iso27001_002",
            domain="iso27001",
            question="How does the organization manage changes to "
                     "information systems?",
            expected_answer="Change management should include documented "
                           "approval, testing, and rollback procedures.",
            tags=["change_management", "A.12"],
            difficulty="medium",
        ),
        GoldenQuestion(
            id="iso27001_003",
            domain="iso27001",
            question="What controls are in place for cryptographic key "
                     "management?",
            expected_answer="Key management should cover key generation, "
                           "distribution, storage, rotation, and destruction.",
            tags=["cryptography", "A.10"],
            difficulty="medium",
        ),
        GoldenQuestion(
            id="iso27001_004",
            domain="iso27001",
            question="How are information security incidents detected, "
                     "reported, and managed?",
            expected_answer="Incident management should include detection, "
                           "classification, response, and lessons learned.",
            tags=["incident_management", "A.16"],
            difficulty="hard",
        ),
        GoldenQuestion(
            id="iso27001_005",
            domain="iso27001",
            question="What is the organization's approach to risk "
                     "assessment and risk treatment?",
            expected_answer="Risk assessment should follow a defined "
                           "methodology with likelihood/impact scoring.",
            tags=["risk_assessment", "clause_6.1"],
            difficulty="hard",
        ),
    ],
)

SOC2_GOLDEN_SET = GoldenSet(
    name="SOC 2 Type II Golden Set",
    domain="soc2",
    description="Evaluation questions for SOC 2 Type II audits",
    questions=[
        GoldenQuestion(
            id="soc2_001",
            domain="soc2",
            question="How does the organization ensure logical access "
                     "controls are operating effectively?",
            expected_answer="Logical access controls include user "
                           "provisioning, MFA, periodic access reviews.",
            tags=["logical_access", "CC6.1"],
            difficulty="easy",
        ),
        GoldenQuestion(
            id="soc2_002",
            domain="soc2",
            question="What monitoring controls exist for detecting "
                     "unauthorized activities?",
            expected_answer="Monitoring controls include SIEM, log "
                           "aggregation, alerting, and periodic review.",
            tags=["monitoring", "CC7.2"],
            difficulty="medium",
        ),
        GoldenQuestion(
            id="soc2_003",
            domain="soc2",
            question="How does the organization manage vendor risk?",
            expected_answer="Vendor risk management includes due diligence, "
                           "contractual requirements, and periodic assessments.",
            tags=["vendor_management", "CC9.2"],
            difficulty="medium",
        ),
        GoldenQuestion(
            id="soc2_004",
            domain="soc2",
            question="What is the organization's business continuity "
                     "and disaster recovery strategy?",
            expected_answer="BC/DR should include RTOs, RPOs, testing, "
                           "and documented recovery procedures.",
            tags=["bcdr", "A1.2"],
            difficulty="hard",
        ),
    ],
)

GENERAL_GOLDEN_SET = GoldenSet(
    name="General Audit Golden Set",
    domain="general",
    description="Domain-agnostic evaluation questions for audit RAG",
    questions=[
        GoldenQuestion(
            id="general_001",
            domain="general",
            question="What evidence exists for this control?",
            expected_answer="Should return relevant evidence documents.",
            tags=["evidence_retrieval"],
            difficulty="easy",
        ),
        GoldenQuestion(
            id="general_002",
            domain="general",
            question="What is the current status of the control testing?",
            expected_answer="Should describe testing status per control.",
            tags=["control_testing"],
            difficulty="medium",
        ),
        GoldenQuestion(
            id="general_003",
            domain="general",
            question="Are there any gaps in the evidence for this period?",
            expected_answer="Should identify missing evidence per period.",
            tags=["evidence_gaps"],
            difficulty="hard",
        ),
    ],
)

BUILT_IN_SETS: dict[str, GoldenSet] = {
    "iso27001": ISO27001_GOLDEN_SET,
    "soc2": SOC2_GOLDEN_SET,
    "general": GENERAL_GOLDEN_SET,
}


def get_golden_set(domain: str) -> Optional[GoldenSet]:
    """Get a built-in golden set by domain name."""
    return BUILT_IN_SETS.get(domain.lower())


def list_golden_sets() -> list[str]:
    """List available golden set domains."""
    return list(BUILT_IN_SETS.keys())


def load_golden_set_from_file(path: str | Path) -> GoldenSet:
    """Load a golden set from a JSON file.

    Args:
        path: Path to JSON file containing a GoldenSet.

    Returns:
        Parsed GoldenSet.
    """
    with open(path) as f:
        data = json.load(f)
    return GoldenSet(**data)


def save_golden_set_to_file(golden_set: GoldenSet, path: str | Path) -> None:
    """Save a golden set to a JSON file."""
    with open(path, "w") as f:
        json.dump(golden_set.model_dump(), f, indent=2)
    logger.info("Golden set saved to %s", path)
