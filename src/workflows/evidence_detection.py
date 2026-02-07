"""Missing evidence detection logic."""

import logging

from src.models.workflow import (
    ControlEvidenceInput,
    EvidenceGap,
    MissingEvidenceResponse,
)
from src.retrieval.evidence import evidence_search

logger = logging.getLogger(__name__)


async def detect_missing_evidence(
    engagement_id: str,
    controls: list[ControlEvidenceInput],
    required_periods: list[str],
) -> MissingEvidenceResponse:
    """Detect missing, expired, or incomplete evidence for controls.

    For each control × period combination, searches the RAG store for
    supporting evidence. Reports gaps where evidence is absent.

    Args:
        engagement_id: Engagement scope.
        controls: Controls with expected evidence metadata.
        required_periods: Time periods that need coverage.

    Returns:
        MissingEvidenceResponse with gaps and completeness stats.
    """
    gaps: list[EvidenceGap] = []
    controls_with_gaps_set: set[str] = set()

    total_checks = 0
    passed_checks = 0

    for ctrl in controls:
        periods = required_periods or [None]

        # Determine expected periods based on frequency
        if not required_periods and ctrl.frequency:
            periods = _infer_periods(ctrl.frequency)

        for period in periods:
            total_checks += 1

            # Search for evidence matching this control + period
            evidence_chunks, _ = await evidence_search(
                query=f"{ctrl.control_ref} {ctrl.control_title}",
                engagement_id=engagement_id,
                top_k=5,
                control_id=ctrl.control_ref,
                period=period,
            )

            if not evidence_chunks:
                # No evidence found at all
                controls_with_gaps_set.add(ctrl.control_id)
                gaps.append(EvidenceGap(
                    control_id=ctrl.control_id,
                    control_ref=ctrl.control_ref,
                    control_title=ctrl.control_title,
                    gap_type="missing",
                    period=period,
                    description=(
                        f"No evidence found for control {ctrl.control_ref} "
                        f"({ctrl.control_title})"
                        + (f" for period {period}" if period else "")
                        + "."
                    ),
                    severity=_assess_severity(ctrl, "missing"),
                ))
                continue

            passed_checks += 1

            # Check if expected evidence types are covered
            if ctrl.expected_evidence_types:
                found_types = {
                    c.doc_type.lower()
                    for c in evidence_chunks
                    if c.doc_type
                }

                for expected_type in ctrl.expected_evidence_types:
                    if expected_type.lower() not in found_types:
                        controls_with_gaps_set.add(ctrl.control_id)
                        gaps.append(EvidenceGap(
                            control_id=ctrl.control_id,
                            control_ref=ctrl.control_ref,
                            control_title=ctrl.control_title,
                            gap_type="incomplete",
                            period=period,
                            expected_type=expected_type,
                            description=(
                                f"Expected evidence type '{expected_type}' "
                                f"not found for control "
                                f"{ctrl.control_ref}"
                                + (f" in period {period}" if period else "")
                                + "."
                            ),
                            severity=_assess_severity(
                                ctrl, "incomplete"
                            ),
                        ))

    completeness = (
        (passed_checks / total_checks * 100)
        if total_checks > 0
        else 0.0
    )

    summary = (
        f"Checked {len(controls)} controls across "
        f"{len(required_periods) or 1} period(s). "
        f"Found {len(gaps)} evidence gap(s) affecting "
        f"{len(controls_with_gaps_set)} control(s). "
        f"Completeness: {completeness:.1f}%."
    )

    logger.info(
        "Evidence detection: %d controls, %d gaps, %.1f%% complete",
        len(controls),
        len(gaps),
        completeness,
    )

    return MissingEvidenceResponse(
        engagement_id=engagement_id,
        total_controls=len(controls),
        controls_with_gaps=len(controls_with_gaps_set),
        gaps=gaps,
        summary=summary,
        completeness_percentage=round(completeness, 1),
    )


def _assess_severity(
    ctrl: ControlEvidenceInput, gap_type: str
) -> str:
    """Assess gap severity based on control frequency and gap type."""
    if gap_type == "missing":
        # Missing evidence is more severe for frequent controls
        if ctrl.frequency in ("daily", "continuous"):
            return "high"
        if ctrl.frequency in ("weekly", "monthly"):
            return "high"
        if ctrl.frequency in ("quarterly",):
            return "medium"
        return "medium"
    # Incomplete evidence
    return "low"


def _infer_periods(frequency: str) -> list[str | None]:
    """Infer expected periods from control frequency.

    Returns a list of generic period labels when no explicit
    periods are provided.
    """
    freq = frequency.lower()
    if freq in ("quarterly",):
        return ["Q1", "Q2", "Q3", "Q4"]
    if freq in ("monthly",):
        return [f"M{i}" for i in range(1, 13)]
    if freq in ("annual", "yearly"):
        return ["Annual"]
    # For daily/weekly/continuous, just check once
    return [None]
