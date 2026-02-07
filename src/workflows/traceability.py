"""Traceability matrix generation logic."""

import logging
from typing import Optional

from src.models.workflow import (
    ControlCoverage,
    ControlInput,
    ControlLink,
    CoverageGap,
    MappingInput,
    MatrixSummary,
    RequirementInput,
    RequirementMapping,
    TraceabilityMatrixResponse,
)
from src.retrieval.evidence import evidence_search

logger = logging.getLogger(__name__)


async def generate_traceability_matrix(
    engagement_id: str,
    requirements: list[RequirementInput],
    controls: list[ControlInput],
    mappings: list[MappingInput],
    framework: Optional[str] = None,
) -> TraceabilityMatrixResponse:
    """Generate a traceability matrix from requirements, controls, and mappings.

    For each requirement, determines which controls are mapped and how much
    evidence exists in the RAG store. Identifies coverage gaps.

    Args:
        engagement_id: Engagement scope.
        requirements: List of requirements.
        controls: List of controls.
        mappings: Requirement-to-control mappings.
        framework: Optional framework filter.

    Returns:
        TraceabilityMatrixResponse with matrix rows, summary, and gaps.
    """
    # Build lookup maps
    control_map = {c.id: c for c in controls}
    req_mappings: dict[str, list[MappingInput]] = {}
    for m in mappings:
        req_mappings.setdefault(m.requirement_id, []).append(m)

    # Filter requirements by framework if specified
    filtered_reqs = requirements
    if framework:
        filtered_reqs = [
            r for r in requirements
            if r.framework.lower() == framework.lower()
        ]

    matrix_rows: list[RequirementMapping] = []
    gaps: list[CoverageGap] = []

    for req in filtered_reqs:
        req_maps = req_mappings.get(req.id, [])

        if not req_maps:
            # Unmapped requirement = gap
            gaps.append(CoverageGap(
                requirement_id=req.id,
                requirement_ref=req.ref,
                requirement_title=req.title,
                framework=req.framework,
                gap_type="unmapped",
                description=(
                    f"Requirement {req.ref} ({req.title}) has no "
                    f"controls mapped to it."
                ),
            ))
            matrix_rows.append(RequirementMapping(
                requirement_id=req.id,
                requirement_ref=req.ref,
                requirement_title=req.title,
                framework=req.framework,
                controls=[],
                evidence_count=0,
                coverage=ControlCoverage.NONE,
            ))
            continue

        # Build control links with evidence counts
        control_links: list[ControlLink] = []
        total_evidence = 0

        for m in req_maps:
            ctrl = control_map.get(m.control_id)
            if not ctrl:
                continue

            # Search for evidence related to this control
            evidence_chunks, _ = await evidence_search(
                query=f"{ctrl.title} {req.title}",
                engagement_id=engagement_id,
                top_k=5,
                control_id=ctrl.ref,
            )

            ev_count = len(evidence_chunks)
            total_evidence += ev_count

            # Determine coverage per control
            if ev_count == 0:
                ctrl_coverage = ControlCoverage.UNTESTED
            elif m.coverage == "full" and ev_count >= 2:
                ctrl_coverage = ControlCoverage.FULL
            elif ev_count >= 1:
                ctrl_coverage = ControlCoverage.PARTIAL
            else:
                ctrl_coverage = ControlCoverage.NONE

            control_links.append(ControlLink(
                control_id=ctrl.id,
                control_ref=ctrl.ref,
                control_title=ctrl.title,
                evidence_count=ev_count,
                coverage=ctrl_coverage,
            ))

            if ev_count == 0:
                gaps.append(CoverageGap(
                    requirement_id=req.id,
                    requirement_ref=req.ref,
                    requirement_title=req.title,
                    framework=req.framework,
                    gap_type="no_evidence",
                    description=(
                        f"Control {ctrl.ref} ({ctrl.title}) mapped to "
                        f"{req.ref} has no supporting evidence."
                    ),
                ))

        # Determine overall requirement coverage
        if not control_links:
            req_coverage = ControlCoverage.NONE
        elif all(c.coverage == ControlCoverage.FULL for c in control_links):
            req_coverage = ControlCoverage.FULL
        elif any(
            c.coverage in (ControlCoverage.FULL, ControlCoverage.PARTIAL)
            for c in control_links
        ):
            req_coverage = ControlCoverage.PARTIAL
        else:
            req_coverage = ControlCoverage.UNTESTED

        matrix_rows.append(RequirementMapping(
            requirement_id=req.id,
            requirement_ref=req.ref,
            requirement_title=req.title,
            framework=req.framework,
            controls=control_links,
            evidence_count=total_evidence,
            coverage=req_coverage,
        ))

    # Build summary
    fully_covered = sum(
        1 for r in matrix_rows if r.coverage == ControlCoverage.FULL
    )
    partially_covered = sum(
        1 for r in matrix_rows if r.coverage == ControlCoverage.PARTIAL
    )
    not_covered = sum(
        1 for r in matrix_rows
        if r.coverage in (ControlCoverage.NONE, ControlCoverage.UNTESTED)
    )
    total = len(matrix_rows)
    coverage_pct = (
        (fully_covered + partially_covered * 0.5) / total * 100
        if total > 0
        else 0.0
    )

    summary = MatrixSummary(
        total_requirements=total,
        total_controls=len(controls),
        fully_covered=fully_covered,
        partially_covered=partially_covered,
        not_covered=not_covered,
        coverage_percentage=round(coverage_pct, 1),
    )

    logger.info(
        "Traceability matrix: %d requirements, %d controls, "
        "%.1f%% coverage, %d gaps",
        total,
        len(controls),
        coverage_pct,
        len(gaps),
    )

    return TraceabilityMatrixResponse(
        engagement_id=engagement_id,
        framework=framework,
        matrix=matrix_rows,
        summary=summary,
        gaps=gaps,
    )
