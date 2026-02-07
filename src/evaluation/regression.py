"""Regression gating: run evaluation suite and gate deployments."""

import json
import logging
import time
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from src.evaluation.citation_precision import (
    CitationPrecisionReport,
    evaluate_citation_precision,
)
from src.evaluation.golden_set import get_golden_set, list_golden_sets
from src.evaluation.retrieval_metrics import RetrievalEvalReport, evaluate_retrieval

logger = logging.getLogger(__name__)


class RegressionThresholds(BaseModel):
    """Thresholds that must be met for a deployment to pass."""

    retrieval_mrr: float = 0.4
    retrieval_hit_rate: float = 0.6
    retrieval_precision: float = 0.3
    citation_precision: float = 0.5
    faithfulness: float = 0.7        # evaluated separately (requires LLM)


class RegressionResult(BaseModel):
    """Result of a single domain regression run."""

    domain: str
    retrieval: Optional[RetrievalEvalReport] = None
    citation: Optional[CitationPrecisionReport] = None
    passed: bool = False


class RegressionReport(BaseModel):
    """Full regression gate report across all domains."""

    run_id: str
    timestamp: float = Field(default_factory=time.time)
    domains: list[RegressionResult] = Field(default_factory=list)
    overall_passed: bool = False
    thresholds: RegressionThresholds = Field(
        default_factory=RegressionThresholds
    )
    duration_seconds: float = 0.0


async def run_regression_gate(
    engagement_id: str,
    domains: Optional[list[str]] = None,
    thresholds: Optional[RegressionThresholds] = None,
    answers_by_domain: Optional[dict[str, list[dict]]] = None,
) -> RegressionReport:
    """Run full regression evaluation suite.

    Runs retrieval and citation evaluations for each domain golden set.
    All domains must pass for the gate to pass.

    Args:
        engagement_id: Engagement with indexed documents to test against.
        domains: Domains to evaluate. Defaults to all built-in sets.
        thresholds: Custom thresholds. Defaults to RegressionThresholds().
        answers_by_domain: Pre-generated answers per domain for citation eval.

    Returns:
        RegressionReport with pass/fail per domain and overall.
    """
    start = time.time()
    thresholds = thresholds or RegressionThresholds()
    domains = domains or list_golden_sets()
    answers_by_domain = answers_by_domain or {}

    run_id = f"regression_{int(time.time())}"
    domain_results: list[RegressionResult] = []

    for domain in domains:
        golden = get_golden_set(domain)
        if not golden:
            logger.warning("No golden set for domain: %s — skipping", domain)
            continue

        # Retrieval evaluation
        retrieval_report = await evaluate_retrieval(
            golden_set=golden,
            engagement_id=engagement_id,
            thresholds={
                "mean_mrr": thresholds.retrieval_mrr,
                "hit_rate": thresholds.retrieval_hit_rate,
                "mean_precision": thresholds.retrieval_precision,
            },
        )

        # Citation precision evaluation (if answers provided)
        citation_report = None
        domain_answers = answers_by_domain.get(domain, [])
        if domain_answers:
            citation_report = evaluate_citation_precision(
                golden_set=golden,
                answers_with_citations=domain_answers,
                threshold=thresholds.citation_precision,
            )

        # Domain passes if retrieval passes and citation passes (if evaluated)
        domain_passed = retrieval_report.passed
        if citation_report is not None:
            domain_passed = domain_passed and citation_report.passed

        domain_results.append(RegressionResult(
            domain=domain,
            retrieval=retrieval_report,
            citation=citation_report,
            passed=domain_passed,
        ))

    overall_passed = all(r.passed for r in domain_results) and len(domain_results) > 0
    duration = time.time() - start

    report = RegressionReport(
        run_id=run_id,
        domains=domain_results,
        overall_passed=overall_passed,
        thresholds=thresholds,
        duration_seconds=round(duration, 2),
    )

    logger.info(
        "Regression gate [%s]: %d domains, overall=%s, duration=%.1fs",
        run_id,
        len(domain_results),
        "PASS" if overall_passed else "FAIL",
        duration,
    )

    return report


def save_regression_report(report: RegressionReport, path: str | Path) -> None:
    """Save a regression report to JSON file for CI/CD integration."""
    with open(path, "w") as f:
        json.dump(report.model_dump(), f, indent=2, default=str)
    logger.info("Regression report saved to %s", path)
