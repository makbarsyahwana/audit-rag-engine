"""Citation precision evaluation: checks if cited chunks are actually relevant."""

import logging

from pydantic import BaseModel, Field

from src.evaluation.golden_set import GoldenSet

logger = logging.getLogger(__name__)


class CitationResult(BaseModel):
    """Result of citation precision evaluation for a single answer."""

    question_id: str
    question: str
    cited_chunk_ids: list[str] = Field(default_factory=list)
    expected_chunk_ids: list[str] = Field(default_factory=list)
    true_positives: int = 0          # cited AND relevant
    false_positives: int = 0         # cited but NOT relevant
    false_negatives: int = 0         # relevant but NOT cited
    precision: float = 0.0           # TP / (TP + FP)
    recall: float = 0.0              # TP / (TP + FN)
    f1: float = 0.0


class CitationPrecisionReport(BaseModel):
    """Aggregate citation precision report."""

    golden_set_name: str
    domain: str
    total_evaluated: int = 0
    mean_precision: float = 0.0
    mean_recall: float = 0.0
    mean_f1: float = 0.0
    perfect_citation_count: int = 0  # precision = 1.0
    no_citation_count: int = 0       # zero citations produced
    results: list[CitationResult] = Field(default_factory=list)
    passed: bool = False
    threshold: float = 0.5


def evaluate_citation_precision(
    golden_set: GoldenSet,
    answers_with_citations: list[dict],
    threshold: float = 0.5,
) -> CitationPrecisionReport:
    """Evaluate citation precision against golden set expected chunks.

    Args:
        golden_set: Golden set with expected_chunk_ids per question.
        answers_with_citations: List of dicts with "citations" key
            containing list of dicts with "chunk_id".
        threshold: Minimum mean precision to pass.

    Returns:
        CitationPrecisionReport.
    """
    results: list[CitationResult] = []

    for i, question in enumerate(golden_set.questions):
        if i >= len(answers_with_citations):
            break

        if not question.expected_chunk_ids:
            continue

        answer_data = answers_with_citations[i]
        citations = answer_data.get("citations", [])
        cited_ids = [
            c.get("chunk_id", "") for c in citations if isinstance(c, dict)
        ]

        expected_set = set(question.expected_chunk_ids)
        cited_set = set(cited_ids)

        tp = len(cited_set & expected_set)
        fp = len(cited_set - expected_set)
        fn = len(expected_set - cited_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        results.append(CitationResult(
            question_id=question.id,
            question=question.question,
            cited_chunk_ids=cited_ids,
            expected_chunk_ids=question.expected_chunk_ids,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
        ))

    # Aggregate
    n = len(results) or 1
    mean_prec = sum(r.precision for r in results) / n
    mean_rec = sum(r.recall for r in results) / n
    mean_f1 = sum(r.f1 for r in results) / n
    perfect = sum(1 for r in results if r.precision >= 1.0 and r.cited_chunk_ids)
    no_cite = sum(1 for r in results if not r.cited_chunk_ids)

    report = CitationPrecisionReport(
        golden_set_name=golden_set.name,
        domain=golden_set.domain,
        total_evaluated=len(results),
        mean_precision=round(mean_prec, 4),
        mean_recall=round(mean_rec, 4),
        mean_f1=round(mean_f1, 4),
        perfect_citation_count=perfect,
        no_citation_count=no_cite,
        results=results,
        passed=mean_prec >= threshold,
        threshold=threshold,
    )

    logger.info(
        "Citation precision eval [%s]: P=%.3f, R=%.3f, F1=%.3f, "
        "perfect=%d, no_cite=%d, passed=%s",
        golden_set.domain, mean_prec, mean_rec, mean_f1,
        perfect, no_cite, report.passed,
    )

    return report
