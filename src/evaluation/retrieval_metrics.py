"""Offline evaluation pipeline for retrieval quality metrics."""

import logging
import time
from typing import Optional

from pydantic import BaseModel, Field

from src.evaluation.golden_set import GoldenSet
from src.retrieval.hybrid import hybrid_search

logger = logging.getLogger(__name__)


class RetrievalResult(BaseModel):
    """Result of a single retrieval evaluation."""

    question_id: str
    question: str
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    expected_chunk_ids: list[str] = Field(default_factory=list)
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    f1_at_k: float = 0.0
    mrr: float = 0.0                # Mean Reciprocal Rank
    ndcg: float = 0.0               # Normalized Discounted Cumulative Gain
    hit: bool = False                # At least one relevant chunk retrieved
    latency_ms: float = 0.0


class RetrievalEvalReport(BaseModel):
    """Aggregate report from running a golden set through retrieval."""

    golden_set_name: str
    domain: str
    total_questions: int = 0
    mean_precision: float = 0.0
    mean_recall: float = 0.0
    mean_f1: float = 0.0
    mean_mrr: float = 0.0
    mean_ndcg: float = 0.0
    hit_rate: float = 0.0
    mean_latency_ms: float = 0.0
    results: list[RetrievalResult] = Field(default_factory=list)
    passed: bool = False             # meets minimum thresholds
    thresholds: dict[str, float] = Field(default_factory=dict)


async def evaluate_retrieval(
    golden_set: GoldenSet,
    engagement_id: str,
    top_k: int = 10,
    thresholds: Optional[dict[str, float]] = None,
) -> RetrievalEvalReport:
    """Run retrieval evaluation against a golden question set.

    Args:
        golden_set: The golden set to evaluate against.
        engagement_id: Engagement scope for retrieval.
        top_k: Number of chunks to retrieve per question.
        thresholds: Minimum thresholds for pass/fail (e.g. {"mean_mrr": 0.5}).

    Returns:
        RetrievalEvalReport with per-question and aggregate metrics.
    """
    if thresholds is None:
        thresholds = {
            "mean_mrr": 0.4,
            "hit_rate": 0.6,
            "mean_precision": 0.3,
        }

    results: list[RetrievalResult] = []

    for q in golden_set.questions:
        if not q.expected_chunk_ids:
            # Skip questions without expected chunks (can't measure)
            continue

        start = time.time()
        chunks, _ = await hybrid_search(
            query=q.question,
            engagement_id=engagement_id,
            top_k=top_k,
        )
        latency_ms = (time.time() - start) * 1000

        retrieved_ids = [c.chunk_id for c in chunks]
        expected_set = set(q.expected_chunk_ids)

        # Precision@K
        relevant_retrieved = [
            cid for cid in retrieved_ids if cid in expected_set
        ]
        precision = (
            len(relevant_retrieved) / len(retrieved_ids)
            if retrieved_ids else 0.0
        )

        # Recall@K
        recall = (
            len(relevant_retrieved) / len(expected_set)
            if expected_set else 0.0
        )

        # F1@K
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        # MRR (reciprocal rank of first relevant result)
        mrr = 0.0
        for i, cid in enumerate(retrieved_ids, 1):
            if cid in expected_set:
                mrr = 1.0 / i
                break

        # NDCG (simplified: binary relevance)
        ndcg = _compute_ndcg(retrieved_ids, expected_set, top_k)

        # Hit (at least one relevant)
        hit = len(relevant_retrieved) > 0

        results.append(RetrievalResult(
            question_id=q.id,
            question=q.question,
            retrieved_chunk_ids=retrieved_ids,
            expected_chunk_ids=q.expected_chunk_ids,
            precision_at_k=round(precision, 4),
            recall_at_k=round(recall, 4),
            f1_at_k=round(f1, 4),
            mrr=round(mrr, 4),
            ndcg=round(ndcg, 4),
            hit=hit,
            latency_ms=round(latency_ms, 1),
        ))

    # Aggregate
    n = len(results) or 1
    mean_precision = sum(r.precision_at_k for r in results) / n
    mean_recall = sum(r.recall_at_k for r in results) / n
    mean_f1 = sum(r.f1_at_k for r in results) / n
    mean_mrr = sum(r.mrr for r in results) / n
    mean_ndcg = sum(r.ndcg for r in results) / n
    hit_rate = sum(1 for r in results if r.hit) / n
    mean_latency = sum(r.latency_ms for r in results) / n

    # Check pass/fail
    passed = all(
        locals().get(f"mean_{k}", getattr(locals(), k, 0)) >= v
        for k, v in thresholds.items()
    )
    # More explicit pass check
    metric_values = {
        "mean_mrr": mean_mrr,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "mean_f1": mean_f1,
        "hit_rate": hit_rate,
        "mean_ndcg": mean_ndcg,
    }
    passed = all(
        metric_values.get(k, 0.0) >= v for k, v in thresholds.items()
    )

    report = RetrievalEvalReport(
        golden_set_name=golden_set.name,
        domain=golden_set.domain,
        total_questions=len(results),
        mean_precision=round(mean_precision, 4),
        mean_recall=round(mean_recall, 4),
        mean_f1=round(mean_f1, 4),
        mean_mrr=round(mean_mrr, 4),
        mean_ndcg=round(mean_ndcg, 4),
        hit_rate=round(hit_rate, 4),
        mean_latency_ms=round(mean_latency, 1),
        results=results,
        passed=passed,
        thresholds=thresholds,
    )

    logger.info(
        "Retrieval eval [%s]: MRR=%.3f, Hit=%.1f%%, P@K=%.3f, R@K=%.3f, "
        "passed=%s",
        golden_set.domain,
        mean_mrr,
        hit_rate * 100,
        mean_precision,
        mean_recall,
        passed,
    )

    return report


def _compute_ndcg(
    retrieved: list[str], relevant: set[str], k: int
) -> float:
    """Compute NDCG with binary relevance."""
    import math

    dcg = 0.0
    for i, cid in enumerate(retrieved[:k], 1):
        if cid in relevant:
            dcg += 1.0 / math.log2(i + 1)

    # Ideal DCG
    ideal_relevant = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_relevant + 1))

    return dcg / idcg if idcg > 0 else 0.0
