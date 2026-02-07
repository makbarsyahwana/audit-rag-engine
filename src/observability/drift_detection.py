"""Goal drift detection: monitors if RAG outputs shift over time."""

import logging
import time
from collections import deque
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DriftSnapshot(BaseModel):
    """A snapshot of key quality metrics at a point in time."""

    timestamp: float = Field(default_factory=time.time)
    mean_confidence: float = 0.0
    abstention_rate: float = 0.0
    mean_chunks_retrieved: float = 0.0
    mean_citations_per_answer: float = 0.0
    mean_latency_ms: float = 0.0
    sample_count: int = 0


class DriftAlert(BaseModel):
    """An alert when drift exceeds thresholds."""

    metric: str
    baseline_value: float
    current_value: float
    deviation_pct: float
    threshold_pct: float
    timestamp: float = Field(default_factory=time.time)
    message: str = ""


class DriftDetector:
    """Monitors query/answer metrics for drift from baseline.

    Collects recent observations in a sliding window and compares
    against a baseline snapshot. Fires alerts when deviation exceeds
    configured thresholds.
    """

    def __init__(
        self,
        window_size: int = 100,
        deviation_threshold_pct: float = 20.0,
    ) -> None:
        self._window_size = window_size
        self._threshold_pct = deviation_threshold_pct
        self._baseline: Optional[DriftSnapshot] = None
        self._observations: deque[dict] = deque(maxlen=window_size)
        self._alerts: list[DriftAlert] = []

    def set_baseline(self, baseline: DriftSnapshot) -> None:
        """Set the baseline snapshot to compare against."""
        self._baseline = baseline
        logger.info(
            "Drift baseline set: confidence=%.3f, abstention=%.3f",
            baseline.mean_confidence,
            baseline.abstention_rate,
        )

    def record_observation(
        self,
        confidence: float,
        abstained: bool,
        chunks_retrieved: int,
        citations_count: int,
        latency_ms: float,
    ) -> None:
        """Record a single query/answer observation."""
        self._observations.append({
            "confidence": confidence,
            "abstained": 1.0 if abstained else 0.0,
            "chunks_retrieved": float(chunks_retrieved),
            "citations_count": float(citations_count),
            "latency_ms": latency_ms,
            "timestamp": time.time(),
        })

    def check_drift(self) -> list[DriftAlert]:
        """Check current window against baseline for drift.

        Returns:
            List of new drift alerts (empty if no drift detected).
        """
        if self._baseline is None or len(self._observations) < 10:
            return []

        current = self._compute_current_snapshot()
        new_alerts: list[DriftAlert] = []

        checks = [
            ("mean_confidence", self._baseline.mean_confidence, current.mean_confidence),
            ("abstention_rate", self._baseline.abstention_rate, current.abstention_rate),
            (
                "mean_chunks_retrieved",
                self._baseline.mean_chunks_retrieved,
                current.mean_chunks_retrieved,
            ),
            (
                "mean_citations_per_answer",
                self._baseline.mean_citations_per_answer,
                current.mean_citations_per_answer,
            ),
            ("mean_latency_ms", self._baseline.mean_latency_ms, current.mean_latency_ms),
        ]

        for metric, baseline_val, current_val in checks:
            if baseline_val == 0:
                continue

            deviation_pct = abs(current_val - baseline_val) / baseline_val * 100

            if deviation_pct > self._threshold_pct:
                alert = DriftAlert(
                    metric=metric,
                    baseline_value=round(baseline_val, 4),
                    current_value=round(current_val, 4),
                    deviation_pct=round(deviation_pct, 1),
                    threshold_pct=self._threshold_pct,
                    message=(
                        f"Drift detected in {metric}: "
                        f"baseline={baseline_val:.4f}, "
                        f"current={current_val:.4f} "
                        f"({deviation_pct:.1f}% deviation, "
                        f"threshold={self._threshold_pct}%)"
                    ),
                )
                new_alerts.append(alert)
                self._alerts.append(alert)
                logger.warning("DRIFT: %s", alert.message)

        return new_alerts

    def get_current_snapshot(self) -> Optional[DriftSnapshot]:
        """Get the current window snapshot."""
        if len(self._observations) < 1:
            return None
        return self._compute_current_snapshot()

    def get_alerts(self, limit: int = 50) -> list[DriftAlert]:
        """Get recent drift alerts."""
        return self._alerts[-limit:]

    def _compute_current_snapshot(self) -> DriftSnapshot:
        """Compute aggregate metrics from the current observation window."""
        obs = list(self._observations)
        n = len(obs) or 1

        return DriftSnapshot(
            mean_confidence=sum(o["confidence"] for o in obs) / n,
            abstention_rate=sum(o["abstained"] for o in obs) / n,
            mean_chunks_retrieved=sum(o["chunks_retrieved"] for o in obs) / n,
            mean_citations_per_answer=sum(o["citations_count"] for o in obs) / n,
            mean_latency_ms=sum(o["latency_ms"] for o in obs) / n,
            sample_count=n,
        )


# Singleton
drift_detector = DriftDetector()
