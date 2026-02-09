"""Output anomaly detection and behavioral monitoring (ASI10 — Rogue Agents).

Tracks response distributions per engagement and alerts on statistical
outliers that may indicate goal drift, poisoning, or rogue behavior.
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EngagementStats:
    """Rolling statistics for an engagement's LLM outputs."""

    response_lengths: list[int] = field(default_factory=list)
    confidence_scores: list[float] = field(default_factory=list)
    citation_counts: list[int] = field(default_factory=list)
    abstention_count: int = 0
    total_count: int = 0

    def _mean_std(self, values: list) -> tuple[float, float]:
        if len(values) < 3:
            return 0.0, 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return mean, math.sqrt(variance)

    @property
    def avg_confidence(self) -> float:
        if not self.confidence_scores:
            return 0.0
        return sum(self.confidence_scores) / len(self.confidence_scores)

    @property
    def abstention_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.abstention_count / self.total_count


@dataclass
class AnomalyAlert:
    """An anomaly detected in LLM behavior."""

    engagement_id: str
    alert_type: str
    message: str
    value: float
    threshold: float


class BehavioralMonitor:
    """Tracks per-engagement LLM output patterns for anomaly detection."""

    MAX_HISTORY = 100  # Keep last N data points per engagement

    def __init__(self) -> None:
        self._stats: dict[str, EngagementStats] = defaultdict(
            EngagementStats
        )

    def record(
        self,
        engagement_id: str,
        response_length: int,
        confidence: float,
        citation_count: int,
        is_abstention: bool = False,
    ) -> list[AnomalyAlert]:
        """Record a response and check for anomalies.

        Returns list of alerts (empty if normal).
        """
        stats = self._stats[engagement_id]
        stats.total_count += 1

        # Trim history
        if len(stats.response_lengths) >= self.MAX_HISTORY:
            stats.response_lengths = stats.response_lengths[-self.MAX_HISTORY:]
            stats.confidence_scores = stats.confidence_scores[-self.MAX_HISTORY:]
            stats.citation_counts = stats.citation_counts[-self.MAX_HISTORY:]

        alerts: list[AnomalyAlert] = []

        # Check for outliers (only after enough data)
        if stats.total_count > 10:
            alerts.extend(
                self._check_anomalies(
                    engagement_id,
                    stats,
                    response_length,
                    confidence,
                    citation_count,
                )
            )

        # Record after checking
        stats.response_lengths.append(response_length)
        stats.confidence_scores.append(confidence)
        stats.citation_counts.append(citation_count)
        if is_abstention:
            stats.abstention_count += 1

        for alert in alerts:
            logger.warning(
                "Behavioral anomaly [%s]: %s (value=%.2f, threshold=%.2f)",
                alert.alert_type,
                alert.message,
                alert.value,
                alert.threshold,
            )

        return alerts

    def _check_anomalies(
        self,
        engagement_id: str,
        stats: EngagementStats,
        response_length: int,
        confidence: float,
        citation_count: int,
    ) -> list[AnomalyAlert]:
        alerts: list[AnomalyAlert] = []

        # Sudden confidence drop
        mean_conf, std_conf = stats._mean_std(stats.confidence_scores)
        if std_conf > 0 and confidence < mean_conf - 2 * std_conf:
            alerts.append(AnomalyAlert(
                engagement_id=engagement_id,
                alert_type="confidence_drop",
                message=(
                    f"Confidence {confidence:.2f} is >2σ below "
                    f"mean {mean_conf:.2f}"
                ),
                value=confidence,
                threshold=mean_conf - 2 * std_conf,
            ))

        # Abnormally long response (potential data dump)
        mean_len, std_len = stats._mean_std(stats.response_lengths)
        if std_len > 0 and response_length > mean_len + 3 * std_len:
            alerts.append(AnomalyAlert(
                engagement_id=engagement_id,
                alert_type="response_length_spike",
                message=(
                    f"Response length {response_length} is >3σ above "
                    f"mean {mean_len:.0f}"
                ),
                value=float(response_length),
                threshold=mean_len + 3 * std_len,
            ))

        # High abstention rate
        if stats.abstention_rate > 0.5 and stats.total_count > 20:
            alerts.append(AnomalyAlert(
                engagement_id=engagement_id,
                alert_type="high_abstention",
                message=(
                    f"Abstention rate {stats.abstention_rate:.0%} "
                    f"exceeds 50% threshold"
                ),
                value=stats.abstention_rate,
                threshold=0.5,
            ))

        return alerts

    def get_stats(self, engagement_id: str) -> dict:
        """Get engagement stats for monitoring dashboard."""
        stats = self._stats.get(engagement_id)
        if not stats:
            return {}
        return {
            "total_count": stats.total_count,
            "avg_confidence": round(stats.avg_confidence, 3),
            "abstention_rate": round(stats.abstention_rate, 3),
            "abstention_count": stats.abstention_count,
        }


behavioral_monitor = BehavioralMonitor()
