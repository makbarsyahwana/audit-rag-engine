"""Alert definitions for ingestion failures and permission errors.

Provides an in-process alert manager that tracks alert conditions and
exposes them via GET /alerts for external monitoring (Grafana, PagerDuty).
"""

import logging
import time
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    FIRING = "firing"
    RESOLVED = "resolved"


class Alert(BaseModel):
    """A single alert instance."""

    id: str
    name: str
    severity: AlertSeverity
    status: AlertStatus = AlertStatus.FIRING
    message: str
    labels: dict[str, str] = Field(default_factory=dict)
    fired_at: float = Field(default_factory=time.time)
    resolved_at: Optional[float] = None
    count: int = 1


class AlertManager:
    """In-process alert manager for tracking alert conditions."""

    def __init__(self, max_alerts: int = 1000) -> None:
        self._alerts: dict[str, Alert] = {}
        self._max_alerts = max_alerts

    def fire(
        self,
        name: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.WARNING,
        labels: Optional[dict[str, str]] = None,
    ) -> Alert:
        """Fire or increment an alert."""
        alert_id = f"{name}:{_label_hash(labels)}"

        if alert_id in self._alerts:
            existing = self._alerts[alert_id]
            existing.count += 1
            existing.status = AlertStatus.FIRING
            existing.message = message
            existing.resolved_at = None
            logger.warning(
                "Alert incremented: %s (count=%d) — %s",
                name, existing.count, message,
            )
            return existing

        alert = Alert(
            id=alert_id,
            name=name,
            severity=severity,
            message=message,
            labels=labels or {},
        )
        self._alerts[alert_id] = alert

        # Evict oldest if over limit
        if len(self._alerts) > self._max_alerts:
            oldest_key = min(
                self._alerts, key=lambda k: self._alerts[k].fired_at
            )
            del self._alerts[oldest_key]

        logger.warning("Alert fired: %s — %s", name, message)
        return alert

    def resolve(self, name: str, labels: Optional[dict[str, str]] = None) -> None:
        """Resolve an alert."""
        alert_id = f"{name}:{_label_hash(labels)}"
        if alert_id in self._alerts:
            self._alerts[alert_id].status = AlertStatus.RESOLVED
            self._alerts[alert_id].resolved_at = time.time()
            logger.info("Alert resolved: %s", name)

    def get_firing(self) -> list[Alert]:
        """Get all currently firing alerts."""
        return [
            a for a in self._alerts.values()
            if a.status == AlertStatus.FIRING
        ]

    def get_all(self) -> list[Alert]:
        """Get all alerts (firing + resolved)."""
        return list(self._alerts.values())

    def clear(self) -> None:
        """Clear all alerts."""
        self._alerts.clear()


def _label_hash(labels: Optional[dict[str, str]]) -> str:
    if not labels:
        return "default"
    return "|".join(f"{k}={v}" for k, v in sorted(labels.items()))


# Singleton
alert_manager = AlertManager()


# ---------------------------------------------------------------------------
# Convenience helpers for common alert types
# ---------------------------------------------------------------------------

def alert_ingestion_failure(
    document_id: str, error: str, engagement_id: str = ""
) -> Alert:
    """Fire an alert for an ingestion failure."""
    return alert_manager.fire(
        name="ingestion_failure",
        message=f"Ingestion failed for document {document_id}: {error}",
        severity=AlertSeverity.CRITICAL,
        labels={
            "document_id": document_id,
            "engagement_id": engagement_id,
        },
    )


def alert_permission_error(
    user_id: str, resource: str, action: str
) -> Alert:
    """Fire an alert for a permission violation."""
    return alert_manager.fire(
        name="permission_error",
        message=(
            f"Permission denied: user {user_id} attempted "
            f"{action} on {resource}"
        ),
        severity=AlertSeverity.WARNING,
        labels={"user_id": user_id, "resource": resource, "action": action},
    )


def alert_llm_error(error: str, model: str = "") -> Alert:
    """Fire an alert for an LLM invocation failure."""
    return alert_manager.fire(
        name="llm_error",
        message=f"LLM error ({model}): {error}",
        severity=AlertSeverity.CRITICAL,
        labels={"model": model},
    )


def alert_store_connection(store: str, error: str) -> Alert:
    """Fire an alert for a database connection failure."""
    return alert_manager.fire(
        name="store_connection_failure",
        message=f"{store} connection failed: {error}",
        severity=AlertSeverity.CRITICAL,
        labels={"store": store},
    )
