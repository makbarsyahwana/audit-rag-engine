"""A/B testing infrastructure for RAG pipeline variants."""

import hashlib
import logging
import time
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Variant(str, Enum):
    CONTROL = "control"
    TREATMENT = "treatment"


class ExperimentConfig(BaseModel):
    """Configuration for an A/B experiment."""

    id: str
    name: str
    description: str = ""
    enabled: bool = True
    traffic_split: float = 0.5       # fraction of traffic to treatment
    control_config: dict = Field(default_factory=dict)
    treatment_config: dict = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)


class ExperimentEvent(BaseModel):
    """A single event logged during an experiment."""

    experiment_id: str
    variant: Variant
    user_id: str = ""
    engagement_id: str = ""
    query: str = ""
    metrics: dict = Field(default_factory=dict)  # latency, confidence, etc.
    timestamp: float = Field(default_factory=time.time)


class ExperimentSummary(BaseModel):
    """Summary statistics for an experiment."""

    experiment_id: str
    experiment_name: str
    control_count: int = 0
    treatment_count: int = 0
    control_metrics: dict = Field(default_factory=dict)
    treatment_metrics: dict = Field(default_factory=dict)
    winner: Optional[str] = None
    significant: bool = False


class ABTestManager:
    """Manages A/B experiments for the RAG pipeline."""

    def __init__(self) -> None:
        self._experiments: dict[str, ExperimentConfig] = {}
        self._events: list[ExperimentEvent] = []
        self._max_events = 10000

    def register_experiment(self, config: ExperimentConfig) -> None:
        """Register a new experiment."""
        self._experiments[config.id] = config
        logger.info(
            "Experiment registered: %s (%s), split=%.0f%%",
            config.id, config.name, config.traffic_split * 100,
        )

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentConfig]:
        """Get an experiment by ID."""
        return self._experiments.get(experiment_id)

    def assign_variant(
        self, experiment_id: str, user_id: str
    ) -> Variant:
        """Deterministically assign a user to a variant.

        Uses consistent hashing so the same user always gets the same
        variant for a given experiment.
        """
        exp = self._experiments.get(experiment_id)
        if not exp or not exp.enabled:
            return Variant.CONTROL

        # Consistent hash for deterministic assignment
        hash_input = f"{experiment_id}:{user_id}"
        hash_val = int(
            hashlib.sha256(hash_input.encode()).hexdigest()[:8], 16
        )
        normalized = (hash_val % 1000) / 1000.0

        return (
            Variant.TREATMENT
            if normalized < exp.traffic_split
            else Variant.CONTROL
        )

    def get_config_for_variant(
        self, experiment_id: str, variant: Variant
    ) -> dict:
        """Get the configuration overrides for a variant."""
        exp = self._experiments.get(experiment_id)
        if not exp:
            return {}

        if variant == Variant.TREATMENT:
            return exp.treatment_config
        return exp.control_config

    def log_event(self, event: ExperimentEvent) -> None:
        """Log an experiment event."""
        self._events.append(event)

        # Evict oldest if over limit
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]

    def get_summary(self, experiment_id: str) -> ExperimentSummary:
        """Get summary statistics for an experiment."""
        exp = self._experiments.get(experiment_id)
        if not exp:
            return ExperimentSummary(
                experiment_id=experiment_id, experiment_name="unknown"
            )

        events = [e for e in self._events if e.experiment_id == experiment_id]
        control_events = [e for e in events if e.variant == Variant.CONTROL]
        treatment_events = [
            e for e in events if e.variant == Variant.TREATMENT
        ]

        control_metrics = _aggregate_metrics(control_events)
        treatment_metrics = _aggregate_metrics(treatment_events)

        # Determine winner (simple: higher mean confidence)
        winner = None
        ctrl_conf = control_metrics.get("mean_confidence", 0)
        treat_conf = treatment_metrics.get("mean_confidence", 0)
        if ctrl_conf > 0 or treat_conf > 0:
            winner = "treatment" if treat_conf > ctrl_conf else "control"

        return ExperimentSummary(
            experiment_id=experiment_id,
            experiment_name=exp.name,
            control_count=len(control_events),
            treatment_count=len(treatment_events),
            control_metrics=control_metrics,
            treatment_metrics=treatment_metrics,
            winner=winner,
            significant=len(events) >= 30,  # minimum sample for significance
        )

    def list_experiments(self) -> list[ExperimentConfig]:
        """List all registered experiments."""
        return list(self._experiments.values())


def _aggregate_metrics(events: list[ExperimentEvent]) -> dict:
    """Aggregate metrics from a list of events."""
    if not events:
        return {}

    all_metrics: dict[str, list[float]] = {}
    for event in events:
        for key, value in event.metrics.items():
            if isinstance(value, (int, float)):
                all_metrics.setdefault(key, []).append(float(value))

    aggregated = {}
    for key, values in all_metrics.items():
        aggregated[f"mean_{key}"] = round(sum(values) / len(values), 4)
        aggregated[f"min_{key}"] = round(min(values), 4)
        aggregated[f"max_{key}"] = round(max(values), 4)
        aggregated[f"count_{key}"] = len(values)

    return aggregated


# Singleton
ab_manager = ABTestManager()
