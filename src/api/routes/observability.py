"""Observability and evaluation API routes."""

import logging

from fastapi import APIRouter

from src.evaluation.ab_testing import ab_manager
from src.evaluation.golden_set import get_golden_set, list_golden_sets
from src.observability.alerts import alert_manager
from src.observability.drift_detection import drift_detector
from src.observability.model_cards import model_registry

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

@router.get("/alerts")
async def get_alerts(firing_only: bool = False):
    """Get current alerts."""
    if firing_only:
        return {"alerts": [a.model_dump() for a in alert_manager.get_firing()]}
    return {"alerts": [a.model_dump() for a in alert_manager.get_all()]}


# ---------------------------------------------------------------------------
# Model cards
# ---------------------------------------------------------------------------

@router.get("/models")
async def get_model_cards():
    """Get registered model cards."""
    return {"models": model_registry.to_dict()}


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

@router.get("/drift")
async def get_drift_status():
    """Get drift detection status and recent alerts."""
    snapshot = drift_detector.get_current_snapshot()
    alerts = drift_detector.get_alerts(limit=20)
    return {
        "current_snapshot": snapshot.model_dump() if snapshot else None,
        "drift_alerts": [a.model_dump() for a in alerts],
    }


# ---------------------------------------------------------------------------
# Golden sets
# ---------------------------------------------------------------------------

@router.get("/eval/golden-sets")
async def list_available_golden_sets():
    """List available golden question set domains."""
    return {"domains": list_golden_sets()}


@router.get("/eval/golden-sets/{domain}")
async def get_golden_set_detail(domain: str):
    """Get a golden set by domain."""
    gs = get_golden_set(domain)
    if not gs:
        return {"error": f"Golden set not found for domain: {domain}"}
    return gs.model_dump()


# ---------------------------------------------------------------------------
# A/B experiments
# ---------------------------------------------------------------------------

@router.get("/experiments")
async def list_experiments():
    """List registered A/B experiments."""
    return {
        "experiments": [e.model_dump() for e in ab_manager.list_experiments()]
    }


@router.get("/experiments/{experiment_id}/summary")
async def get_experiment_summary(experiment_id: str):
    """Get summary for an A/B experiment."""
    return ab_manager.get_summary(experiment_id).model_dump()
