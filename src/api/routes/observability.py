"""Observability and evaluation API routes."""

import logging

from fastapi import APIRouter, Request

from src.evaluation.golden_set import get_golden_set, list_golden_sets
from src.observability.alerts import alert_manager
from src.observability.drift_detection import drift_detector
from src.observability.model_cards import model_registry
from src.security.service_auth import parse_identity

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

@router.get("/alerts")
async def get_alerts(req: Request, firing_only: bool = False):
    """Get current alerts."""
    # Security: identity verification (ASI03)
    parse_identity(req)
    if firing_only:
        return {"alerts": [a.model_dump() for a in alert_manager.get_firing()]}
    return {"alerts": [a.model_dump() for a in alert_manager.get_all()]}


# ---------------------------------------------------------------------------
# Model cards
# ---------------------------------------------------------------------------

@router.get("/models")
async def get_model_cards(req: Request):
    """Get registered model cards."""
    # Security: identity verification (ASI03)
    parse_identity(req)
    return {"models": model_registry.to_dict()}


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

@router.get("/drift")
async def get_drift_status(req: Request):
    """Get drift detection status and recent alerts."""
    # Security: identity verification (ASI03)
    parse_identity(req)
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
async def list_available_golden_sets(req: Request):
    """List available golden question set domains."""
    # Security: identity verification (ASI03)
    parse_identity(req)
    return {"domains": list_golden_sets()}


@router.get("/eval/golden-sets/{domain}")
async def get_golden_set_detail(domain: str, req: Request):
    """Get a golden set by domain."""
    # Security: identity verification (ASI03)
    parse_identity(req)
    gs = get_golden_set(domain)
    if not gs:
        return {"error": f"Golden set not found for domain: {domain}"}
    return gs.model_dump()

