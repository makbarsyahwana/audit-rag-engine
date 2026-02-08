"""API routes for external connectors and sync management (Phase 4)."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.connectors.base import ConnectorConfig, ConnectorType
from src.connectors.sync_manager import get_freshness_report, sync_manager

logger = logging.getLogger(__name__)

router = APIRouter()


# ------------------------------------------------------------------
# Request / Response models
# ------------------------------------------------------------------


class RegisterConnectorRequest(BaseModel):
    connector_type: ConnectorType
    name: str
    base_url: str = ""
    auth_type: str = "oauth2"
    credentials: dict[str, str] = Field(default_factory=dict)
    sync_interval_minutes: int = 60
    enabled: bool = True
    filters: dict = Field(default_factory=dict)


class SyncRequest(BaseModel):
    engagement_id: str
    full_sync: bool = False
    filters: dict = Field(default_factory=dict)


class SyncResponse(BaseModel):
    connector_id: str
    engagement_id: str
    status: str
    documents_synced: int = 0
    documents_failed: int = 0
    error_message: Optional[str] = None


# ------------------------------------------------------------------
# Connector management
# ------------------------------------------------------------------


@router.get("/connectors")
async def list_connectors():
    """List all registered connectors."""
    return sync_manager.list_connectors()


@router.post("/connectors")
async def register_connector(req: RegisterConnectorRequest):
    """Register a new connector."""
    config = ConnectorConfig(**req.model_dump())

    # Create the appropriate connector instance
    connector = _create_connector(config)
    sync_manager.register(connector)

    connector_id = f"{config.connector_type.value}:{config.name}"
    return {"connector_id": connector_id, "status": "registered"}


@router.delete("/connectors/{connector_id}")
async def unregister_connector(connector_id: str):
    """Unregister a connector."""
    parts = connector_id.split(":", 1)
    if len(parts) != 2:
        raise HTTPException(400, "Invalid connector_id format (type:name)")

    sync_manager.unregister(parts[0], parts[1])
    return {"connector_id": connector_id, "status": "unregistered"}


@router.post("/connectors/{connector_id}/test")
async def test_connector(connector_id: str):
    """Test a connector's connection."""
    connector = sync_manager.get_connector(connector_id)
    if not connector:
        raise HTTPException(404, f"Connector not found: {connector_id}")

    try:
        await connector.connect()
        ok = await connector.test_connection()
        await connector.disconnect()
        return {"connector_id": connector_id, "connected": ok}
    except Exception as exc:
        return {
            "connector_id": connector_id,
            "connected": False,
            "error": str(exc),
        }


# ------------------------------------------------------------------
# Sync operations
# ------------------------------------------------------------------


@router.post("/connectors/{connector_id}/sync", response_model=SyncResponse)
async def run_sync(connector_id: str, req: SyncRequest):
    """Trigger a sync for a connector."""
    try:
        state = await sync_manager.run_sync(
            connector_id=connector_id,
            engagement_id=req.engagement_id,
            full_sync=req.full_sync,
            **req.filters,
        )
        return SyncResponse(
            connector_id=state.connector_id,
            engagement_id=state.engagement_id,
            status=state.last_sync_status.value,
            documents_synced=state.documents_synced,
            documents_failed=state.documents_failed,
            error_message=state.error_message,
        )
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        logger.error("Sync failed: %s", exc, exc_info=True)
        raise HTTPException(500, f"Sync failed: {exc}")


@router.get("/connectors/sync-states")
async def list_sync_states(engagement_id: Optional[str] = None):
    """Get sync states for all connectors."""
    states = await sync_manager.get_all_sync_states(engagement_id)
    return [s.model_dump(mode="json") for s in states]


# ------------------------------------------------------------------
# Freshness monitoring
# ------------------------------------------------------------------


@router.get("/connectors/freshness")
async def freshness_report(engagement_id: Optional[str] = None):
    """Get freshness report for all connectors.

    Shows staleness metrics and whether connectors are overdue for sync.
    """
    return await get_freshness_report(engagement_id)


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------


def _create_connector(config: ConnectorConfig):
    """Instantiate the correct connector class from config."""
    if config.connector_type == ConnectorType.SHAREPOINT:
        from src.connectors.sharepoint import SharePointConnector

        return SharePointConnector(config)
    elif config.connector_type == ConnectorType.CONFLUENCE:
        from src.connectors.confluence import ConfluenceConnector

        return ConfluenceConnector(config)
    elif config.connector_type == ConnectorType.GRC:
        from src.connectors.grc import GRCConnector

        return GRCConnector(config)
    elif config.connector_type == ConnectorType.JIRA:
        from src.connectors.ticketing import JiraConnector

        return JiraConnector(config)
    elif config.connector_type == ConnectorType.SERVICENOW:
        from src.connectors.ticketing import ServiceNowConnector

        return ServiceNowConnector(config)
    else:
        raise ValueError(
            f"Unsupported connector type: {config.connector_type}"
        )
