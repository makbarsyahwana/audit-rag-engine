"""Incremental sync manager — orchestrates connector sync and ingestion.

Tracks sync state per connector+engagement in MongoDB, supports
scheduled and on-demand sync, and feeds fetched documents into
the existing ingestion pipeline.
"""

import logging
from datetime import UTC, datetime
from typing import Any, Optional

from src.connectors.base import (
    BaseConnector,
    ConnectorDocument,
    SyncState,
    SyncStatus,
)
from src.stores.document_store import document_store

logger = logging.getLogger(__name__)

# MongoDB collection for sync state
SYNC_STATE_COLLECTION = "connector_sync_state"


class SyncManager:
    """Manages incremental sync across all configured connectors."""

    def __init__(self) -> None:
        self._connectors: dict[str, BaseConnector] = {}

    def register(self, connector: BaseConnector) -> None:
        """Register a connector instance."""
        key = f"{connector.config.connector_type.value}:{connector.config.name}"
        self._connectors[key] = connector
        logger.info("Registered connector: %s", key)

    def unregister(self, connector_type: str, name: str) -> None:
        key = f"{connector_type}:{name}"
        self._connectors.pop(key, None)
        logger.info("Unregistered connector: %s", key)

    def get_connector(self, connector_id: str) -> Optional[BaseConnector]:
        return self._connectors.get(connector_id)

    def list_connectors(self) -> list[dict[str, Any]]:
        """List all registered connectors with their config (sans credentials)."""
        result = []
        for key, conn in self._connectors.items():
            cfg = conn.config.model_dump(exclude={"credentials"})
            cfg["connector_id"] = key
            result.append(cfg)
        return result

    # ------------------------------------------------------------------
    # Sync state persistence (MongoDB)
    # ------------------------------------------------------------------

    async def get_sync_state(
        self, connector_id: str, engagement_id: str
    ) -> Optional[SyncState]:
        """Get the last sync state for a connector+engagement pair."""
        doc = await document_store.db[SYNC_STATE_COLLECTION].find_one(
            {"connector_id": connector_id, "engagement_id": engagement_id}
        )
        if doc:
            doc.pop("_id", None)
            return SyncState(**doc)
        return None

    async def save_sync_state(self, state: SyncState) -> None:
        """Upsert sync state."""
        await document_store.db[SYNC_STATE_COLLECTION].update_one(
            {
                "connector_id": state.connector_id,
                "engagement_id": state.engagement_id,
            },
            {"$set": state.model_dump(mode="json")},
            upsert=True,
        )

    async def get_all_sync_states(
        self, engagement_id: Optional[str] = None
    ) -> list[SyncState]:
        """Get all sync states, optionally filtered by engagement."""
        query: dict[str, Any] = {}
        if engagement_id:
            query["engagement_id"] = engagement_id

        cursor = document_store.db[SYNC_STATE_COLLECTION].find(query)
        states = []
        async for doc in cursor:
            doc.pop("_id", None)
            states.append(SyncState(**doc))
        return states

    # ------------------------------------------------------------------
    # Sync execution
    # ------------------------------------------------------------------

    async def run_sync(
        self,
        connector_id: str,
        engagement_id: str,
        full_sync: bool = False,
        **filters: Any,
    ) -> SyncState:
        """Run a sync for a specific connector and engagement.

        Args:
            connector_id: Registered connector key.
            engagement_id: Target engagement for ingested documents.
            full_sync: If True, ignore last_sync_at and fetch everything.
            **filters: Additional filters passed to the connector.

        Returns:
            Updated SyncState with sync results.
        """
        connector = self._connectors.get(connector_id)
        if not connector:
            raise ValueError(f"Connector not found: {connector_id}")

        # Get previous sync state for incremental sync
        since = None
        if not full_sync:
            prev_state = await self.get_sync_state(
                connector_id, engagement_id
            )
            if prev_state and prev_state.last_sync_at:
                since = prev_state.last_sync_at

        state = SyncState(
            connector_id=connector_id,
            connector_type=connector.config.connector_type,
            engagement_id=engagement_id,
            last_sync_status=SyncStatus.RUNNING,
        )
        await self.save_sync_state(state)

        try:
            await connector.connect()

            # List documents (incremental if since is set)
            documents = await connector.list_documents(
                since=since, **filters
            )
            logger.info(
                "Sync %s: found %d documents (since=%s)",
                connector_id,
                len(documents),
                since,
            )

            synced = 0
            failed = 0

            for doc_meta in documents:
                try:
                    doc = await connector.fetch_document(
                        doc_meta.external_id
                    )
                    await self._ingest_document(
                        doc, engagement_id, connector
                    )
                    synced += 1
                except Exception as exc:
                    logger.error(
                        "Failed to sync document %s: %s",
                        doc_meta.external_id,
                        exc,
                    )
                    failed += 1

            state.last_sync_at = datetime.now(UTC)
            state.last_sync_status = SyncStatus.COMPLETED
            state.documents_synced = synced
            state.documents_failed = failed

        except Exception as exc:
            logger.error("Sync failed for %s: %s", connector_id, exc)
            state.last_sync_status = SyncStatus.FAILED
            state.error_message = str(exc)

        finally:
            try:
                await connector.disconnect()
            except Exception:
                pass

        await self.save_sync_state(state)
        return state

    async def _ingest_document(
        self,
        doc: ConnectorDocument,
        engagement_id: str,
        connector: BaseConnector,
    ) -> dict[str, Any]:
        """Feed a connector document into the ingestion pipeline."""
        from src.ingestion.stages import run_ingestion_pipeline

        result = await run_ingestion_pipeline(
            file_data=doc.content,
            filename=doc.filename,
            engagement_id=engagement_id,
            doc_type=doc.metadata.get("doc_type", "other"),
            source_system=connector.config.connector_type.value,
            title=doc.title,
            tags=doc.metadata.get("labels", []),
            content_type=doc.content_type,
        )
        logger.info(
            "Ingested %s from %s: %s",
            doc.filename,
            connector.config.name,
            result.get("status"),
        )
        return result


# Global sync manager instance
sync_manager = SyncManager()


# ------------------------------------------------------------------
# Freshness monitoring
# ------------------------------------------------------------------


async def get_freshness_report(
    engagement_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Generate a freshness report for all connectors.

    Returns a list of dicts with connector info and staleness metrics.
    """
    states = await sync_manager.get_all_sync_states(engagement_id)
    now = datetime.now(UTC)
    report = []

    for state in states:
        staleness_hours = None
        if state.last_sync_at:
            delta = now - state.last_sync_at
            staleness_hours = round(delta.total_seconds() / 3600, 1)

        connector = sync_manager.get_connector(state.connector_id)
        interval_minutes = (
            connector.config.sync_interval_minutes if connector else 60
        )

        is_stale = False
        if staleness_hours is not None:
            is_stale = staleness_hours > (interval_minutes / 60) * 2

        report.append(
            {
                "connector_id": state.connector_id,
                "connector_type": state.connector_type.value,
                "engagement_id": state.engagement_id,
                "last_sync_at": (
                    state.last_sync_at.isoformat()
                    if state.last_sync_at
                    else None
                ),
                "last_sync_status": state.last_sync_status.value,
                "documents_synced": state.documents_synced,
                "documents_failed": state.documents_failed,
                "staleness_hours": staleness_hours,
                "is_stale": is_stale,
                "sync_interval_minutes": interval_minutes,
                "error_message": state.error_message,
            }
        )

    return report
