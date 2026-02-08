"""Base connector interface for external document sources."""

import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SyncStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ConnectorType(str, Enum):
    SHAREPOINT = "sharepoint"
    CONFLUENCE = "confluence"
    GRC = "grc"
    JIRA = "jira"
    SERVICENOW = "servicenow"


class ConnectorConfig(BaseModel):
    """Configuration for a connector instance."""

    connector_type: ConnectorType
    name: str
    base_url: str = ""
    auth_type: str = "oauth2"  # oauth2, api_key, basic, pat
    credentials: dict[str, str] = Field(default_factory=dict)
    sync_interval_minutes: int = 60
    enabled: bool = True
    filters: dict[str, Any] = Field(default_factory=dict)


class SyncState(BaseModel):
    """Tracks incremental sync state for a connector."""

    connector_id: str
    connector_type: ConnectorType
    engagement_id: str
    last_sync_at: Optional[datetime] = None
    last_sync_token: Optional[str] = None
    last_sync_status: SyncStatus = SyncStatus.IDLE
    documents_synced: int = 0
    documents_failed: int = 0
    error_message: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConnectorDocument(BaseModel):
    """A document fetched from an external source."""

    external_id: str
    title: str
    filename: str
    content: bytes = b""
    content_type: str = "application/octet-stream"
    source_url: str = ""
    last_modified: Optional[datetime] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaseConnector(ABC):
    """Abstract base class for all external connectors."""

    connector_type: ConnectorType

    def __init__(self, config: ConnectorConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{config.connector_type.value}")

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the external system."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the connection."""

    @abstractmethod
    async def test_connection(self) -> bool:
        """Test if the connection is healthy. Returns True if OK."""

    @abstractmethod
    async def list_documents(
        self,
        since: Optional[datetime] = None,
        **filters: Any,
    ) -> list[ConnectorDocument]:
        """List documents available for sync.

        Args:
            since: Only return docs modified after this timestamp (incremental sync).
            **filters: Connector-specific filters (site, space, project, etc.).

        Returns:
            List of ConnectorDocument metadata (content may be empty until fetch).
        """

    @abstractmethod
    async def fetch_document(self, external_id: str) -> ConnectorDocument:
        """Fetch a single document's full content by its external ID."""

    async def sync(
        self,
        engagement_id: str,
        since: Optional[datetime] = None,
        **filters: Any,
    ) -> SyncState:
        """Run a full or incremental sync.

        1. List documents (optionally since last sync).
        2. Fetch each document.
        3. Return sync state for tracking.
        """
        state = SyncState(
            connector_id=f"{self.config.connector_type.value}:{self.config.name}",
            connector_type=self.config.connector_type,
            engagement_id=engagement_id,
            last_sync_status=SyncStatus.RUNNING,
        )

        try:
            docs = await self.list_documents(since=since, **filters)
            self.logger.info(
                "Found %d documents to sync from %s",
                len(docs),
                self.config.name,
            )

            synced = 0
            failed = 0

            for doc_meta in docs:
                try:
                    doc = await self.fetch_document(doc_meta.external_id)
                    yield doc  # type: ignore[misc]
                    synced += 1
                except Exception as exc:
                    self.logger.error(
                        "Failed to fetch %s: %s", doc_meta.external_id, exc
                    )
                    failed += 1

            state.last_sync_at = datetime.now(UTC)
            state.last_sync_status = SyncStatus.COMPLETED
            state.documents_synced = synced
            state.documents_failed = failed

        except Exception as exc:
            self.logger.error("Sync failed for %s: %s", self.config.name, exc)
            state.last_sync_status = SyncStatus.FAILED
            state.error_message = str(exc)

        self._last_sync_state = state
