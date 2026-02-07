"""Direct file upload connector (MVP default)."""

import logging

from src.ingestion.connectors.base import BaseConnector

logger = logging.getLogger(__name__)


class FileUploadConnector(BaseConnector):
    """Connector for direct file uploads via the API.

    This is the simplest connector — files are uploaded directly
    through the /ingest/document endpoint. No external system sync.
    """

    async def list_documents(self, **kwargs) -> list[dict]:
        """Not applicable for direct uploads. Returns empty list."""
        return []

    async def fetch_document(self, source_id: str) -> tuple[bytes, dict]:
        """Not applicable for direct uploads."""
        raise NotImplementedError(
            "FileUploadConnector does not fetch from external systems. "
            "Use the /ingest/document endpoint directly."
        )
