"""Base connector interface for document source systems."""

from abc import ABC, abstractmethod
from typing import AsyncIterator


class BaseConnector(ABC):
    """Abstract base class for document source connectors.

    Connectors sync documents from external systems (SharePoint, GRC, etc.)
    into the ingestion pipeline.
    """

    @abstractmethod
    async def list_documents(self, **kwargs) -> list[dict]:
        """List available documents from the source system.

        Returns:
            List of document metadata dicts.
        """

    @abstractmethod
    async def fetch_document(self, source_id: str) -> tuple[bytes, dict]:
        """Fetch a single document by source ID.

        Args:
            source_id: Identifier in the source system.

        Returns:
            Tuple of (file_bytes, metadata_dict).
        """

    async def sync(self, **kwargs) -> AsyncIterator[tuple[bytes, dict]]:
        """Iterate over documents to sync from the source.

        Yields:
            Tuple of (file_bytes, metadata_dict) for each document.
        """
        docs = await self.list_documents(**kwargs)
        for doc_meta in docs:
            source_id = doc_meta.get("source_id", "")
            if source_id:
                file_data, metadata = await self.fetch_document(source_id)
                yield file_data, {**doc_meta, **metadata}
