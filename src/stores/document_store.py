"""MongoDB client for document and chunk storage."""

import logging
from typing import Any, Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from src.config import settings

logger = logging.getLogger(__name__)


class DocumentStore:
    """Async MongoDB client for documents and chunks collections."""

    _client: Optional[AsyncIOMotorClient] = None
    _db: Optional[AsyncIOMotorDatabase] = None

    async def connect(self) -> None:
        """Initialize the MongoDB async client."""
        self._client = AsyncIOMotorClient(settings.mongodb_uri)
        self._db = self._client[settings.mongodb_database]
        await self._ensure_indexes()
        logger.info("Connected to MongoDB database: %s", settings.mongodb_database)

    async def close(self) -> None:
        """Close the MongoDB client."""
        if self._client is not None:
            self._client.close()
            logger.info("MongoDB connection closed")

    @property
    def db(self) -> AsyncIOMotorDatabase:
        if self._db is None:
            raise RuntimeError("MongoDB not initialized. Call connect() first.")
        return self._db

    async def _ensure_indexes(self) -> None:
        """Create MongoDB indexes for documents and chunks."""
        docs = self.db["documents"]
        await docs.create_index("engagement_id")
        await docs.create_index("acl.engagement_ids")
        await docs.create_index("processing_status")
        await docs.create_index("content_hash")

        chunks = self.db["chunks"]
        await chunks.create_index("document_id")
        await chunks.create_index("engagement_id")
        await chunks.create_index("chunk_id", unique=True)

        jobs = self.db["ingestion_jobs"]
        await jobs.create_index("job_id", unique=True)
        await jobs.create_index("document_id")

        logger.info("MongoDB indexes ensured")

    # ------------------------------------------------------------------
    # Document CRUD
    # ------------------------------------------------------------------

    async def insert_document(self, doc: dict[str, Any]) -> str:
        """Insert a document record. Returns the inserted _id as string."""
        result = await self.db["documents"].insert_one(doc)
        return str(result.inserted_id)

    async def get_document(self, document_id: str) -> Optional[dict[str, Any]]:
        """Get a document by its document_id field."""
        return await self.db["documents"].find_one({"_id": document_id})

    async def get_document_by_hash(
        self, content_hash: str, engagement_id: str
    ) -> Optional[dict[str, Any]]:
        """Find a document by content hash within an engagement (dedup)."""
        return await self.db["documents"].find_one(
            {"content_hash": content_hash, "engagement_id": engagement_id}
        )

    async def update_document_status(
        self, document_id: str, status: str, **extra_fields: Any
    ) -> None:
        """Update a document's processing status."""
        update = {"$set": {"processing_status": status, **extra_fields}}
        await self.db["documents"].update_one({"_id": document_id}, update)

    async def delete_document(self, document_id: str) -> None:
        """Delete a document and its chunks."""
        await self.db["chunks"].delete_many({"document_id": document_id})
        await self.db["documents"].delete_one({"_id": document_id})

    # ------------------------------------------------------------------
    # Chunk CRUD
    # ------------------------------------------------------------------

    async def insert_chunks(self, chunks: list[dict[str, Any]]) -> int:
        """Insert multiple chunk records. Returns count inserted."""
        if not chunks:
            return 0
        result = await self.db["chunks"].insert_many(chunks)
        return len(result.inserted_ids)

    async def get_chunks_by_document(self, document_id: str) -> list[dict[str, Any]]:
        """Get all chunks for a document, ordered by chunk_index."""
        cursor = self.db["chunks"].find({"document_id": document_id}).sort("chunk_index", 1)
        return await cursor.to_list(length=None)

    async def get_chunk(self, chunk_id: str) -> Optional[dict[str, Any]]:
        """Get a single chunk by chunk_id."""
        return await self.db["chunks"].find_one({"chunk_id": chunk_id})

    # ------------------------------------------------------------------
    # Ingestion job tracking
    # ------------------------------------------------------------------

    async def create_job(self, job: dict[str, Any]) -> None:
        """Create an ingestion job record."""
        await self.db["ingestion_jobs"].insert_one(job)

    async def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        """Get an ingestion job by job_id."""
        return await self.db["ingestion_jobs"].find_one({"job_id": job_id})

    async def update_job(self, job_id: str, **fields: Any) -> None:
        """Update fields on an ingestion job."""
        await self.db["ingestion_jobs"].update_one(
            {"job_id": job_id}, {"$set": fields}
        )


document_store = DocumentStore()
