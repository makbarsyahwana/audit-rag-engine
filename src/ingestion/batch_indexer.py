"""Batch indexer for large-scale document ingestion.

Optimizations:
- Batch embedding generation (reduces API calls)
- Batch Neo4j upserts (single transaction per batch)
- Batch MongoDB inserts
- Configurable batch sizes and concurrency
- Progress tracking and resumability
"""

import asyncio
import logging
import uuid
from datetime import UTC, datetime
from typing import Any, Optional

from src.ingestion.chunker import chunk_document
from src.ingestion.converter import convert_document, create_converter
from src.ingestion.embedder import embed_texts
from src.models.chunk import ChunkMetadata, ChunkRecord
from src.models.document import ProcessingStatus  # noqa: F401
from src.stores.document_store import document_store
from src.stores.neo4j_store import neo4j_store

logger = logging.getLogger(__name__)

# Batch size defaults
EMBEDDING_BATCH_SIZE = 64
NEO4J_BATCH_SIZE = 200
MONGO_BATCH_SIZE = 500
MAX_CONCURRENT_CONVERSIONS = 4


class BatchIndexer:
    """Optimized batch indexer for large-scale ingestion."""

    def __init__(
        self,
        embedding_batch_size: int = EMBEDDING_BATCH_SIZE,
        neo4j_batch_size: int = NEO4J_BATCH_SIZE,
        mongo_batch_size: int = MONGO_BATCH_SIZE,
        max_concurrent: int = MAX_CONCURRENT_CONVERSIONS,
    ) -> None:
        self.embedding_batch_size = embedding_batch_size
        self.neo4j_batch_size = neo4j_batch_size
        self.mongo_batch_size = mongo_batch_size
        self.max_concurrent = max_concurrent
        self._converter = None
        self._semaphore: Optional[asyncio.Semaphore] = None

    def _get_converter(self):
        if self._converter is None:
            self._converter = create_converter()
        return self._converter

    async def index_batch(
        self,
        documents: list[dict[str, Any]],
        engagement_id: str,
    ) -> dict[str, Any]:
        """Index a batch of documents with optimized batching.

        Args:
            documents: List of dicts with keys:
                - file_data (bytes)
                - filename (str)
                - doc_type (str, optional)
                - title (str, optional)
                - tags (list[str], optional)
                - content_type (str, optional)
            engagement_id: Target engagement.

        Returns:
            Summary with total_documents, total_chunks, failed, timing.
        """
        started_at = datetime.now(UTC)
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

        total_docs = len(documents)
        successful = 0
        failed = 0
        total_chunks = 0
        errors: list[dict[str, str]] = []

        # Phase 1: Convert all documents concurrently (bounded)
        logger.info(
            "Batch indexing %d documents for engagement %s",
            total_docs,
            engagement_id,
        )

        conversion_tasks = [
            self._convert_one(doc) for doc in documents
        ]
        conversion_results = await asyncio.gather(
            *conversion_tasks, return_exceptions=True
        )

        # Phase 2: Chunk all converted documents
        all_chunks: list[dict[str, Any]] = []
        doc_chunk_map: list[tuple[int, int]] = []  # (start, count)

        for idx, result in enumerate(conversion_results):
            if isinstance(result, Exception):
                failed += 1
                errors.append({
                    "filename": documents[idx].get("filename", "?"),
                    "error": str(result),
                })
                continue

            docling_doc, doc_meta = result
            chunks = chunk_document(docling_doc)
            if not chunks:
                failed += 1
                errors.append({
                    "filename": doc_meta["filename"],
                    "error": "No chunks produced",
                })
                continue

            start_idx = len(all_chunks)
            for chunk_idx, chunk_dict in enumerate(chunks):
                all_chunks.append({
                    "chunk_dict": chunk_dict,
                    "doc_meta": doc_meta,
                    "chunk_index": chunk_idx,
                })
            doc_chunk_map.append((start_idx, len(chunks)))

        if not all_chunks:
            return {
                "total_documents": total_docs,
                "successful": 0,
                "failed": failed,
                "total_chunks": 0,
                "errors": errors,
                "elapsed_seconds": (
                    datetime.now(UTC) - started_at
                ).total_seconds(),
            }

        # Phase 3: Batch embed all chunks
        texts = [c["chunk_dict"]["content"] for c in all_chunks]
        all_embeddings = await self._batch_embed(texts)

        # Phase 4: Batch upsert to MongoDB and Neo4j
        mongo_chunks: list[dict[str, Any]] = []
        neo4j_chunks: list[dict[str, Any]] = []

        for idx, (chunk_info, embedding) in enumerate(
            zip(all_chunks, all_embeddings)
        ):
            chunk_id = str(uuid.uuid4())
            doc_meta = chunk_info["doc_meta"]
            chunk_dict = chunk_info["chunk_dict"]
            meta = chunk_dict.get("metadata", {})

            mongo_chunk = ChunkRecord(
                chunk_id=chunk_id,
                document_id=doc_meta["document_id"],
                engagement_id=engagement_id,
                chunk_index=chunk_info["chunk_index"],
                content=chunk_dict["content"],
                content_preview=chunk_dict.get("content_preview", ""),
                token_count=len(chunk_dict["content"].split()),
                doc_type=doc_meta.get("doc_type", "other"),
                metadata=ChunkMetadata(
                    page_number=meta.get("page_number"),
                    section=meta.get("section"),
                    heading=meta.get("heading"),
                    heading_path=meta.get("heading_path", []),
                    bbox=meta.get("bbox"),
                ),
            )
            mongo_chunks.append(mongo_chunk.model_dump(by_alias=True))

            neo4j_chunks.append({
                "chunk_id": chunk_id,
                "document_id": doc_meta["document_id"],
                "engagement_id": engagement_id,
                "content": chunk_dict["content"],
                "content_preview": chunk_dict.get("content_preview", ""),
                "chunk_index": chunk_info["chunk_index"],
                "page_number": meta.get("page_number"),
                "section": meta.get("section"),
                "doc_type": doc_meta.get("doc_type", "other"),
                "embedding": embedding,
            })

        # Batch insert to MongoDB
        for i in range(0, len(mongo_chunks), self.mongo_batch_size):
            batch = mongo_chunks[i : i + self.mongo_batch_size]
            await document_store.insert_chunks(batch)

        # Batch upsert to Neo4j
        for i in range(0, len(neo4j_chunks), self.neo4j_batch_size):
            batch = neo4j_chunks[i : i + self.neo4j_batch_size]
            await neo4j_store.upsert_chunks_batch(batch)

        total_chunks = len(all_chunks)
        successful = total_docs - failed

        elapsed = (datetime.now(UTC) - started_at).total_seconds()
        logger.info(
            "Batch indexing complete: %d docs, %d chunks, %d failed, %.1fs",
            successful,
            total_chunks,
            failed,
            elapsed,
        )

        return {
            "total_documents": total_docs,
            "successful": successful,
            "failed": failed,
            "total_chunks": total_chunks,
            "errors": errors,
            "elapsed_seconds": round(elapsed, 2),
        }

    async def _convert_one(
        self, doc: dict[str, Any]
    ) -> tuple[Any, dict[str, Any]]:
        """Convert a single document with concurrency limiting."""
        async with self._semaphore:  # type: ignore[union-attr]
            import tempfile
            from pathlib import Path

            document_id = str(uuid.uuid4())
            filename = doc.get("filename", "unknown")
            file_data = doc.get("file_data", b"")

            suffix = Path(filename).suffix
            with tempfile.NamedTemporaryFile(
                suffix=suffix, delete=True
            ) as tmp:
                tmp.write(file_data)
                tmp.flush()
                docling_doc = await asyncio.to_thread(
                    convert_document,
                    tmp.name,
                    converter=self._get_converter(),
                )

            return docling_doc, {
                "document_id": document_id,
                "filename": filename,
                "doc_type": doc.get("doc_type", "other"),
                "title": doc.get("title", filename),
            }

    async def _batch_embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches to avoid API limits."""
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.embedding_batch_size):
            batch = texts[i : i + self.embedding_batch_size]
            embeddings = await embed_texts(batch)
            all_embeddings.extend(embeddings)
            logger.debug(
                "Embedded batch %d-%d / %d",
                i,
                i + len(batch),
                len(texts),
            )
        return all_embeddings


# Global batch indexer
batch_indexer = BatchIndexer()
