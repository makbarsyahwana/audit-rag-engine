"""Docling HybridChunker - structure-aware, tokenizer-aligned chunking."""

import logging
from typing import Optional

from docling_core.transforms.chunker import HybridChunker
from docling_core.types.doc import DoclingDocument

from src.config import settings

logger = logging.getLogger(__name__)


def create_chunker(
    tokenizer: Optional[str] = None,
    max_tokens: Optional[int] = None,
    merge_peers: Optional[bool] = None,
) -> HybridChunker:
    """Create a Docling HybridChunker aligned to the embedding model tokenizer.

    Args:
        tokenizer: HuggingFace tokenizer name or path. Defaults to embedding model.
        max_tokens: Max tokens per chunk. Defaults to settings.chunk_max_tokens.
        merge_peers: Merge undersized adjacent chunks with same headings.

    Returns:
        Configured HybridChunker instance.
    """
    _tokenizer = tokenizer or settings.embedding_model
    _max_tokens = max_tokens if max_tokens is not None else settings.chunk_max_tokens
    _merge_peers = merge_peers if merge_peers is not None else settings.chunk_merge_peers

    chunker = HybridChunker(
        tokenizer=_tokenizer,
        max_tokens=_max_tokens,
        merge_peers=_merge_peers,
    )

    logger.info(
        "HybridChunker created (tokenizer=%s, max_tokens=%d, merge_peers=%s)",
        _tokenizer,
        _max_tokens,
        _merge_peers,
    )
    return chunker


def chunk_document(
    doc: DoclingDocument,
    chunker: Optional[HybridChunker] = None,
) -> list[dict]:
    """Chunk a DoclingDocument using HybridChunker.

    Args:
        doc: Parsed DoclingDocument from Docling converter.
        chunker: Optional pre-configured chunker. Creates default if None.

    Returns:
        List of chunk dicts with content, metadata, and provenance.
    """
    if chunker is None:
        chunker = create_chunker()

    raw_chunks = list(chunker.chunk(doc))
    logger.info("Chunked document into %d chunks", len(raw_chunks))

    chunks = []
    for idx, chunk in enumerate(raw_chunks):
        # Extract text content
        text = chunk.text if hasattr(chunk, "text") else str(chunk)

        # Extract heading path for context
        headings = []
        if hasattr(chunk, "headings"):
            headings = chunk.headings if chunk.headings else []

        # Extract provenance (page number, bbox)
        page_number = None
        bbox = None
        if hasattr(chunk, "prov") and chunk.prov:
            prov = chunk.prov[0] if isinstance(chunk.prov, list) else chunk.prov
            page_number = getattr(prov, "page_no", None)
            if hasattr(prov, "bbox"):
                bbox_obj = prov.bbox
                bbox = {
                    "x0": getattr(bbox_obj, "l", 0),
                    "y0": getattr(bbox_obj, "t", 0),
                    "x1": getattr(bbox_obj, "r", 0),
                    "y1": getattr(bbox_obj, "b", 0),
                    "page": page_number,
                }

        chunk_dict = {
            "chunk_index": idx,
            "content": text,
            "content_preview": text[:200] if text else "",
            "metadata": {
                "page_number": page_number,
                "section": headings[-1] if headings else None,
                "heading": headings[-1] if headings else None,
                "heading_path": headings,
                "bbox": bbox,
            },
        }
        chunks.append(chunk_dict)

    return chunks
