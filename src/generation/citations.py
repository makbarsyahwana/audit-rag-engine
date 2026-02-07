"""Citation extraction from LLM response."""

import logging
import re

from src.models.retrieval import Citation, RetrievedChunk

logger = logging.getLogger(__name__)

# Pattern to match [CITE:chunk_id] in LLM output
CITE_PATTERN = re.compile(r"\[CITE:([a-f0-9\-]+)\]")

# Pattern to extract CONFIDENCE: 0.X from LLM output
CONFIDENCE_PATTERN = re.compile(r"CONFIDENCE:\s*([\d.]+)", re.IGNORECASE)


def extract_citations(
    llm_response: str,
    retrieved_chunks: list[RetrievedChunk],
) -> tuple[list[Citation], float, str]:
    """Extract citations and confidence from LLM response.

    Args:
        llm_response: Raw LLM response text.
        retrieved_chunks: The chunks that were provided as context.

    Returns:
        Tuple of (citations, confidence, cleaned_answer).
    """
    # Build chunk lookup
    chunk_map: dict[str, RetrievedChunk] = {c.chunk_id: c for c in retrieved_chunks}

    # Extract cited chunk IDs
    cited_ids = CITE_PATTERN.findall(llm_response)
    seen_ids: set[str] = set()
    citations: list[Citation] = []

    for chunk_id in cited_ids:
        if chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)

        chunk = chunk_map.get(chunk_id)
        if chunk:
            citations.append(Citation(
                chunk_id=chunk_id,
                document_id=chunk.document_id,
                document_name=chunk.document_name,
                excerpt=chunk.content[:300],
                page=chunk.page_number,
                score=chunk.score,
            ))

    # Extract confidence score
    confidence = 0.0
    confidence_match = CONFIDENCE_PATTERN.search(llm_response)
    if confidence_match:
        try:
            confidence = min(1.0, max(0.0, float(confidence_match.group(1))))
        except ValueError:
            confidence = 0.0

    # Clean the answer: remove CONFIDENCE line
    cleaned = CONFIDENCE_PATTERN.sub("", llm_response).strip()

    logger.info(
        "Extracted %d citations, confidence=%.2f from LLM response",
        len(citations),
        confidence,
    )
    return citations, confidence, cleaned


def check_abstention(llm_response: str) -> bool:
    """Check if the LLM response indicates abstention (insufficient evidence).

    Args:
        llm_response: The LLM response text.

    Returns:
        True if the response indicates abstention.
    """
    abstention_phrases = [
        "don't have sufficient evidence",
        "insufficient evidence",
        "cannot answer",
        "no relevant information",
        "not enough information",
        "unable to find",
        "no supporting evidence",
    ]
    lower_response = llm_response.lower()
    return any(phrase in lower_response for phrase in abstention_phrases)
