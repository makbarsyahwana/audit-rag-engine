"""Unit tests for citation extraction."""

from src.generation.citations import check_abstention, extract_citations
from src.models.retrieval import RetrievedChunk


def _make_chunk(chunk_id: str, content: str = "test content") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        content=content,
        score=0.9,
        document_id="doc-1",
        document_name="Policy.pdf",
        page_number=5,
    )


def test_extract_single_citation():
    """Should extract a single [CITE:...] reference."""
    chunks = [_make_chunk("aaa-111")]
    response = "The policy states X [CITE:aaa-111]. CONFIDENCE: 0.85"
    citations, confidence, cleaned = extract_citations(response, chunks)

    assert len(citations) == 1
    assert citations[0].chunk_id == "aaa-111"
    assert citations[0].document_name == "Policy.pdf"
    assert confidence == 0.85
    assert "CONFIDENCE" not in cleaned


def test_extract_multiple_citations():
    """Should extract multiple unique citations."""
    chunks = [_make_chunk("aaa-111"), _make_chunk("bbb-222")]
    response = "First [CITE:aaa-111] and second [CITE:bbb-222]. CONFIDENCE: 0.7"
    citations, confidence, _ = extract_citations(response, chunks)

    assert len(citations) == 2
    assert confidence == 0.7


def test_deduplicate_citations():
    """Same chunk cited twice should only appear once."""
    chunks = [_make_chunk("aaa-111")]
    response = "One [CITE:aaa-111] two [CITE:aaa-111]. CONFIDENCE: 0.9"
    citations, _, _ = extract_citations(response, chunks)

    assert len(citations) == 1


def test_unknown_citation_skipped():
    """Citation referencing a non-retrieved chunk is skipped."""
    chunks = [_make_chunk("aaa-111")]
    response = "Ref [CITE:unknown-999]. CONFIDENCE: 0.5"
    citations, _, _ = extract_citations(response, chunks)

    assert len(citations) == 0


def test_no_confidence():
    """Missing CONFIDENCE line → 0.0."""
    chunks = [_make_chunk("aaa-111")]
    response = "The answer [CITE:aaa-111]."
    _, confidence, _ = extract_citations(response, chunks)

    assert confidence == 0.0


def test_abstention_detected():
    """Should detect abstention phrases."""
    assert check_abstention("I don't have sufficient evidence to answer.") is True
    assert check_abstention("The policy clearly states...") is False


def test_abstention_insufficient():
    assert check_abstention("There is insufficient evidence in the documents.") is True


def test_abstention_normal_answer():
    assert check_abstention("Control CHG-01 implements clause A.12.1.") is False
