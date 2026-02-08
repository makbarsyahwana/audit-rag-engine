"""Unit tests for the query router."""

from src.models.retrieval import RetrievalMode
from src.retrieval.router import classify_query


def test_fulltext_clause_reference():
    """Exact clause references should route to fulltext."""
    assert classify_query("ISO 27001 A.12.4.1") == RetrievalMode.FULLTEXT


def test_fulltext_control_id():
    """Control IDs should route to fulltext."""
    assert classify_query("What is CHG-01?") == RetrievalMode.FULLTEXT


def test_graph_ownership_query():
    """Ownership queries should route to graph."""
    assert classify_query("Who owns the change management process?") == RetrievalMode.GRAPH


def test_graph_relationship_query():
    """Fulltext + graph signals → graph_vector_fulltext."""
    result = classify_query("What controls implement clause 6.1.2?")
    assert result == RetrievalMode.GRAPH_VECTOR_FULLTEXT


def test_vector_conceptual_query():
    """Conceptual queries should route to vector."""
    result = classify_query("Explain the risk assessment process for IT audits")
    assert result == RetrievalMode.VECTOR


def test_hybrid_short_query():
    """Short ambiguous queries should route to hybrid."""
    assert classify_query("access controls") == RetrievalMode.HYBRID


def test_hybrid_mixed_signals():
    """Queries with both fulltext and graph signals → graph_vector_fulltext."""
    result = classify_query("Who owns CHG-01 and what does it implement?")
    assert result == RetrievalMode.GRAPH_VECTOR_FULLTEXT
