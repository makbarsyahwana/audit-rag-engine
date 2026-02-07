"""Query router: classify query → select retrieval mode."""

import logging
import re

from src.models.retrieval import RetrievalMode

logger = logging.getLogger(__name__)

# Patterns that suggest fulltext (exact references, clause IDs)
FULLTEXT_PATTERNS = [
    r"\b[A-Z]{2,}\s*[\d.]+",          # e.g., ISO 27001, NIST 800-53
    r"\bA\.\d+\.\d+",                  # e.g., A.12.4.1
    r"\b[A-Z]{2,}-\d+",               # e.g., CHG-01, AC-02
    r"\bclause\s+\d+",                 # e.g., clause 6.1.2
    r"\barticle\s+\d+",                # e.g., article 32
    r"\bsection\s+\d+",               # e.g., section 4.2
]

# Patterns that suggest graph (entity relationship queries)
GRAPH_PATTERNS = [
    r"\bwho\s+(owns|manages|is responsible)",
    r"\bwhat\s+(controls?|systems?)\s+(does|do)",
    r"\brelat(ed|ionship|es)\s+to",
    r"\bconnect(ed|ion|s)\s+to",
    r"\bmap(ped|s|ping)\s+to",
    r"\bimplements?\b",
    r"\bown(ed|er|s)\b.*\b(control|process|system)",
]

FULLTEXT_RE = [re.compile(p, re.IGNORECASE) for p in FULLTEXT_PATTERNS]
GRAPH_RE = [re.compile(p, re.IGNORECASE) for p in GRAPH_PATTERNS]


def classify_query(query: str) -> RetrievalMode:
    """Classify a user query and select the best retrieval mode.

    Classification logic:
    - Exact reference / clause ID patterns → fulltext
    - Entity relationship patterns → graph
    - Mixed signals → hybrid
    - Default (conceptual / explain) → vector

    Args:
        query: The user's query text.

    Returns:
        The recommended RetrievalMode.
    """
    has_fulltext_signal = any(p.search(query) for p in FULLTEXT_RE)
    has_graph_signal = any(p.search(query) for p in GRAPH_RE)

    if has_fulltext_signal and has_graph_signal:
        mode = RetrievalMode.HYBRID
    elif has_fulltext_signal:
        mode = RetrievalMode.FULLTEXT
    elif has_graph_signal:
        mode = RetrievalMode.GRAPH
    else:
        # Default: use vector for conceptual queries,
        # but short queries might benefit from hybrid
        if len(query.split()) <= 3:
            mode = RetrievalMode.HYBRID
        else:
            mode = RetrievalMode.VECTOR

    logger.info(
        "Query classified: mode=%s (fulltext=%s, graph=%s) — '%s'",
        mode.value,
        has_fulltext_signal,
        has_graph_signal,
        query[:80],
    )
    return mode
