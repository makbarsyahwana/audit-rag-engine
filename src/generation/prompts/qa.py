"""Q&A prompt templates with citation format instructions."""

QA_SYSTEM_PROMPT = """You are an expert audit assistant. Your role is to answer questions \
about audit documents, standards, controls, and compliance topics accurately and concisely.

Rules:
1. ONLY use information from the provided context to answer the question.
2. Every claim MUST be backed by a citation in the format [CITE:chunk_id].
3. If the context does not contain enough information to answer, say: \
"I don't have sufficient evidence in the provided documents to answer this question."
4. Do NOT make up or infer information not present in the context.
5. Prefer quoting relevant clauses, paragraphs, or control descriptions directly.
6. Be precise and audit-grade — accuracy over creativity.
7. Structure your answer clearly with bullet points or numbered lists when appropriate.

At the end of your answer, provide a CONFIDENCE score (0.0 to 1.0) indicating how well \
the provided context supports your answer:
- 1.0 = directly and fully supported
- 0.7-0.9 = well supported with minor gaps
- 0.4-0.6 = partially supported
- 0.1-0.3 = weakly supported, mostly inference
- 0.0 = no supporting evidence

Format your confidence as: CONFIDENCE: 0.X
"""

QA_USER_PROMPT = """Context (retrieved documents):
{context}

Question: {query}

Answer the question using ONLY the context above. Cite sources using [CITE:chunk_id] format."""


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context string for the prompt.

    Args:
        chunks: List of chunk dicts with chunk_id, content, document_name, page_number.

    Returns:
        Formatted context string.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        chunk_id = chunk.get("chunk_id", f"chunk_{i}")
        doc_name = chunk.get("document_name", "Unknown")
        page = chunk.get("page_number")
        content = chunk.get("content", "")

        header = f"[Source {i}] (chunk_id: {chunk_id}, document: {doc_name}"
        if page:
            header += f", page: {page}"
        header += ")"

        parts = [f"{header}\n{content}"]

        # Include entity context if available
        entities = chunk.get("entities", [])
        if entities:
            parts.append(f"Entities: {', '.join(entities)}")

        related = chunk.get("related_entities", [])
        if related:
            parts.append(f"Related entities: {', '.join(related)}")

        context_parts.append("\n".join(parts))

    return "\n\n---\n\n".join(context_parts)


def build_qa_messages(query: str, chunks: list[dict]) -> list[dict[str, str]]:
    """Build the full message list for Q&A generation.

    Args:
        query: The user's question.
        chunks: Retrieved chunk dicts.

    Returns:
        List of message dicts for LLM invocation.
    """
    context = format_context(chunks)
    return [
        {"role": "system", "content": QA_SYSTEM_PROMPT},
        {"role": "user", "content": QA_USER_PROMPT.format(context=context, query=query)},
    ]
