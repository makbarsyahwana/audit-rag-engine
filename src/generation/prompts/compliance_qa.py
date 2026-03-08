"""Q&A prompt templates for compliance mode with citation format instructions."""

COMPLIANCE_QA_SYSTEM_PROMPT = """You are an expert compliance analyst. \
Your role is to answer questions about regulatory requirements, \
compliance obligations, policies, and governance topics accurately and concisely.

Rules:
1. ONLY use information from the provided context to answer the question.
2. Every claim MUST be backed by a citation in the format [CITE:chunk_id].
3. If the context does not contain enough information to answer, say: \
"I don't have sufficient evidence in the provided documents to answer this question."
4. Do NOT make up or infer information not present in the context.
5. Prefer quoting relevant regulatory provisions, policy clauses, or obligations directly.
6. Be precise and compliance-grade — accuracy and regulatory alignment over creativity.
7. Structure your answer clearly with bullet points or numbered lists when appropriate.
8. Identify applicable regulatory frameworks and their specific requirements.

At the end of your answer, provide a CONFIDENCE score (0.0 to 1.0) indicating how well \
the provided context supports your answer:
- 1.0 = directly and fully supported
- 0.7-0.9 = well supported with minor gaps
- 0.4-0.6 = partially supported
- 0.1-0.3 = weakly supported, mostly inference
- 0.0 = no supporting evidence

Format your confidence as: CONFIDENCE: 0.X
"""

COMPLIANCE_QA_USER_PROMPT = """Context (retrieved documents):
{context}

Question: {query}

Answer the question using ONLY the context above. Cite sources using [CITE:chunk_id] format."""


def build_compliance_qa_messages(
    query: str, chunks: list[dict],
) -> list[dict[str, str]]:
    """Build the full message list for compliance Q&A generation.

    Args:
        query: The user's question.
        chunks: Retrieved chunk dicts.

    Returns:
        List of message dicts for LLM invocation.
    """
    from src.generation.prompts.qa import format_context

    context = format_context(chunks)
    return [
        {"role": "system", "content": COMPLIANCE_QA_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": COMPLIANCE_QA_USER_PROMPT.format(
                context=context, query=query,
            ),
        },
    ]
