"""Prompt templates for workpaper narrative drafting."""

WORKPAPER_SYSTEM_PROMPT = """You are an expert audit workpaper writer. Your role is to draft \
structured workpaper narratives based on retrieved audit evidence and context.

Rules:
1. ONLY use information from the provided context — never fabricate evidence.
2. Every claim MUST be backed by a citation in the format [CITE:chunk_id].
3. Follow the standard workpaper structure: Objective, Scope, Criteria, Testing Performed, \
Results, and Conclusion.
4. Be precise, formal, and audit-grade — accuracy over creativity.
5. If the context does not contain enough information for a section, mark it as \
"[INSUFFICIENT EVIDENCE — requires further testing]".
6. Use professional audit language consistent with IIA/ISACA standards.

At the end, provide a CONFIDENCE score (0.0 to 1.0) indicating how well \
the context supports the drafted workpaper:
- 1.0 = fully supported, comprehensive evidence
- 0.7-0.9 = well supported with minor gaps
- 0.4-0.6 = partially supported, some sections need more evidence
- 0.1-0.3 = weakly supported, significant gaps
- 0.0 = insufficient evidence

Format your confidence as: CONFIDENCE: 0.X
"""

WORKPAPER_USER_PROMPT = """Context (retrieved audit evidence):
{context}

Workpaper Details:
- Title: {title}
- Objective: {objective}
- Scope: {scope}
- Control/Process Under Review: {control_ref}

Draft a complete workpaper narrative covering:
1. **Objective** — What we are testing and why
2. **Scope** — Period, systems, entities covered
3. **Criteria** — Applicable standards, policies, or control requirements
4. **Testing Performed** — Procedures executed and evidence examined
5. **Results** — Findings from the testing, including any exceptions
6. **Conclusion** — Overall assessment and opinion

Cite all sources using [CITE:chunk_id] format."""


def build_workpaper_messages(
    chunks: list[dict],
    title: str,
    objective: str = "",
    scope: str = "",
    control_ref: str = "",
) -> list[dict[str, str]]:
    """Build the message list for workpaper drafting.

    Args:
        chunks: Retrieved chunk dicts with chunk_id, content, document_name, page_number.
        title: Workpaper title.
        objective: Audit objective description.
        scope: Audit scope description.
        control_ref: Control or process reference ID.

    Returns:
        List of message dicts for LLM invocation.
    """
    from src.generation.prompts.qa import format_context

    context = format_context(chunks)
    return [
        {"role": "system", "content": WORKPAPER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": WORKPAPER_USER_PROMPT.format(
                context=context,
                title=title,
                objective=objective or "To be determined based on evidence",
                scope=scope or "To be determined",
                control_ref=control_ref or "N/A",
            ),
        },
    ]
