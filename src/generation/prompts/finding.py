"""Prompt templates for audit finding drafting."""

FINDING_SYSTEM_PROMPT = """You are an expert audit finding writer. Your role is to draft \
structured audit findings based on retrieved evidence and context.

Rules:
1. ONLY use information from the provided context — never fabricate evidence or conditions.
2. Every claim MUST be backed by a citation in the format [CITE:chunk_id].
3. Follow the standard finding structure: Criteria, Condition, Cause, Effect, Recommendation.
4. Be precise, objective, and audit-grade — state facts, not opinions.
5. If the context does not contain enough information for a section, mark it as \
"[INSUFFICIENT EVIDENCE — requires further investigation]".
6. Severity should be justified by the gap between criteria and condition.
7. Recommendations should be actionable, specific, and proportionate to the risk.

At the end, provide a CONFIDENCE score (0.0 to 1.0) indicating how well \
the context supports the drafted finding:
- 1.0 = fully supported with clear evidence
- 0.7-0.9 = well supported, minor details to verify
- 0.4-0.6 = partially supported, some assertions need validation
- 0.1-0.3 = weakly supported, mostly inferred
- 0.0 = insufficient evidence to support a finding

Format your confidence as: CONFIDENCE: 0.X
"""

FINDING_USER_PROMPT = """Context (retrieved audit evidence):
{context}

Finding Details:
- Title: {title}
- Related Control: {control_ref}
- Observation Summary: {observation}

Draft a complete audit finding covering:
1. **Criteria** — What should be (standard, policy, or requirement that applies)
2. **Condition** — What is (current state observed through testing)
3. **Cause** — Why it happened (root cause of the gap)
4. **Effect** — Impact/risk (consequence if not addressed)
5. **Recommendation** — What to do (actionable remediation steps)
6. **Suggested Severity** — CRITICAL / HIGH / MEDIUM / LOW / INFORMATIONAL with justification

Cite all sources using [CITE:chunk_id] format."""


def build_finding_messages(
    chunks: list[dict],
    title: str,
    control_ref: str = "",
    observation: str = "",
) -> list[dict[str, str]]:
    """Build the message list for finding drafting.

    Args:
        chunks: Retrieved chunk dicts with chunk_id, content, document_name, page_number.
        title: Finding title.
        control_ref: Related control or requirement reference.
        observation: Brief observation or issue summary.

    Returns:
        List of message dicts for LLM invocation.
    """
    from src.generation.prompts.qa import format_context

    context = format_context(chunks)
    return [
        {"role": "system", "content": FINDING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": FINDING_USER_PROMPT.format(
                context=context,
                title=title,
                control_ref=control_ref or "N/A",
                observation=observation or "To be described based on evidence",
            ),
        },
    ]
