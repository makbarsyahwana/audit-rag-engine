"""Prompt templates for legal workflow drafting (memo + issue analysis)."""

LEGAL_MEMO_SYSTEM_PROMPT = """You are an expert legal memorandum writer. \
Your role is to draft structured legal memoranda based on retrieved \
legal documents, contracts, case law, and regulatory context.

Rules:
1. ONLY use information from the provided context — never fabricate evidence.
2. Every claim MUST be backed by a citation in the format [CITE:chunk_id].
3. Follow the standard legal memo structure: Issue, Rule, Application, Conclusion.
4. Be precise, formal, and legal-grade — accuracy over creativity.
5. If the context does not contain enough information for a section, mark it as \
"[INSUFFICIENT EVIDENCE — requires further research]".
6. Use professional legal language consistent with legal practice standards.

At the end, provide a CONFIDENCE score (0.0 to 1.0) indicating how well \
the context supports the drafted memorandum:
- 1.0 = fully supported, comprehensive evidence
- 0.7-0.9 = well supported with minor gaps
- 0.4-0.6 = partially supported, some sections need more evidence
- 0.1-0.3 = weakly supported, significant gaps
- 0.0 = insufficient evidence

Format your confidence as: CONFIDENCE: 0.X
"""

LEGAL_MEMO_USER_PROMPT = """Context (retrieved legal documents):
{context}

Memorandum Details:
- Title: {title}
- Issue: {issue}
- Jurisdiction: {jurisdiction}
- Matter Reference: {matter_ref}

Draft a complete legal memorandum covering:
1. **Issue** — The legal question(s) to be addressed
2. **Rule** — Applicable law, regulations, or contractual provisions
3. **Application** — Analysis of how the rules apply to the facts
4. **Conclusion** — Legal opinion and recommended course of action

Cite all sources using [CITE:chunk_id] format."""

LEGAL_ISSUE_SYSTEM_PROMPT = """You are an expert legal issue analyst. \
Your role is to draft structured legal issue analyses based on \
retrieved evidence and legal context.

Rules:
1. ONLY use information from the provided context — never fabricate evidence.
2. Every claim MUST be backed by a citation in the format [CITE:chunk_id].
3. Follow the standard issue analysis structure: Issue, Facts, Analysis, \
Risk Assessment, Recommendation.
4. Be precise, objective, and legal-grade — state facts, not opinions.
5. If the context does not contain enough information for a section, mark it as \
"[INSUFFICIENT EVIDENCE — requires further investigation]".
6. Risk should be justified by the gap between legal requirements and current state.
7. Recommendations should be actionable, specific, and proportionate to the risk.

At the end, provide a CONFIDENCE score (0.0 to 1.0) indicating how well \
the context supports the drafted analysis:
- 1.0 = fully supported with clear evidence
- 0.7-0.9 = well supported, minor details to verify
- 0.4-0.6 = partially supported, some assertions need validation
- 0.1-0.3 = weakly supported, mostly inferred
- 0.0 = insufficient evidence to support an analysis

Format your confidence as: CONFIDENCE: 0.X
"""

LEGAL_ISSUE_USER_PROMPT = """Context (retrieved legal documents):
{context}

Issue Details:
- Title: {title}
- Related Provision: {provision_ref}
- Observation Summary: {observation}

Draft a complete legal issue analysis covering:
1. **Issue** — The legal concern identified
2. **Facts** — Relevant facts established from the evidence
3. **Analysis** — Legal reasoning and application of law to facts
4. **Risk Assessment** — Potential exposure and likelihood (HIGH / MEDIUM / LOW)
5. **Recommendation** — Actionable remediation or mitigation steps

Cite all sources using [CITE:chunk_id] format."""


def build_legal_memo_messages(
    chunks: list[dict],
    title: str,
    issue: str = "",
    jurisdiction: str = "",
    matter_ref: str = "",
) -> list[dict[str, str]]:
    """Build the message list for legal memorandum drafting.

    Args:
        chunks: Retrieved chunk dicts.
        title: Memo title.
        issue: Legal issue description.
        jurisdiction: Applicable jurisdiction.
        matter_ref: Matter or case reference.

    Returns:
        List of message dicts for LLM invocation.
    """
    from src.generation.prompts.qa import format_context

    context = format_context(chunks)
    return [
        {"role": "system", "content": LEGAL_MEMO_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": LEGAL_MEMO_USER_PROMPT.format(
                context=context,
                title=title,
                issue=issue or "To be determined based on evidence",
                jurisdiction=jurisdiction or "To be determined",
                matter_ref=matter_ref or "N/A",
            ),
        },
    ]


def build_legal_issue_messages(
    chunks: list[dict],
    title: str,
    provision_ref: str = "",
    observation: str = "",
) -> list[dict[str, str]]:
    """Build the message list for legal issue analysis drafting.

    Args:
        chunks: Retrieved chunk dicts.
        title: Issue title.
        provision_ref: Related legal provision or contract clause.
        observation: Brief observation or issue summary.

    Returns:
        List of message dicts for LLM invocation.
    """
    from src.generation.prompts.qa import format_context

    context = format_context(chunks)
    return [
        {"role": "system", "content": LEGAL_ISSUE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": LEGAL_ISSUE_USER_PROMPT.format(
                context=context,
                title=title,
                provision_ref=provision_ref or "N/A",
                observation=observation
                or "To be described based on evidence",
            ),
        },
    ]
