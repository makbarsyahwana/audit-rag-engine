"""Prompt templates for compliance workflow drafting (gap analysis + finding)."""

COMPLIANCE_GAP_SYSTEM_PROMPT = """You are an expert compliance gap analyst. \
Your role is to draft structured gap analysis reports based on retrieved \
regulatory documents, policies, and compliance evidence.

Rules:
1. ONLY use information from the provided context — never fabricate evidence.
2. Every claim MUST be backed by a citation in the format [CITE:chunk_id].
3. Follow the standard gap analysis structure: Requirement, Current State, \
Gap Assessment, Risk Rating, and Remediation Plan.
4. Be precise, formal, and compliance-grade — accuracy over creativity.
5. If the context does not contain enough information for a section, mark it as \
"[INSUFFICIENT EVIDENCE — requires further assessment]".
6. Use professional compliance language consistent with regulatory standards.

At the end, provide a CONFIDENCE score (0.0 to 1.0) indicating how well \
the context supports the drafted gap analysis:
- 1.0 = fully supported, comprehensive evidence
- 0.7-0.9 = well supported with minor gaps
- 0.4-0.6 = partially supported, some sections need more evidence
- 0.1-0.3 = weakly supported, significant gaps
- 0.0 = insufficient evidence

Format your confidence as: CONFIDENCE: 0.X
"""

COMPLIANCE_GAP_USER_PROMPT = """Context (retrieved compliance documents):
{context}

Gap Analysis Details:
- Title: {title}
- Regulatory Framework: {framework}
- Scope: {scope}
- Control/Process Under Review: {control_ref}

Draft a complete compliance gap analysis covering:
1. **Requirement** — Applicable regulatory or policy requirement
2. **Current State** — How the organization currently addresses this requirement
3. **Gap Assessment** — Identified gaps between requirement and current state
4. **Risk Rating** — Impact and likelihood (CRITICAL / HIGH / MEDIUM / LOW)
5. **Remediation Plan** — Actionable steps to close the gap
6. **Timeline** — Recommended implementation timeline

Cite all sources using [CITE:chunk_id] format."""

COMPLIANCE_FINDING_SYSTEM_PROMPT = """You are an expert compliance finding writer. \
Your role is to draft structured compliance findings based on retrieved \
evidence and regulatory context.

Rules:
1. ONLY use information from the provided context — never fabricate evidence.
2. Every claim MUST be backed by a citation in the format [CITE:chunk_id].
3. Follow the standard compliance finding structure: Requirement, Observation, \
Root Cause, Impact, Recommendation.
4. Be precise, objective, and compliance-grade — state facts, not opinions.
5. If the context does not contain enough information for a section, mark it as \
"[INSUFFICIENT EVIDENCE — requires further investigation]".
6. Severity should be justified by regulatory risk and potential impact.
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

COMPLIANCE_FINDING_USER_PROMPT = """Context (retrieved compliance evidence):
{context}

Finding Details:
- Title: {title}
- Regulatory Requirement: {requirement_ref}
- Observation Summary: {observation}

Draft a complete compliance finding covering:
1. **Requirement** — The regulatory or policy requirement that applies
2. **Observation** — Current state observed through assessment
3. **Root Cause** — Why the gap exists
4. **Impact** — Regulatory risk and potential consequences
5. **Recommendation** — Actionable remediation steps
6. **Suggested Severity** — CRITICAL / HIGH / MEDIUM / LOW with justification

Cite all sources using [CITE:chunk_id] format."""


def build_compliance_gap_messages(
    chunks: list[dict],
    title: str,
    framework: str = "",
    scope: str = "",
    control_ref: str = "",
) -> list[dict[str, str]]:
    """Build the message list for compliance gap analysis drafting.

    Args:
        chunks: Retrieved chunk dicts.
        title: Gap analysis title.
        framework: Regulatory framework being assessed.
        scope: Assessment scope description.
        control_ref: Control or process reference.

    Returns:
        List of message dicts for LLM invocation.
    """
    from src.generation.prompts.qa import format_context

    context = format_context(chunks)
    return [
        {"role": "system", "content": COMPLIANCE_GAP_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": COMPLIANCE_GAP_USER_PROMPT.format(
                context=context,
                title=title,
                framework=framework or "To be determined",
                scope=scope or "To be determined",
                control_ref=control_ref or "N/A",
            ),
        },
    ]


def build_compliance_finding_messages(
    chunks: list[dict],
    title: str,
    requirement_ref: str = "",
    observation: str = "",
) -> list[dict[str, str]]:
    """Build the message list for compliance finding drafting.

    Args:
        chunks: Retrieved chunk dicts.
        title: Finding title.
        requirement_ref: Related regulatory requirement reference.
        observation: Brief observation or issue summary.

    Returns:
        List of message dicts for LLM invocation.
    """
    from src.generation.prompts.qa import format_context

    context = format_context(chunks)
    return [
        {"role": "system", "content": COMPLIANCE_FINDING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": COMPLIANCE_FINDING_USER_PROMPT.format(
                context=context,
                title=title,
                requirement_ref=requirement_ref or "N/A",
                observation=observation
                or "To be described based on evidence",
            ),
        },
    ]
