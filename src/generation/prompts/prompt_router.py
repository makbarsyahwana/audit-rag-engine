"""Prompt router — selects the correct prompt builder based on application mode."""

from typing import Callable


def get_qa_builder(mode: str) -> Callable[[str, list[dict]], list[dict[str, str]]]:
    """Return the QA message builder function for the given application mode.

    Args:
        mode: Application mode — "audit", "legal", or "compliance".

    Returns:
        A function with signature (query, chunks) -> list[dict[str, str]].
    """
    if mode == "legal":
        from src.generation.prompts.legal_qa import build_legal_qa_messages

        return build_legal_qa_messages
    elif mode == "compliance":
        from src.generation.prompts.compliance_qa import build_compliance_qa_messages

        return build_compliance_qa_messages
    else:
        from src.generation.prompts.qa import build_qa_messages

        return build_qa_messages


def get_workflow_builders(
    mode: str,
) -> dict[str, Callable[..., list[dict[str, str]]]]:
    """Return workflow prompt builders (finding + workpaper equivalents) for the mode.

    Args:
        mode: Application mode — "audit", "legal", or "compliance".

    Returns:
        Dict with keys "finding" and "workpaper", each mapping to a builder function.
    """
    if mode == "legal":
        from src.generation.prompts.legal_workflow import (
            build_legal_issue_messages,
            build_legal_memo_messages,
        )

        return {"finding": build_legal_issue_messages, "workpaper": build_legal_memo_messages}
    elif mode == "compliance":
        from src.generation.prompts.compliance_workflow import (
            build_compliance_finding_messages,
            build_compliance_gap_messages,
        )

        return {
            "finding": build_compliance_finding_messages,
            "workpaper": build_compliance_gap_messages,
        }
    else:
        from src.generation.prompts.finding import build_finding_messages
        from src.generation.prompts.workpaper import build_workpaper_messages

        return {"finding": build_finding_messages, "workpaper": build_workpaper_messages}
