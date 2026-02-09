"""Server-side HITL approval gate (ASI09 — Human-Agent Trust).

Stores high-risk LLM responses as pending approval in MongoDB
before delivering them to the user.
"""

import logging
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


# Output types that require human approval before delivery
HIGH_RISK_OUTPUT_TYPES = frozenset({
    "finding",
    "workpaper",
    "report",
    "regulatory_response",
})


def requires_approval(output_type: str) -> bool:
    """Check if an output type requires human approval."""
    return output_type.lower() in HIGH_RISK_OUTPUT_TYPES


async def create_pending_approval(
    document_store: Any,
    response_text: str,
    query: str,
    engagement_id: str,
    output_type: str,
    user_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> str:
    """Store a response as pending approval.

    Returns the approval_id for tracking.
    """
    import uuid

    approval_id = str(uuid.uuid4())
    record = {
        "approval_id": approval_id,
        "status": ApprovalStatus.PENDING.value,
        "query": query,
        "response_text": response_text,
        "engagement_id": engagement_id,
        "output_type": output_type,
        "user_id": user_id,
        "metadata": metadata or {},
        "created_at": datetime.now(UTC),
        "reviewed_at": None,
        "reviewer_id": None,
        "review_comment": None,
    }

    await document_store.db["pending_approvals"].insert_one(record)
    logger.info(
        "Created pending approval %s for %s output",
        approval_id,
        output_type,
    )
    return approval_id


async def get_approval(
    document_store: Any, approval_id: str
) -> Optional[dict]:
    """Get an approval record by ID."""
    return await document_store.db["pending_approvals"].find_one(
        {"approval_id": approval_id}
    )


async def update_approval(
    document_store: Any,
    approval_id: str,
    status: ApprovalStatus,
    reviewer_id: str,
    comment: str = "",
) -> bool:
    """Approve or reject a pending response."""
    result = await document_store.db["pending_approvals"].update_one(
        {"approval_id": approval_id},
        {
            "$set": {
                "status": status.value,
                "reviewer_id": reviewer_id,
                "review_comment": comment,
                "reviewed_at": datetime.now(UTC),
            }
        },
    )
    if result.modified_count > 0:
        logger.info(
            "Approval %s updated to %s by %s",
            approval_id,
            status.value,
            reviewer_id,
        )
        return True
    return False
