from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict


class ResumeDraftStatus(str, Enum):
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"


class ResumeDraftRecord(BaseModel):
    """A saved tailored-resume draft. `draft` holds the TailoredResumeDraft fields."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    profile_id: str
    job_description: str
    status: ResumeDraftStatus = ResumeDraftStatus.PENDING_REVIEW
    draft: dict
    created_at: datetime | None = None
    reviewed_at: datetime | None = None
