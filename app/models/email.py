from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict


class EmailStatus(str, Enum):
    DRAFT = "draft"
    GENERATING = "generating"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    SCHEDULED = "scheduled"
    SENDING = "sending"
    SENT = "sent"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EmailUpdate(BaseModel):
    subject: str | None = None
    body: str | None = None


class Email(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    profile_id: str
    company_id: str
    recipient_id: str
    subject: str | None = None
    body: str | None = None
    status: EmailStatus = EmailStatus.DRAFT
    scheduled_at: datetime | None = None
    sent_at: datetime | None = None
    error: str | None = None
    resume_draft_id: str | None = None
