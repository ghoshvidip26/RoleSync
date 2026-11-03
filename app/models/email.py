from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
    """What a reviewer may change on an email that is still pending review. A field that is not sent is
    left alone; a field that is sent must not be blank."""

    subject: str | None = Field(default=None, max_length=150)
    body: str | None = Field(default=None, max_length=8000)

    @field_validator("subject")
    @classmethod
    def _subject(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = " ".join(value.split())                 # a subject is one line
        if not value:
            raise ValueError("subject cannot be blank")
        return value

    @field_validator("body")
    @classmethod
    def _body(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.replace("\r\n", "\n").strip()
        if not value:
            raise ValueError("body cannot be blank")
        return value


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
