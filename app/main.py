from pydantic import Field
import sys
from pathlib import Path

# Lets `python3 main.py` work even from inside app/: the `app.` imports below need the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime, timezone
import logging
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, ValidationError

from app.db import models as db
from app.models.company import Company, CompanyCreate
from app.models.email import Email, EmailStatus
from app.models.profile import ExtractedProfile, Profile, ProfileCreate
from app.models.recipient import Recipient, RecipientCreate
from app.models.resume_draft import ResumeDraftRecord, ResumeDraftStatus
from app.services import email_service, outreach_agent, resume_service
from fastapi.middleware.cors import CORSMiddleware
from app.agents.review import ReviewError, orchestrate
from app.agents.tailoring_decision import needs_tailoring
from app.agents.resume_tailor import ResumeTailorError,tailor_resume

logger = logging.getLogger(__name__)
MAX_RESUME_BYTES = 5 * 1024 * 1024

app = FastAPI(title="Founders Outreach")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)
agent = outreach_agent.OutreachAgent()


class OutreachRequest(BaseModel):
    profile_id: str
    company_id: str
    recipient_id: str
    resume_draft_id: str | None = None  # an APPROVED tailored resume to write from; omit to use the confirmed resume


def _get_entity(table: str, id_: str) -> dict[str, Any]:
    row = db.fetch(table, id_)
    if row is None:
        raise HTTPException(404, f"{table} {id_} not found")
    return row


def _validate_and_read_pdf(file: UploadFile) -> bytes:
    filename = (file.filename or "").lower()
    content_type = (file.content_type or "").lower()

    if not filename.endswith(".pdf") and "pdf" not in content_type:
        raise HTTPException(415, "Upload a PDF file")

    data = file.file.read(MAX_RESUME_BYTES + 1)
    if len(data) > MAX_RESUME_BYTES:
        raise HTTPException(413, "Resume must be 5 MB or smaller")
    if len(data) == 0:
        raise HTTPException(422, "Uploaded PDF file is empty")
    if not data.startswith(b"%PDF"):
        raise HTTPException(415, "Invalid PDF header: file is not a valid PDF")

    return data


# --- Resume & Profile Endpoints ---
@app.post("/profiles/resume", response_model=ExtractedProfile, status_code=200)
def analyze_resume(file: UploadFile = File(...)):
    """Upload PDF resume, extract text via PyMuPDF, and parse candidate profile with ChatOllama for review."""
    data = _validate_and_read_pdf(file)
    try:
        return resume_service.extract_and_analyze_resume(data)
    except resume_service.ResumeError as exc:
        raise HTTPException(422, str(exc))
    except resume_service.ResumeProcessingError as exc:
        raise HTTPException(502, str(exc))
    except Exception as exc:
        logger.exception("Unexpected error analyzing resume upload: %s", exc)
        raise HTTPException(500, "Internal server error during resume analysis")

@app.post("/profiles", response_model=Profile, status_code=201)
def create_profile(data: ProfileCreate):
    """Save confirmed candidate profile and build its FAISS vector retrieval index."""
    return resume_service.confirm_and_persist_profile(data)

@app.get("/profiles", response_model=list[Profile])
def list_profiles():
    return db.fetch_all(db.PROFILES)

@app.get("/profiles/{profile_id}", response_model=Profile)
def get_profile(profile_id: str):
    return _get_entity(db.PROFILES, profile_id)


# --- Company Endpoints ---
@app.post("/companies", response_model=Company, status_code=201)
def create_company(data: CompanyCreate):
    return db.insert(db.COMPANIES, data.model_dump())

@app.get("/companies/{company_id}", response_model=Company)
def get_company(company_id: str):
    return _get_entity(db.COMPANIES, company_id)


# --- Recipient Endpoints ---
@app.post("/recipients", response_model=Recipient, status_code=201)
def create_recipient(data: RecipientCreate):
    return db.insert(db.RECIPIENTS, data.model_dump())

@app.get("/recipients/{recipient_id}", response_model=Recipient)
def get_recipient(recipient_id: str):
    return _get_entity(db.RECIPIENTS, recipient_id)


# --- Outreach & Email Endpoints ---
@app.post("/outreach", response_model=Email, status_code=201)
def create_outreach(req: OutreachRequest):
    """Retrieve candidate evidence via RAG, generate cold email draft, and hold for human review."""
    profile_dict = _get_entity(db.PROFILES, req.profile_id)
    company_dict = _get_entity(db.COMPANIES, req.company_id)
    recipient_dict = _get_entity(db.RECIPIENTS, req.recipient_id)

    profile = Profile(**profile_dict)
    company = Company(**company_dict)
    try:
        recipient = Recipient(**recipient_dict)
    except ValidationError:
        # A recipient saved before addresses were validated: fail now, not after the email is approved.
        raise HTTPException(
            422,
            f"Recipient {req.recipient_id} has an invalid email address ({recipient_dict.get('email')!r}). "
            "Fix the recipient before generating outreach.",
        )

    tailored = None
    if req.resume_draft_id:
        draft_record = _get_resume_draft(req.resume_draft_id)
        if draft_record.profile_id != req.profile_id:
            raise HTTPException(409, "That tailored resume belongs to a different profile")
        if draft_record.status != ResumeDraftStatus.APPROVED:
            raise HTTPException(
                409, f"Only an approved tailored resume can be used (current: {draft_record.status.value})"
            )
        tailored = draft_record.draft

    # Insert draft entry
    row = db.insert(
        db.EMAILS,
        {
            "profile_id": req.profile_id,
            "company_id": req.company_id,
            "recipient_id": req.recipient_id,
            "status": EmailStatus.GENERATING,
            # Only sent when used, so outreach keeps working before the emails.resume_draft_id column exists.
            **({"resume_draft_id": req.resume_draft_id} if tailored else {}),
        },
    )
    email_id = row["id"]

    try:
        draft = agent.run(profile, company, recipient, tailored=tailored)
        updated = db.update(
            db.EMAILS,
            email_id,
            {
                "subject": draft.subject,
                "body": draft.body,
                "status": EmailStatus.PENDING_REVIEW,
                "error": None,
            },
        )
        return Email(**updated)
    except Exception as exc:
        logger.exception("Failed to generate outreach email %s: %s", email_id, exc)
        failed = db.update(
            db.EMAILS,
            email_id,
            {"status": EmailStatus.FAILED, "error": str(exc)},
        )
        return Email(**failed)

@app.get("/emails", response_model=list[Email])
def list_emails(status: EmailStatus | None = None):
    return db.fetch_all(db.EMAILS, **({"status": status} if status else {}))

@app.get("/emails/{email_id}", response_model=Email)
def get_email(email_id: str):
    return _get_entity(db.EMAILS, email_id)

@app.post("/emails/{email_id}/approve", response_model=Email)
def approve_and_send_email(email_id: str):
    """Approve draft and deliver via SMTP."""
    email_row = _get_entity(db.EMAILS, email_id)
    email = Email(**email_row)

    if email.status != EmailStatus.PENDING_REVIEW:
        raise HTTPException(409, f"Only emails pending review can be approved (current: {email.status.value})")

    profile_row = _get_entity(db.PROFILES, email.profile_id)
    recipient_row = _get_entity(db.RECIPIENTS, email.recipient_id)

    try:
        email_service.send_email(
            to_email=recipient_row["email"],
            subject=email.subject or "Outreach",
            body=email.body or "",
            from_email=profile_row.get("email"),
        )
        updated = db.update(
            db.EMAILS,
            email_id,
            {
                "status": EmailStatus.SENT,
                "sent_at": datetime.now(timezone.utc),
                "error": None,
            },
        )
        return Email(**updated)
    except Exception as exc:
        logger.exception("Failed to send email %s via SMTP: %s", email_id, exc)
        failed = db.update(
            db.EMAILS,
            email_id,
            {"status": EmailStatus.FAILED, "error": str(exc)},
        )
        return Email(**failed)

@app.post("/emails/{email_id}/reject", response_model=Email)
def reject_email(email_id: str):
    email_row = _get_entity(db.EMAILS, email_id)
    email = Email(**email_row)
    if email.status != EmailStatus.PENDING_REVIEW:
        raise HTTPException(409, f"Only emails pending review can be rejected (current: {email.status.value})")

    updated = db.update(db.EMAILS, email_id, {"status": EmailStatus.REJECTED})
    return Email(**updated)

class ProfileReviewRequest(BaseModel):
    # profile_id: str
    job_description: str

@app.post('/profiles/{profile_id}/review')
def review_profile(
    profile_id: str,
    data: ProfileReviewRequest,
):
    profile = db.fetch(db.PROFILES, profile_id)
    if not profile:
        raise HTTPException(
            status_code=404,
            detail="Profile not found",
        )
    
    try:
        result = orchestrate(
            profile_id=profile_id,
            profile=profile,
            job_description=data.job_description
        )
    except ReviewError as exc:
        raise HTTPException(502, str(exc))
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    result["tailoring"] = needs_tailoring(profile, data.job_description)
    return result

class ResumeTailorRequest(BaseModel):
    job_description: str

def _save_resume_draft(profile_id: str, job_description: str, draft: dict) -> str | None:
    """Save the generated draft so it can be reviewed. If it cannot be saved (e.g. the resume_drafts
    table has not been created yet), the draft is still returned, without an id."""
    try:
        row = db.insert(
            db.RESUME_DRAFTS,
            {
                "profile_id": profile_id,
                "job_description": job_description,
                "draft": draft,
                "status": ResumeDraftStatus.PENDING_REVIEW,
            },
        )
        return row["id"]
    except Exception as exc:
        logger.warning("Could not save the tailored resume draft (is the resume_drafts table created?): %s", exc)
        return None


@app.post("/profiles/{profile_id}/tailor-resume")
def tailor_profile_resume(profile_id: str,data: ResumeTailorRequest):
    profile = db.fetch(db.PROFILES,profile_id)
    if not profile: 
        raise HTTPException(status_code=404,
            detail="Profile not found")
    try:
        result = tailor_resume(
            profile_id=profile_id,profile=profile,job_description=data.job_description
        )
    except ResumeTailorError as exc:
        raise HTTPException(status_code=502,detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400,detail=str(exc))
    result["draft_id"] = _save_resume_draft(profile_id, data.job_description, result["draft"])
    return result


def _draft_db(call, *args, **kwargs):
    """Run a resume_drafts query; a missing table becomes a clear 503 instead of a raw database error."""
    try:
        return call(*args, **kwargs)
    except Exception as exc:
        if "PGRST205" in str(exc) or "resume_drafts" in str(exc):
            raise HTTPException(
                503,
                "The resume_drafts table does not exist yet. Run the resume_drafts section of "
                "supabase/schema.sql in the Supabase SQL editor.",
            ) from exc
        raise


def _get_resume_draft(draft_id: str) -> ResumeDraftRecord:
    row = _draft_db(db.fetch, db.RESUME_DRAFTS, draft_id)
    if row is None:
        raise HTTPException(404, f"Resume draft {draft_id} not found")
    return ResumeDraftRecord(**row)


@app.get("/profiles/{profile_id}/resume-drafts", response_model=list[ResumeDraftRecord])
def list_resume_drafts(profile_id: str):
    """A profile's tailored resume drafts, newest first."""
    rows = _draft_db(db.fetch_all, db.RESUME_DRAFTS, profile_id=profile_id)
    return sorted((ResumeDraftRecord(**r) for r in rows), key=lambda r: str(r.created_at or ""), reverse=True)


@app.get("/resume-drafts/{draft_id}", response_model=ResumeDraftRecord)
def get_resume_draft(draft_id: str):
    return _get_resume_draft(draft_id)


def _review_resume_draft(draft_id: str, status: ResumeDraftStatus) -> ResumeDraftRecord:
    record = _get_resume_draft(draft_id)
    if record.status != ResumeDraftStatus.PENDING_REVIEW:
        raise HTTPException(409, f"Only drafts pending review can be {status.value} (current: {record.status.value})")
    updated = _draft_db(
        db.update, db.RESUME_DRAFTS, draft_id, {"status": status, "reviewed_at": datetime.now(timezone.utc)}
    )
    return ResumeDraftRecord(**updated)


@app.post("/resume-drafts/{draft_id}/approve", response_model=ResumeDraftRecord)
def approve_resume_draft(draft_id: str):
    return _review_resume_draft(draft_id, ResumeDraftStatus.APPROVED)


@app.post("/resume-drafts/{draft_id}/reject", response_model=ResumeDraftRecord)
def reject_resume_draft(draft_id: str):
    return _review_resume_draft(draft_id, ResumeDraftStatus.REJECTED)

class ResumeDraftEdit(BaseModel):
    """The fields a user may change while reviewing a draft. A field that is not sent is left alone;
    a field sent as null or as blank text is cleared."""

    headline: str | None = Field(default=None, max_length=120)
    summary: str | None = Field(default=None, max_length=350)


@app.patch("/resume-drafts/{draft_id}", response_model=ResumeDraftRecord)
def edit_resume_draft(draft_id: str, data: ResumeDraftEdit):
    """Edit the headline and/or summary of a draft that is still pending review."""
    changes = data.model_dump(exclude_unset=True)
    if not changes:
        raise HTTPException(422, "Send at least one of: headline, summary")

    record = _get_resume_draft(draft_id)
    if record.status != ResumeDraftStatus.PENDING_REVIEW:
        raise HTTPException(409, f"Only drafts pending review can be edited (current: {record.status.value})")

    new_draft = {
        **record.draft,
        **{field: (value or "").strip() or None for field, value in changes.items()},
    }
    updated = _draft_db(db.update, db.RESUME_DRAFTS, draft_id, {"draft": new_draft})
    return ResumeDraftRecord(**updated)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
