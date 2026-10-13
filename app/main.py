from datetime import datetime, timezone
import logging
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.db import models as db
from app.models.company import Company, CompanyCreate
from app.models.email import Email, EmailStatus
from app.models.profile import ExtractedProfile, Profile, ProfileCreate
from app.models.recipient import Recipient, RecipientCreate
from app.services import email_service, outreach_agent, resume_service
from fastapi.middleware.cors import CORSMiddleware

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
    recipient = Recipient(**recipient_dict)

    # Insert draft entry
    row = db.insert(
        db.EMAILS,
        {
            "profile_id": req.profile_id,
            "company_id": req.company_id,
            "recipient_id": req.recipient_id,
            "status": EmailStatus.GENERATING,
        },
    )
    email_id = row["id"]

    try:
        draft = agent.run(profile, company, recipient)
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
