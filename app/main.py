from numpy import isin
from sqlalchemy import exc
from h11._abnf import status_code
from pydantic import Field
import sys
from pathlib import Path

# Lets `python3 main.py` work even from inside app/: the `app.` imports below need the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime, timezone
import logging
import re
import unicodedata
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, ValidationError, Field, field_validator
from typing import Optional

from app.db import models as db
from app.models.company import Company, CompanyCreate
from app.models.email import Email, EmailStatus, EmailUpdate
from app.models.profile import ExtractedProfile, Profile, ProfileCreate
from app.models.recipient import Recipient, RecipientCreate
from app.models.resume_draft import ResumeDraftRecord, ResumeDraftStatus
from app.services import email_service, outreach_agent, resume_service
from fastapi.middleware.cors import CORSMiddleware
from app.agents.review import ReviewError, orchestrate
from app.agents.role_recommender import recommend_roles
from app.agents.tailoring_decision import needs_tailoring
from app.agents.resume_tailor import ResumeTailorError,tailor_resume
from fastapi.responses import JSONResponse, Response
from app.services.resume_pdf_renderers import original_resume, resume_layout_analyzer
from app.services.resume_pdf_renderers import templates as resume_templates
from app.services.resume_pdf_renderers.router import ORIGINAL, TemplateChoiceRequired, select_renderer
from app.services.resume_pdf_renderers.render_validator import validate_rendering
from app.services.resume_pdf_renderers.template_renderer import preview_png

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
    # Questions the posting or recipient asked ("Why should you choose me?"). Each is answered in the email.
    questions: list[str] = Field(default_factory=list, max_length=4)
    # One sentence in the candidate's own words about why this company (used exactly as written, first in the intro).
    personal_note: str | None = Field(default=None, max_length=240)

    @field_validator("questions")
    @classmethod
    def _clean_questions(cls, value: list[str]) -> list[str]:
        cleaned = [" ".join(q.split()) for q in value if q and q.strip()]
        for q in cleaned:
            if not 3 <= len(q) <= 200:
                raise ValueError("each question must be 3 to 200 characters")
        return cleaned

    @field_validator("personal_note")
    @classmethod
    def _clean_note(cls, value: str | None) -> str | None:
        value = " ".join((value or "").split())
        return value or None


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
        extras = {k: v for k, v in (("questions", req.questions), ("personal_note", req.personal_note)) if v}
        draft = agent.run(profile, company, recipient, tailored=tailored, **extras)
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

def _resume_attachment(email: Email, profile_row: dict[str, Any], template: str | None):
    """The tailored resume PDF that goes with this email, as ``(attachments, None)``; ``(None, None)``
    when the email is not tied to a tailored resume; ``(None, response)`` when the user must choose a
    template first (the email stays pending, nothing is sent)."""
    if not email.resume_draft_id:
        return None, None
    record = _get_resume_draft(email.resume_draft_id)
    if record.profile_id != email.profile_id:
        raise HTTPException(409, "This email's tailored resume belongs to a different profile")
    try:
        pdf_bytes, _kind, _status, _report = _render_approved_resume(record, template)
    except _ChooseTemplate as exc:
        return None, _choose_template(exc.reason, **exc.extra)
    return [(_attachment_name(profile_row), pdf_bytes, "application/pdf")], None


def _attachment_name(profile_row: dict[str, Any]) -> str:
    """e.g. "Vidip_Ghosh_Resume.pdf": letters, digits and underscores only."""
    plain = unicodedata.normalize("NFKD", str(profile_row.get("name") or "")).encode("ascii", "ignore").decode()
    stem = re.sub(r"[^A-Za-z0-9]+", "_", plain).strip("_")
    return f"{stem or 'Resume'}_Resume.pdf" if stem else "Resume.pdf"


@app.post("/emails/{email_id}/approve", response_model=Email)
def approve_and_send_email(email_id: str, template: str | None = None):
    """Approve draft and deliver via SMTP.

    An email written from an approved tailored resume goes out with that resume attached as a PDF.
    ``template`` picks its design (see GET /resume-templates, or ``original``); when it is needed and
    missing the answer is 409 with the gallery and the email stays pending review.
    """
    email_row = _get_entity(db.EMAILS, email_id)
    email = Email(**email_row)

    if email.status != EmailStatus.PENDING_REVIEW:
        raise HTTPException(409, f"Only emails pending review can be approved (current: {email.status.value})")

    profile_row = _get_entity(db.PROFILES, email.profile_id)
    recipient_row = _get_entity(db.RECIPIENTS, email.recipient_id)

    # Built before anything is sent or recorded: no resume, no email.
    attachments, needs_choice = _resume_attachment(email, profile_row, template)
    if needs_choice is not None:
        return needs_choice

    try:
        email_service.send_email(
            to_email=recipient_row["email"],
            subject=email.subject or "Outreach",
            body=email.body or "",
            from_email=profile_row.get("email"),
            **({"attachments": attachments} if attachments else {}),
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

@app.patch("/emails/{email_id}", response_model=Email)
def edit_email(email_id: str, data: EmailUpdate):
    """Change the subject and/or body of an email that is still pending review."""
    changes = data.model_dump(exclude_unset=True)
    if not changes or all(v is None for v in changes.values()):
        raise HTTPException(422, "Send at least one of: subject, body")
    email = Email(**_get_entity(db.EMAILS, email_id))
    if email.status != EmailStatus.PENDING_REVIEW:
        raise HTTPException(409, f"Only emails pending review can be edited (current: {email.status.value})")
    updated = db.update(db.EMAILS, email_id, {k: v for k, v in changes.items() if v is not None})
    return Email(**updated)


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

@app.post("/profiles/{profile_id}/recommend-roles")
def recommend_profile_roles(profile_id: str):
    """Which known roles fit this profile best, and what each one is missing."""
    profile = db.fetch(db.PROFILES, profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return {"profile_id": profile_id, "recommendations": recommend_roles(profile)}


class ResumeTailorRequest(BaseModel):
    job_description: str

def _save_resume_draft(profile_id: str, job_description: str, draft: dict) -> str | None:
    """Persist a generated resume draft for user review."""
    try:
        row = _draft_db(
            db.insert,
            db.RESUME_DRAFTS,
            {
                "profile_id":profile_id,
                "job_description":job_description,
                "draft":draft,
                "status":ResumeDraftStatus.PENDING_REVIEW
            }
        )
        if not row or not row.get("id"):
            raise HTTPException(
                status_code=500,
                detail="Resume draft was generated but could not be saved."
            )
        return row["id"]
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to save resume draft: %s", exc)

        raise HTTPException(
            status_code=500,
            detail="Failed to persist resume draft."
        ) from exc


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

class ExperienceEntry(BaseModel):
    company: str
    job_title: str
    duration: Optional[str] = None
    location: Optional[str] = None
    bullets: list[str] = Field(default_factory=list)

class ProjectEntry(BaseModel):
    name: str
    technologies: list[str] = Field(default_factory=list)
    description: Optional[str] = None
    bullets: list[str] = Field(default_factory=list)

class ResumeDraftEdit(BaseModel):
    """The fields a user may change while reviewing a draft. A field that is not sent is left alone;
    a field sent as null or as blank text is cleared."""

    headline: str | None = Field(default=None, max_length=120)
    summary: str | None = Field(default=None, max_length=600)
    skills: Optional[list[str]] = None
    experience: Optional[list[ExperienceEntry]] = None
    projects: Optional[list[ProjectEntry]] = None
    achievements: Optional[list[str]] = None
    education: Optional[list[dict]] = None


@app.patch("/resume-drafts/{draft_id}", response_model=ResumeDraftRecord)
def edit_resume_draft(draft_id: str, data: ResumeDraftEdit):
    """Edit the headline and/or summary of a draft that is still pending review."""
    changes = data.model_dump(exclude_unset=True,mode="json")
    if not changes:
        raise HTTPException(422, "Send at least one of: headline, summary, skills, experience, projects, achievements, education")

    record = _get_resume_draft(draft_id)
    if record.status != ResumeDraftStatus.PENDING_REVIEW:
        raise HTTPException(409, f"Only drafts pending review can be edited (current: {record.status.value})")

    new_draft = dict(record.draft)
    scalar_fields = {"headline","summary"}
    list_fields = {"skills","achievements","education"}
    structured_fields = {"experience","projects"}

    for field,value in changes.items():
        if field in scalar_fields:
            if value is None:
                new_draft[field] = None
            else:
                new_draft[field] = value.strip() or None
        elif field in list_fields:
            if value is None:
                new_draft[field] = []
            else:
                new_draft[field] = value
        elif field in structured_fields:
            if value is None:
                new_draft[field] = []
            else:
                new_draft[field] = value
    updated = _draft_db(db.update, db.RESUME_DRAFTS, draft_id, {"draft": new_draft})
    return ResumeDraftRecord(**updated)

def _load_original(profile_id: str) -> bytes | None:
    """The uploaded resume PDF stored for this profile, or None."""
    try:
        return original_resume.load(profile_id)
    except ValueError:
        return None


def _choose_template(reason: str, **extra: Any) -> JSONResponse:
    """409 asking the user to pick a template; carries the gallery so the UI can show it."""
    return JSONResponse(
        status_code=409,
        content={"detail": reason, "action": "choose_template",
                 "templates": resume_templates.list_templates(), **extra},
    )


class _ChooseTemplate(Exception):
    """The user has to pick a template before a PDF can be made."""

    def __init__(self, reason: str, **extra: Any):
        super().__init__(reason)
        self.reason, self.extra = reason, extra


def _render_approved_resume(record: ResumeDraftRecord, template: str | None):
    """The PDF of an approved tailored resume: ``(pdf_bytes, renderer_name, validation_status, report)``.

    Shared by the download endpoint and by emails, so what is previewed is exactly what is sent. Raises
    ``_ChooseTemplate`` when the user must pick a design, and HTTPException for real errors. A PDF that
    failed its checks is never returned.
    """
    if record.status != ResumeDraftStatus.APPROVED:
        raise HTTPException(409, "Approve the resume draft before generating its PDF")

    profile = _get_entity(db.PROFILES, record.profile_id)
    draft_data = record.draft
    if hasattr(draft_data, "model_dump"):
        draft_data = draft_data.model_dump()
    if not isinstance(draft_data, dict):
        raise HTTPException(422, "Resume draft must be a dictionary")

    original = _load_original(record.profile_id)
    try:
        renderer, kind = select_renderer(template, original)
    except TemplateChoiceRequired as exc:
        raise _ChooseTemplate(exc.reason)
    except ValueError as exc:                                   # unknown template id
        raise HTTPException(422, str(exc))

    logger.info("Generating PDF for draft %s with %s", record.id, kind)
    try:
        if kind == ORIGINAL:
            pdf_bytes, report = renderer.render_with_report(draft_data)
            verdict = validate_rendering(original, pdf_bytes, draft_data)
            if not verdict["passed"]:
                failed = [{"name": c["name"], "detail": c["detail"]} for c in verdict["checks"] if c["status"] == "fail"]
                logger.warning("Rebuilt original design failed validation for draft %s: %s", record.id, failed)
                raise _ChooseTemplate("Your resume's own design could not be reproduced reliably for this draft.",
                                      failed_checks=failed)
            status = "warn" if verdict["summary"]["warn"] else "pass"
        else:
            if original:      # the profile may lack details (e.g. phone) that the uploaded resume has
                profile = original_resume.fill_contacts(profile, original_resume.load_assessment(record.profile_id))
            pdf_bytes, report = renderer.render_with_report(profile, draft_data)
            status = "pass"
    except _ChooseTemplate:
        raise
    except ValueError as exc:
        if kind == ORIGINAL:                                     # design not supported: let the user choose
            raise _ChooseTemplate(str(exc))
        logger.warning("PDF generation failed: %s", exc)
        raise HTTPException(422, detail=str(exc))
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected PDF generation error")
        raise HTTPException(500, detail="Failed to generate resume PDF")
    return pdf_bytes, kind, status, report


def _pdf_response(pdf_bytes: bytes, filename: str, kind: str, status: str, report: dict) -> Response:
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'inline; filename="{filename}"',
            "X-Resume-Renderer": kind,
            "X-Resume-Validation": status,
            "X-Resume-Pages": str(report.get("pages", "")),
        },
    )


@app.get("/resume-drafts/{draft_id}/pdf")
def download_tailored_resume(draft_id: str, template: str | None = None):
    """The approved tailored resume as a PDF.

    ``template`` is a gallery id (see GET /resume-templates) or ``original`` for the user's own uploaded
    design. Omitted, the uploaded design is used when there is one. When it is missing or cannot be
    reproduced reliably the answer is 409 with the gallery, so the user picks a template; a PDF that
    failed its checks is never returned.
    """
    record = _get_resume_draft(draft_id)
    try:
        pdf_bytes, kind, status, report = _render_approved_resume(record, template)
    except _ChooseTemplate as exc:
        return _choose_template(exc.reason, **exc.extra)
    return _pdf_response(pdf_bytes, f"tailored-resume-{draft_id}.pdf", kind, status, report)


@app.get("/emails/{email_id}/attachment")
def preview_email_attachment(email_id: str, template: str | None = None):
    """The PDF that approving this email would attach (same template choice as POST /emails/{id}/approve),
    so the user can look at it before sending. 404 when the email carries no tailored resume."""
    email = Email(**_get_entity(db.EMAILS, email_id))
    if not email.resume_draft_id:
        raise HTTPException(404, "This email is not tied to a tailored resume, so it has no attachment")
    record = _get_resume_draft(email.resume_draft_id)
    if record.profile_id != email.profile_id:
        raise HTTPException(409, "This email's tailored resume belongs to a different profile")
    profile_row = _get_entity(db.PROFILES, email.profile_id)
    try:
        pdf_bytes, kind, status, report = _render_approved_resume(record, template)
    except _ChooseTemplate as exc:
        return _choose_template(exc.reason, **exc.extra)
    return _pdf_response(pdf_bytes, _attachment_name(profile_row), kind, status, report)


# --- Resume template gallery -------------------------------------------------------------------

@app.get("/resume-templates")
def list_resume_templates(profile_id: str | None = None):
    """The ATS-safe templates a user can pick from. With ``profile_id`` it also says whether that
    user's own uploaded design can be used."""
    items = [dict(t, preview_url=f"/resume-templates/{t['id']}/preview") for t in resume_templates.list_templates()]
    own = None
    if profile_id:
        assessment = None
        try:
            assessment = original_resume.load_assessment(profile_id)
        except ValueError:
            pass
        own = {"id": ORIGINAL, "name": "Your uploaded design",
               "available": bool(assessment) and _load_original(profile_id) is not None,
               "preserve_supported": bool((assessment or {}).get("preserve_supported")),
               "reason": (assessment or {}).get("reason") or (None if assessment else "No resume PDF uploaded yet"),
               "ats": (assessment or {}).get("ats")}
    return {"templates": items, "original": own}


@app.get("/resume-templates/{template_id}/preview")
def resume_template_preview(template_id: str):
    """PNG preview of a template filled with fictional sample content."""
    try:
        return Response(content=preview_png(template_id), media_type="image/png")
    except ValueError as exc:
        raise HTTPException(404, str(exc))


@app.post("/profiles/{profile_id}/original-resume")
def upload_original_resume(profile_id: str, file: UploadFile = File(...)):
    """Keep the user's original resume PDF so tailored resumes can reuse its design. The response says
    whether that design can be reproduced and lists any ATS risks found in it."""
    _get_entity(db.PROFILES, profile_id)
    data = _validate_and_read_pdf(file)
    try:
        assessment = original_resume.assess(data)
        original_resume.save(profile_id, data)
        original_resume.save_assessment(profile_id, assessment)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    return {"stored": True, **assessment}


@app.get("/profiles/{profile_id}/original-resume")
def get_original_resume_status(profile_id: str):
    _get_entity(db.PROFILES, profile_id)
    assessment = original_resume.load_assessment(profile_id)
    if assessment is None or _load_original(profile_id) is None:
        raise HTTPException(404, "No original resume PDF is stored for this profile")
    return {"stored": True, **assessment}
@app.post("/profiles/resume/layout")
def analyze_resume_layout(file: UploadFile = File(...)):
    data = _validate_and_read_pdf(file)
    try:
        return resume_layout_analyzer.analyze_resume_layout(data)
    except ValueError as exc:
        raise HTTPException(422,detail=str(exc))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
