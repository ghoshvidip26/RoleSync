import json
import logging

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, ValidationError

from app.services.outreach_agent import get_retriever

logger = logging.getLogger(__name__)


class ResumeTailorError(Exception):
    """Raised when resume evidence retrieval or draft generation fails."""


class TailoredResumeDraft(BaseModel):
    """Proposed resume edits; these do not overwrite the original profile."""

    headline: str | None = None
    summary: str | None = None
    skills_to_highlight: list[str] = Field(default_factory=list)
    experience_edits: list[str] = Field(default_factory=list)
    projects_to_highlight: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    claims_requiring_confirmation: list[str] = Field(default_factory=list)


tailor_agent = ChatOllama(
    model="qwen:latest",
    temperature=0.2,
    repeat_penalty=1.1,
    num_ctx=8192,
)


TAILOR_PROMPT = """
You are a resume tailoring assistant.

Use the candidate profile and retrieved resume evidence as the sources of truth.
Tailor the resume to the supplied job description.

Rules:
- Keep all claims truthful and grounded in the supplied evidence.
- Do not invent skills, technologies, employers, dates, metrics, or achievements.
- Fill every field with useful content whenever the evidence supports it.
- Do not return empty lists if relevant evidence is available.
- Write a professional headline using the candidate's existing role and relevant skills.
- Write a concise summary of 2-3 sentences based on the candidate's actual work.
- Select relevant existing skills, experience, and projects for the target role.
- Make recommendations specific and actionable.
- Use claims_requiring_confirmation only for claims that genuinely need verification.
- Avoid repetition, placeholder text, and meaningless outputs such as ")".
- Return only valid JSON matching the requested schema.
"""


def _as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(str(item) for item in value if item)
    return str(value)


def tailor_resume(
    profile_id: str,
    profile: dict,
    job_description: str,
) -> dict:
    """Generate one reviewable, evidence-grounded resume tailoring draft."""

    if not job_description.strip():
        raise ValueError("Job description must not be empty.")

    if not any(
        _as_text(profile.get(field)).strip()
        for field in ("summary", "skills", "experience", "projects")
    ):
        raise ValueError("The profile does not contain enough resume information.")

    retriever = get_retriever(profile_id)
    if retriever is None:
        raise ResumeTailorError(
            f"No resume index found for profile {profile_id}. "
            "Re-upload the resume to rebuild it."
        )

    try:
        documents = retriever.invoke(job_description)
    except Exception as exc:
        logger.exception("Failed to retrieve resume evidence for %s", profile_id)
        raise ResumeTailorError(
            "Could not retrieve resume evidence. Check that the profile has been indexed."
        ) from exc

    evidence = "\n\n".join(
        doc.page_content for doc in documents if doc.page_content
    )

    candidate_profile = {
        field: profile.get(field)
        for field in (
            "summary",
            "skills",
            "programming_languages",
            "frameworks",
            "databases_tools",
            "experience",
            "projects",
            "achievements",
            "education",
            "target_roles",
        )
        if profile.get(field) is not None
    }

    profile_json = json.dumps(candidate_profile, ensure_ascii=False, default=str)
    messages = [
        ("system", TAILOR_PROMPT),
        (
            "human",
            f"JOB DESCRIPTION:\n{job_description[:3000]}\n\n"
            f"CANDIDATE PROFILE:\n{profile_json[:5000]}\n\n"
            f"RETRIEVED RESUME EVIDENCE:\n{evidence[:4000]}",
        ),
    ]

    try:
        response = tailor_agent.invoke(
            messages,
            format=TailoredResumeDraft.model_json_schema(),
        )
        raw_output = response.content
        if not isinstance(raw_output, str):
            raw_output = json.dumps(raw_output, ensure_ascii=False)
        draft_data = json.loads(raw_output)
        draft = TailoredResumeDraft.model_validate(draft_data)
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning("Resume tailoring returned invalid JSON or fields: %s", exc)
        raise ResumeTailorError(
            "The model returned an invalid resume draft. Please try again."
        ) from exc
    except Exception as exc:
        logger.exception("Resume tailoring model call failed for %s", profile_id)
        raise ResumeTailorError(
            "Could not generate the resume draft. Check that Ollama is running "
            "and the configured model is available."
        ) from exc

    # Light cleanup only; no per-section quality checks or retry loops.
    for field in (
        "skills_to_highlight",
        "experience_edits",
        "projects_to_highlight",
        "recommendations",
        "claims_requiring_confirmation",
    ):
        cleaned = []
        for item in getattr(draft, field):
            item = item.strip()
            if item and item.lower() not in {x.lower() for x in cleaned}:
                cleaned.append(item)
        setattr(draft, field, cleaned[:10])

    if draft.headline is not None:
        draft.headline = draft.headline.strip() or None
    if draft.summary is not None:
        draft.summary = draft.summary.strip() or None

    return {
        "profile_id": profile_id,
        "status": "pending_review",
        "draft": draft.model_dump(),
        "evidence_used": [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in documents
        ],
    }
