
import json
import logging
import re
from typing import Annotated, Callable, TypeVar

from langchain_ollama import ChatOllama
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.agents.review import _JUNK, _THIRD_PERSON, _terms, _tokens  # shared text helpers
from app.services.outreach_agent import get_retriever

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 2

ShortText = Annotated[str, Field(max_length=500)]
TextList = Annotated[list[ShortText], Field(max_length=20)]


class HeadlineDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    headline: Annotated[str | None, Field(max_length=120)]


class SummaryDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: Annotated[str | None, Field(max_length=350)]


class SkillsDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    skills_to_highlight: Annotated[list[Annotated[str, Field(max_length=100)]], Field(max_length=10)]


class ExperienceDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    experience_edits: Annotated[list[Annotated[str, Field(max_length=250)]], Field(max_length=5)]


class ProjectsDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    projects_to_highlight: Annotated[list[Annotated[str, Field(max_length=180)]], Field(max_length=5)]


class RecommendationsDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    recommendations: Annotated[list[Annotated[str, Field(max_length=200)]], Field(max_length=5)]
    claims_requiring_confirmation: Annotated[list[Annotated[str, Field(max_length=200)]], Field(max_length=5)]


class _RemainingDraft(SkillsDraft, ExperienceDraft, ProjectsDraft, RecommendationsDraft):
    """The list sections, still generated in one call. The next steps split these up too."""


class ResumeTailorError(Exception):
    """Raised when the resume tailoring model fails or returns invalid output."""


class TailoredResumeDraft(BaseModel):
    """Proposed resume edits; these do not overwrite the original profile."""

    model_config = ConfigDict(extra="forbid")

    headline: str | None
    summary: str | None
    skills_to_highlight: TextList
    experience_edits: TextList
    projects_to_highlight: TextList
    recommendations: TextList
    claims_requiring_confirmation: TextList


tailor_agent = ChatOllama(
    model="qwen:latest",
    temperature=0.3,
    repeat_penalty=1.05,
    num_ctx=8192,
)


TAILOR_PROMPT = """
You are a careful resume tailoring assistant.

Use the candidate profile and retrieved resume evidence as the only sources
of truth. Tailor the proposed resume content to the supplied job description.

Rules:
- Never invent skills, technologies, employers, job titles, dates, metrics,
  achievements, responsibilities, or project details.
- Never turn a skill mentioned only in the job description into a candidate skill.
- Preserve the meaning of the candidate's actual work.
- Improve clarity, relevance, and wording without exaggerating accomplishments.
- Highlight only existing skills and projects relevant to the role.
- Keep experience edits grounded in the supplied candidate evidence.
- If a useful claim cannot be verified, put it in claims_requiring_confirmation
  instead of presenting it as a fact.
- Do not claim to generate a formatted PDF or modify the original resume.
- Return only JSON matching the requested schema.
"""


def _as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(str(item) for item in value if item)
    return str(value)

class OutputQualityError(Exception):
    """Raised when generated resume content is malformed or repetitive."""


def _check_repetition(field: str, words: list[str]) -> None:
    """Reject suspiciously repetitive output, such as "and and and" or one phrase repeated."""
    if len(words) < 8:
        return
    and_ratio = sum(word.lower() == "and" for word in words) / len(words)
    lowered = [word.lower().strip(".,;:()") for word in words]
    unique_ratio = len(set(lowered)) / len(words)
    trigrams = [tuple(lowered[i : i + 3]) for i in range(len(lowered) - 2)]
    duplicate_trigrams = len(trigrams) - len(set(trigrams))
    if and_ratio > 0.25 or unique_ratio < 0.5 or duplicate_trigrams > 1:  # a phrase repeated
        raise OutputQualityError(f"{field} contains excessive repetition.")


def _check_text(field: str, value: str | None) -> None:
    """Quality rules for one text field. None or empty text is allowed: it means "nothing to suggest"."""
    if value is None or not value.strip():
        return

    cleaned = value.strip()
    words = cleaned.split()

    # Reject strings that contain no meaningful alphanumeric content.
    if not any(char.isalnum() for char in cleaned):
        raise OutputQualityError(f"{field} contains no meaningful text.")

    _check_repetition(field, words)

    # A headline should not be a lone symbol, runaway token, or fragment.
    if field == "headline":
        if any(len(w) > 30 for w in words):
            raise OutputQualityError("Headline contains malformed tokens.")
        alpha_words = [w for w in words if any(char.isalpha() for char in w)]
        if len(words) < 2 or len(alpha_words) < 2:
            raise OutputQualityError("Headline is incomplete.")
        if len(words) > 15:
            raise OutputQualityError("Headline is too long; it should be a short title line.")


def _check_items(field: str, values: list[str]) -> None:
    """Quality rules for the entries of one list field."""
    for value in values:
        cleaned = value.strip()

        if not cleaned or not any(char.isalnum() for char in cleaned):
            raise OutputQualityError(
                f"{field} contains an empty or meaningless entry."
            )

        _check_repetition(field, cleaned.split())


def _validate_draft_quality(draft: TailoredResumeDraft) -> None:
    """Reject obviously corrupted or meaningless generated text."""

    _check_text("headline", draft.headline)
    _check_text("summary", draft.summary)

    for field, values in {
        "skills_to_highlight": draft.skills_to_highlight,
        "experience_edits": draft.experience_edits,
        "projects_to_highlight": draft.projects_to_highlight,
        "recommendations": draft.recommendations,
        "claims_requiring_confirmation": draft.claims_requiring_confirmation,
    }.items():
        _check_items(field, values)


SectionT = TypeVar("SectionT", bound=BaseModel)


def _validate_section_quality(section_output: BaseModel) -> None:
    """The same quality rules, applied to whichever fields one section's schema has."""
    for field, value in section_output.model_dump().items():
        if isinstance(value, list):
            _check_items(field, value)
        else:
            _check_text(field, value)


def _describe_error(exc: Exception) -> str:
    """A short, useful reason: for schema errors the field and the rule, not just the model name."""
    if isinstance(exc, ValidationError):
        return "; ".join(
            f"{'.'.join(str(part) for part in err['loc']) or 'output'}: {err['msg']}"
            for err in exc.errors()[:3]
        )
    if isinstance(exc, json.JSONDecodeError):
        return f"not valid JSON ({exc.msg})"
    return str(exc).splitlines()[0]


def _generate_section(
    section: str,
    schema: type[SectionT],
    messages: list[tuple[str, str]],
    validate: Callable[[SectionT], None] | None = None,
) -> SectionT:
    """Generate ONE section of the draft with its own small, bounded schema.

    The model call, JSON parsing, schema validation, quality checks and the optional `validate`
    callback (which raises OutputQualityError) all happen here. If any of them fails, only this
    section is retried, with the reason fed back to the model. After MAX_ATTEMPTS a
    ResumeTailorError names the section that could not be generated.
    """
    last_error = "unknown error"

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            content = tailor_agent.invoke(
                messages,
                format=schema.model_json_schema(),
            ).content
        except Exception as exc:  # Ollama down, model missing, generation aborted, ...
            last_error = f"model call failed: {exc}"
            logger.warning("Tailoring %s, attempt %d: %s", section, attempt, last_error)
            continue

        try:
            result = schema.model_validate(json.loads(content))
            _validate_section_quality(result)
            if validate is not None:
                validate(result)
            return result
        except (json.JSONDecodeError, ValidationError, OutputQualityError) as exc:
            last_error = f"invalid output: {_describe_error(exc)}"
            logger.warning("Tailoring %s, attempt %d: %s", section, attempt, last_error)
            messages = messages + [
                ("assistant", content),
                (
                    "human",
                    f"That {section} was rejected ({last_error}). Return only the requested "
                    "JSON, with short, natural, meaningful text taken from the candidate's "
                    "real resume. Use null or an empty list if nothing applies.",
                ),
            ]

    raise ResumeTailorError(
        f"Could not generate the {section} after {MAX_ATTEMPTS} attempts: {last_error}"
    )


HEADLINE_PROMPT = """
Write a resume headline for this candidate, to send for the job below: a short professional
title line of 3 to 10 words, for example "Backend Engineer | Python, Kafka, PostgreSQL".

Use only job titles, skills and technologies that appear in the candidate's profile or
evidence. Do not use the job description's title unless the candidate has really held that
role. Do not write sentences, conversational explanations, or quotes. If you cannot do this truthfully, return null.
"""

SUMMARY_PROMPT = """
Write a resume summary of 2-3 sentences for this candidate, to send for the job below.

Write it as the resume's owner (no "I", no "the candidate"). Use only facts from the
candidate's profile and evidence: their real roles, technologies and results, with numbers
exactly as written there. Do not copy existing summary paragraphs verbatim, and do not use
vertical bar "|" characters. Do not add any skill, tool, employer, title, number or achievement
that the candidate's resume does not contain, even if the job description asks for it. If
you cannot do this truthfully, return null.
"""

_TECHNICAL = re.compile(r"[A-Z].*[A-Z]|\d|[+#]")   # CamelCase, ALLCAPS, digits, C++ / C#
_NUMBERS = re.compile(r"\d+(?:[.,]\d+)?")


def _check_grounded(field: str, text: str | None, candidate_text: str, job_description: str) -> None:
    """Reject text that states something the candidate's own resume does not support: skills or
    wording that only the job description has, technical-looking terms or numbers the resume lacks,
    stray JSON debris, or text written about "the candidate"."""
    if not text:
        return
    if _JUNK.search(text) or _THIRD_PERSON.search(text):
        raise OutputQualityError(
            f"{field} contains stray quote/JSON characters or is written about 'the candidate'."
        )

    if field == "summary":
        if "|" in text:
            raise OutputQualityError("summary looks like a pasted resume header, not a summary.")
        if len(text.split()) >= 8 and re.sub(r"\W+", "", text.lower()) in re.sub(r"\W+", "", candidate_text.lower()):
            raise OutputQualityError("summary is copied from the resume instead of written for this job.")

    known = set(_tokens(candidate_text))
    jd_terms = set(_terms(job_description))
    unsupported = []
    for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9.+#_-]*", text):
        stem = _tokens(word)[0]
        if stem not in known and (stem in jd_terms or _TECHNICAL.search(word.strip("."))):
            unsupported.append(word)
    candidate_numbers = set(_NUMBERS.findall(candidate_text))
    unsupported += [n for n in _NUMBERS.findall(text) if n not in candidate_numbers]
    if unsupported:
        raise OutputQualityError(
            f"{field} mentions things not in the candidate's resume: {', '.join(dict.fromkeys(unsupported[:5]))}."
        )


def tailor_resume(
    profile_id: str,
    profile: dict,
    job_description: str,
) -> dict:
    """Generate a reviewable, evidence-grounded resume tailoring draft."""

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
            f"No resume index found for profile {profile_id}. Re-upload the resume to rebuild it."
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
    candidate_text = f"{profile_json}\n{evidence}"      # everything the draft is allowed to draw on

    def messages_for(prompt: str) -> list[tuple[str, str]]:
        return [
            ("system", prompt),
            (
                "human",
                f"JOB DESCRIPTION:\n{job_description[:2000]}\n\n"
                f"CANDIDATE PROFILE:\n{profile_json[:3500]}\n\n"
                f"RETRIEVED RESUME EVIDENCE:\n{evidence[:2000]}",
            ),
        ]

    # Each section is generated and checked on its own; a failure retries only that section.
    headline = _generate_section(
        "headline", HeadlineDraft, messages_for(HEADLINE_PROMPT),
        lambda out: _check_grounded("headline", out.headline, candidate_text, job_description),
    )
    summary = _generate_section(
        "summary", SummaryDraft, messages_for(SUMMARY_PROMPT),
        lambda out: _check_grounded("summary", out.summary, candidate_text, job_description),
    )
    remaining = _generate_section("list sections", _RemainingDraft, messages_for(TAILOR_PROMPT))

    draft = TailoredResumeDraft(
        headline=headline.headline,
        summary=summary.summary,
        **remaining.model_dump(),
    )
    try:
        _validate_draft_quality(draft)
    except OutputQualityError as exc:
        raise ResumeTailorError(f"The generated draft failed validation: {exc}") from exc

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
