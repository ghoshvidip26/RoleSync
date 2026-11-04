import json
import logging
import os
import re
from typing import Annotated

from langchain_ollama import ChatOllama

from app.agents import positioning
from app.services.llm_fallback import FallbackChat
from pydantic import BaseModel, ConfigDict, Field

from app.agents.review import _JUNK, _THIRD_PERSON, _WORD, _norm, _terms, _tokens  # shared text helpers
from app.services.outreach_agent import (
    CHAT_MODEL,
    experience_metadata,
    get_retriever,
    parse_entry,
    project_metadata,
)

logger = logging.getLogger(__name__)

MAX_ITEMS = 10
_TECHNICAL = re.compile(r"[A-Z].*[A-Z]|\d|[+#]")  # CamelCase, ALLCAPS, digits, C++ / C#
# Job-title nouns say what the role is, not what skills it needs ("engineer" would even match "search engine").
_ROLE_STEMS = {
    _norm(w)
    for w in ("engineer", "engineering", "developer", "programmer", "architect", "analyst", "manager", "specialist", "consultant", "intern")
}


class ResumeTailorError(Exception):
    """Raised when resume evidence retrieval or draft generation fails."""

class ExperienceItem(BaseModel):
    company: str
    job_title: str
    duration: str | None=None
    location: str | None=None
    bullets: list[str] = Field(default_factory=list)

class ProjectItem(BaseModel):
    name: str
    technologies: list[str] = Field(default_factory=list)
    description: str | None = None
    bullets: list[str] = Field(default_factory=list)

class TailoredResumeDraft(BaseModel):
    """Complete, editable resume draft. Original profile remains unchanged."""
    headline: str | None = None
    summary: str | None = None
    skills: list[str] = Field(default_factory=list)
    experience: list[ExperienceItem] = Field(default_factory=list)
    projects: list[ProjectItem] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    education: list[dict] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    claims_requiring_confirmation: list[str] = Field(default_factory=list)


def _as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(str(item) for item in value if item)
    if isinstance(value, dict):
        return "\n".join(f"{key}: {_as_text(item)}" for key, item in value.items() if item)
    return str(value)

def _merge_wrapped_items(items: list[str]) -> list[str]:
    """
    Rejoin obvious continuation fragments from PDF-extracted list items.
    Conservative: only merge when the preceding text clearly continues.
    """
    if not items:
        return []

    merged = []

    continuation_endings = re.compile(
        r"\b(?:for|of|with|including|across|through|using|and|or|"
        r"to|in|on|by|from|at|the|a|an|their|its)$",
        re.IGNORECASE,
    )

    for raw in items:
        item = " ".join(str(raw).split()).strip()

        if not item:
            continue

        if not merged:
            merged.append(item)
            continue

        previous = merged[-1]

        # Specific continuation pattern present in your stored profile:
        # "...including 1st" + "Runner-Up at Build on Aptos..."
        first_achievement_continuation = (
            re.search(r"\bincluding\s+1st$", previous, re.IGNORECASE)
            and re.match(r"^Runner[- ]Up\b", item, re.IGNORECASE)
        )

        # General PDF line-wrap patterns:
        # "...checkout for" + "local merchants..."
        obvious_continuation = (
            continuation_endings.search(previous.rstrip(".;:,"))
            or item[:1].islower()
        )

        if first_achievement_continuation or obvious_continuation:
            merged[-1] = previous.rstrip() + " " + item
        else:
            merged.append(item)

    return merged

def _as_list(value) -> list[str]:
    if isinstance(value, str):
        value = [value]
    return [str(v).strip() for v in value if str(v).strip()] if isinstance(value, list) else []


def _shorten(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "…"


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for item in items:
        key = re.sub(r"\W+", "", item.lower())
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _absent_technical_terms(job_description: str, known: set[str]) -> list[str]:
    """Technology-looking words the job description uses that the candidate's resume never mentions."""
    out, sentence_start = [], True
    for m in re.finditer(r"[A-Za-z0-9][A-Za-z0-9.+#_-]*|[.!?\n]", job_description):
        word = m.group(0)
        if word in ".!?\n":
            sentence_start = True
            continue
        first, sentence_start = sentence_start, word.endswith(".")   # a glued period ends the sentence
        word = word.rstrip(".")
        terms = _terms(word)
        if not terms or terms[0] in known:
            continue
        if _TECHNICAL.search(word) or (word[:1].isupper() and not first):
            out.append(word)
    return _dedupe(out)


# The model only rewrites the headline and summary. Everything else in the draft is selected from the
# resume in code, and the model's text is used only if it passes the checks below.
tailor_agent = FallbackChat(ChatOllama(
    model=CHAT_MODEL,           # OLLAMA_CHAT_MODEL switches it, e.g. to a stronger model
    temperature=0.3,
    repeat_penalty=1.15,
    num_ctx=4096,
    num_predict=300,            # a short answer is all that is needed; this stops runaway loops early
    client_kwargs={"timeout": 60},
), label="tailoring model")     # if the call fails, Groq answers (see app/services/llm_fallback.py)


class _Prose(BaseModel):
    """The model writes only the summary; the headline is built by rules (see positioning.py)."""

    model_config = ConfigDict(extra="forbid")

    summary: Annotated[str, Field(min_length=30, max_length=350)]


PROSE_PROMPT = """
You write the professional summary of a resume, tailored to a job. Use ONLY the facts you are given.

- Write as the resume's owner: no "I", no "my", no "the candidate".
- The target job's title is NOT the candidate's title. Never use it as the candidate's own title.
- Two sentences, at most 60 words. Lead with who the candidate is (their real role and experience), then the one
  or two things from the facts that matter most for this job, with the skills that match it.
- Do not mention any skill, tool, employer, number or achievement that is not in the facts.
- Do not copy a fact word for word, and do not repeat the same skill or idea twice.
Return a JSON object with exactly the key "summary".
"""

_FIRST_PERSON = re.compile(r"\b(?:I|my|me|myself)\b")


def _verify_prose(prose: _Prose, facts: dict) -> None:
    """Accept the model's summary only if everything in it is supported by the resume. Raises ValueError."""
    known, jd_terms = facts["known"], facts["jd_terms"]
    candidate_text = facts["candidate_text"]
    text = prose.summary.strip()
    words = text.split()
    if _JUNK.search(text) or _THIRD_PERSON.search(text) or _FIRST_PERSON.search(text):
        raise ValueError("summary has stray quotes/JSON, or is not written as the resume's owner")
    if not text[0].isalpha() or not any(c.isalpha() for c in text):
        raise ValueError("summary is not meaningful text")
    lowered = [w.lower().strip(".,;:()|") for w in words]
    trigrams = [tuple(lowered[i : i + 3]) for i in range(len(lowered) - 2)]
    if len(set(lowered)) / len(words) < 0.6 or len(trigrams) != len(set(trigrams)):
        raise ValueError("summary is repetitive")
    unsupported = []
    for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9.+#_-]*", text):
        stem = _tokens(word)[0]
        if stem not in known and (stem in jd_terms or _TECHNICAL.search(word.strip("."))):
            unsupported.append(word)
    unsupported += [n for n in re.findall(r"\d+(?:[.,]\d+)?", text) if n not in set(re.findall(r"\d+(?:[.,]\d+)?", candidate_text))]
    if unsupported:
        raise ValueError(f"summary mentions things not in the resume: {', '.join(dict.fromkeys(unsupported[:5]))}")

    if not 8 <= len(words) <= 70 or not text.endswith((".", "!")):
        raise ValueError("summary is not 2 short sentences")
    if "|" in text:
        raise ValueError("summary looks like a pasted resume header")

    squash = lambda value: re.sub(r"\W+", "", value.lower())
    candidate_sq = squash(candidate_text)

    # Never claim the job's own title: words such as "backend" or "engineer" can appear elsewhere in
    # the resume, so check the whole phrase.
    role = squash(facts["role"])
    if role and role in squash(text) and role not in candidate_sq:
        raise ValueError(f"claims the job's title ({facts['role']}), which the candidate has not held")

    # Pasting resume sentences one after another is not a rewrite.
    sentences = [x for x in re.split(r"(?<=[.!?])\s+", text) if len(squash(x)) >= 12]
    if sentences and all(squash(x) in candidate_sq for x in sentences):
        raise ValueError("summary is copied from the resume instead of written for this job")


def _llm_prose(job_description: str, facts: dict) -> _Prose | None:
    """One attempt to have the model write the summary from verified facts. Returns None
    (and logs why) if the model is unavailable or its text is not fully supported by the resume."""
    if os.getenv("TAILOR_USE_LLM", "1") == "0":
        return None  # model step switched off: keep the rule-based text
    if not facts["anchor_terms"]:
        return None  # nothing verified to write from
    lines = [f"The job being applied for (NOT the candidate's title): {facts['role']}"]
    if facts["title"]:
        lines.append(f"Most recent role: {facts['title']}" + (f" at {facts['company']}" if facts["company"] else ""))
    if facts.get("profile_summary"):
        lines.append(f"The candidate's own summary: {facts['profile_summary']}")
    if facts["skills"]:
        lines.append(f"Skills you may mention (they match the job): {', '.join(facts['skills'])}")
    if facts["bullets"]:
        lines.append("Real accomplishments you may refer to:\n" + "\n".join(f"- {b}" for b in facts["bullets"]))
    try:
        content = tailor_agent.invoke(
            [("system", PROSE_PROMPT), ("human", "FACTS:\n" + "\n".join(lines))],
            format=_Prose.model_json_schema(),
        ).content
        prose = _Prose.model_validate(json.loads(content))
        _verify_prose(prose, facts)
        return prose
    except Exception as exc:  # model missing/down, aborted, invalid JSON, or text that failed verification
        logger.warning("Model-written summary not used (rule-based text kept): %s", str(exc).splitlines()[0][:200])
        return None


def _build_draft(profile: dict,job_description: str,evidence: str, *, variant: int = 0, target_title: str | None = None)-> tuple[TailoredResumeDraft,dict]:
    jd_terms = set(_terms(job_description)) - _ROLE_STEMS
    jd_words = {}

    for word in _WORD.findall(job_description):
        jd_words.setdefault(_norm(word),word.rstrip("."))
    
    def matchingTerms(text: str) -> list[str]:
        return list(dict.fromkeys(
            term for term in _terms(text) if term in jd_terms
        ))

    def relevance(text: str) -> int:
        return len(matchingTerms(text))
    
    def show(terms: list[str])->str:
        return ", ".join(jd_words.get(term,term) for term in terms)

    raw_skills = []

    for field in (
        "skills",
        "programming_languages",
        "frameworks",
        "databases_tools",
    ):
        for entry in _as_list(profile.get(field)):
            raw_skills.extend(part.strip() for part in re.split(r"[,;|]", entry) if part.strip())
    
    skills = _dedupe(raw_skills)
    # Every distinctive word of the skill must be in the job: "Gemini API" is not a match just because
    # the job says "API", and "Data Structures & Algorithms" is not one because it says "data".
    matched_skills = [
        skill for skill in skills
        if _terms(skill) and set(_terms(skill)) <= jd_terms
    ]

    matched_skills.sort(
        key=lambda skill: min(
            (
                _terms(job_description).index(term)
                for term in _terms(skill)
                if term in _terms(job_description)
            ),
            default=9999,
        )
    )

    experience_items = []
    roles = []

    for raw in _as_list(profile.get("experience")):
        entry = parse_entry(raw)
        meta = experience_metadata(entry)

        company = meta.get("company", "")
        job_title = meta.get("job_title", "")

        bullets = [
            line.lstrip("•-* ").strip()
            for line in entry.lines
            if line.strip()
        ]

        # Rank bullets by relevance, but retain all of them.
        ranked_bullets = sorted(
            enumerate(bullets),
            key=lambda item: relevance(item[1]),
            reverse=True,
        )

        ordered_bullets = [
            bullet for _, bullet in ranked_bullets
        ]

        item = ExperienceItem(
            company=company,
            job_title=job_title,
            duration=meta.get("dates"),
            location=meta.get("location"),
            bullets=ordered_bullets,
        )

        score = relevance(raw)

        experience_items.append((score, item))
        roles.append((job_title, company))

    # Put the most relevant experience first.
    # experience_items.sort(
    #     key=lambda item: item[0],
    #     reverse=True,
    # )

    # Keep every named project and its original content.
    project_items = []

    for raw in _as_list(profile.get("projects")):
        entry = parse_entry(raw)
        meta = project_metadata(entry)

        name = meta.get("project_name")

        # Never invent a project name.
        if not name:
            continue

        technologies = meta.get("technologies") or []

        if isinstance(technologies, str):
            technologies = [
                value.strip()
                for value in re.split(r"[,;|]", technologies)
                if value.strip()
            ]

        bullets = [
            line.lstrip("•-* ").strip()
            for line in entry.lines
            if line.strip()
        ]

        # Keep the original project bullets, prioritizing relevant ones.
        bullets.sort(
            key=relevance,
            reverse=True,
        )

        project = ProjectItem(
            name=name,
            technologies=technologies,
            description=None,
            bullets=bullets,
        )

        project_items.append((relevance(raw), project))

    project_items.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    title, company = next(
        ((title, company) for title, company in roles if title),
        (None, None),
    )

    # Headline and summary (see positioning.py): role-specific, built from the resume's own words. The role named
    # in the headline is the job's, but only when the resume evidences it; otherwise the candidate's real title.
    profile_blob = json.dumps(profile, default=str)
    job_role = positioning.target_role(job_description, target_title)   # (not `job_title`: the loop above uses that name)
    headline_role = positioning.supported_role(job_role, profile_blob)
    headline = positioning.build_headline(
        headline_role, title, matched_skills, positioning._years_phrase(_as_text(profile.get("summary"))), variant)
    summary = positioning.build_summary(
        profile_summary=_as_text(profile.get("summary")), title=title, company=company, skills=matched_skills,
        accomplishments=positioning.accomplishment_lines([item for _, item in experience_items], relevance),
        project_lines=positioning.project_lines([item for _, item in project_items], relevance), variant=variant,
    ) or _shorten(_as_text(profile.get("summary")), 350) or None

    # Missing technical terms remain suggestions, never candidate skills.
    known = set(_tokens(
        f"{json.dumps(profile, default=str)}\n{evidence}"
    ))

    claims = [
        (
            f"{word}: requested by the job description but not found "
            "in the resume. Add only if you have actually used it."
        )
        for word in _absent_technical_terms(job_description, known)[:10]
    ]

    recommendations = []

    if matched_skills:
        recommendations.append(
            "Prioritize these matching skills: "
            + ", ".join(matched_skills[:5])
        )

    if project_items:
        recommendations.append(
            f"Prioritize the project '{project_items[0][1].name}' "
            "because it has the strongest keyword overlap."
        )

    overlap_found = (
        bool(matched_skills)
        or any(score for score, _ in experience_items)
        or any(score for score, _ in project_items)
    )
    if experience_items and overlap_found:
        recommendations.append(
            "Relevant experience bullets have been moved to the top "
            "of each role. All original bullets are retained."
        )
    if not overlap_found:
        recommendations.append(
            "Your resume shows little overlap with this job description. Check whether the role fits, "
            "or add real experience you have that it asks for."
        )

    if claims:
        recommendations.append(
            "Review missing job requirements separately. "
            "Do not add unverified skills or technologies."
        )

    draft = TailoredResumeDraft(
        headline=_shorten(headline, 120) if headline else None,
        summary=summary,
        skills=matched_skills or skills,
        experience=[item for _, item in experience_items],
        projects=[item for _, item in project_items],
        achievements=_merge_wrapped_items(
            _as_list(profile.get("achievements"))
        ),
        education=[{"details": item} for item in _as_list(profile.get("education"))],
        recommendations=recommendations,
        claims_requiring_confirmation=claims,
    )

    # Everything the optional model-written headline/summary is given and is checked against
    # (_llm_prose and _verify_prose read all of these keys).
    facts = {
        "title": title,
        "company": company,
        "skills": matched_skills[:5],
        "bullets": [
            _shorten(item.bullets[0], 180)
            for _, item in experience_items[:2]
            if item.bullets and relevance(item.bullets[0])
        ],
        "titles": [t for t, _ in roles if t],
        "role": job_role or re.split(r"[.\n]", job_description.strip(), maxsplit=1)[0][:80],
        "matched_skills": matched_skills,
        "profile_summary": _shorten(_as_text(profile.get("summary")), 300),
        "known": known,
        "jd_terms": set(_terms(job_description)),
        "candidate_text": f"{json.dumps(profile, default=str)}\n{evidence}".replace("/n", "\n"),
        "anchor_terms": {t for part in (title, company, *matched_skills[:5]) if part for t in _terms(part)},
    }

    return draft, facts


def tailor_resume(
    profile_id: str,
    profile: dict,
    job_description: str,
    job_title: str | None = None,
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

    draft, facts = _build_draft(
        profile,
        job_description,
        evidence,
        target_title=job_title,
    )

    prose = _llm_prose(job_description, facts)

    if prose:
        draft.summary = prose.summary.strip()

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


SUGGESTABLE_FIELDS = ("headline", "summary", "skills")


def suggest_field(profile: dict, job_description: str, field: str, variant: int = 0, job_title: str | None = None):
    """A fresh suggestion for one field of the positioning section: the headline or summary as text, the
    skills to highlight as a list. Built by rules from the profile and the job; nothing is stored here.
    Different `variant`s give different, equally grounded wording."""
    if field not in SUGGESTABLE_FIELDS:
        raise ValueError(f"field must be one of: {', '.join(SUGGESTABLE_FIELDS)}")
    if not (job_description or "").strip():
        raise ValueError("Job description must not be empty.")
    draft, facts = _build_draft(profile, job_description, "", variant=variant, target_title=job_title)
    if field == "skills":
        return list(facts["matched_skills"])[:12]        # only skills the profile already has
    return getattr(draft, field)
