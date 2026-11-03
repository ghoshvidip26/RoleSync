import json
import logging
import os
import re
from typing import Annotated

from langchain_ollama import ChatOllama
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
tailor_agent = ChatOllama(
    model=CHAT_MODEL,           # OLLAMA_CHAT_MODEL switches it, e.g. to a stronger model
    temperature=0.3,
    repeat_penalty=1.15,
    num_ctx=4096,
    num_predict=300,            # a short answer is all that is needed; this stops runaway loops early
    client_kwargs={"timeout": 60},
)


class _Prose(BaseModel):
    """Both fields are required, so an empty `{}` is not a valid answer."""

    model_config = ConfigDict(extra="forbid")

    headline: Annotated[str, Field(min_length=3, max_length=120)]
    summary: Annotated[str, Field(min_length=30, max_length=350)]


PROSE_PROMPT = """
You write the headline and summary of a resume, tailored to a job. Use ONLY the facts you are given.

- Write as the resume's owner: no "I", no "my", no "the candidate".
- The target job's title is NOT the candidate's title. Never use it as the candidate's own title.
- headline: the candidate's most recent role (as given in the facts), then " | ", then two or three
  of the listed skills. At most 12 words.
- summary: two sentences, at most 60 words, built from the facts.
- Do not mention any skill, tool, employer, number or achievement that is not in the facts.
- Do not copy a fact word for word; write it naturally.
Return a JSON object with exactly the keys "headline" and "summary".
"""

_FIRST_PERSON = re.compile(r"\b(?:I|my|me|myself)\b")


def _verify_prose(prose: _Prose, facts: dict) -> None:
    """Accept the model's text only if everything in it is supported by the resume. Raises ValueError."""
    known, jd_terms = facts["known"], facts["jd_terms"]
    candidate_text = facts["candidate_text"]
    for field, text in (("headline", prose.headline.strip()), ("summary", prose.summary.strip())):
        words = text.split()
        if _JUNK.search(text) or _THIRD_PERSON.search(text) or _FIRST_PERSON.search(text):
            raise ValueError(f"{field} has stray quotes/JSON, or is not written as the resume's owner")
        if not text[0].isalpha() or not any(c.isalpha() for c in text):
            raise ValueError(f"{field} is not meaningful text")
        lowered = [w.lower().strip(".,;:()|") for w in words]
        trigrams = [tuple(lowered[i : i + 3]) for i in range(len(lowered) - 2)]
        if len(set(lowered)) / len(words) < 0.6 or len(trigrams) != len(set(trigrams)):
            raise ValueError(f"{field} is repetitive")
        unsupported = []
        for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9.+#_-]*", text):
            stem = _tokens(word)[0]
            if stem not in known and (stem in jd_terms or _TECHNICAL.search(word.strip("."))):
                unsupported.append(word)
        unsupported += [n for n in re.findall(r"\d+(?:[.,]\d+)?", text) if n not in set(re.findall(r"\d+(?:[.,]\d+)?", candidate_text))]
        if unsupported:
            raise ValueError(f"{field} mentions things not in the resume: {', '.join(dict.fromkeys(unsupported[:5]))}")

    headline_words, summary_words = prose.headline.split(), prose.summary.split()
    if not 2 <= len(headline_words) <= 15:
        raise ValueError("headline is not a short title line")
    if not 8 <= len(summary_words) <= 70 or not prose.summary.strip().endswith((".", "!")):
        raise ValueError("summary is not 2 short sentences")
    if "|" in prose.summary:
        raise ValueError("summary looks like a pasted resume header")

    squash = lambda text: re.sub(r"\W+", "", text.lower())
    candidate_sq = squash(candidate_text)

    # Never claim the job's own title: words such as "backend" or "engineer" can appear elsewhere in
    # the resume, so check the whole phrase.
    role = squash(facts["role"])
    if role and any(role in squash(text) and role not in candidate_sq for text in (prose.headline, prose.summary)):
        raise ValueError(f"claims the job's title ({facts['role']}), which the candidate has not held")

    # The part of the headline before "|" or "at" must be a real job title (dropping "Junior" is fine,
    # adding words is not), or just a list of the candidate's own matching skills.
    lead = re.split(r"\||\bat\b", prose.headline, maxsplit=1)[0].strip(" ,")
    titles = [squash(t) for t in facts["titles"]]
    skill_items = [squash(i) for i in re.split(r"[,;]", lead) if i.strip()]
    allowed_skills = {squash(sk) for sk in facts["skills"]}
    if titles and not (
        any(squash(lead) in t for t in titles) or (skill_items and all(i in allowed_skills for i in skill_items))
    ):
        raise ValueError(f"headline title '{lead}' is not one of the candidate's real job titles")

    # Pasting resume sentences one after another is not a rewrite.
    sentences = [x for x in re.split(r"(?<=[.!?])\s+", prose.summary.strip()) if len(squash(x)) >= 12]
    if sentences and all(squash(x) in candidate_sq for x in sentences):
        raise ValueError("summary is copied from the resume instead of written for this job")
    mentioned = set(_tokens(prose.headline + " " + prose.summary)) & facts["anchor_terms"]
    if len(mentioned) < 2:
        raise ValueError("text does not mention the candidate's real role, employer or matching skills")


def _llm_prose(job_description: str, facts: dict) -> _Prose | None:
    """One attempt to have the model write the headline and summary from verified facts. Returns None
    (and logs why) if the model is unavailable or its text is not fully supported by the resume."""
    if os.getenv("TAILOR_USE_LLM", "1") == "0":
        return None  # model step switched off: keep the rule-based text
    if not facts["anchor_terms"]:
        return None  # nothing verified to write from
    lines = [f"The job being applied for (NOT the candidate's title): {facts['role']}"]
    if facts["title"]:
        lines.append(f"Most recent role: {facts['title']}" + (f" at {facts['company']}" if facts["company"] else ""))
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
        logger.warning("Model-written headline/summary not used (rule-based text kept): %s", str(exc).splitlines()[0][:200])
        return None


def _build_draft(profile: dict,job_description: str,evidence: str)-> tuple[TailoredResumeDraft,dict]:
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

    # Headline uses the candidate's real most recent title.
    title, company = next(
        ((title, company) for title, company in roles if title),
        (None, None),
    )

    headline_skills = matched_skills[:3]

    headline = " | ".join(
        part for part in (
            title,
            ", ".join(headline_skills),
        )
        if part
    ) or None

    # Build a conservative summary from actual profile facts.
    summary_parts = []

    if title and company:
        summary_parts.append(f"{title} at {company}")

    if matched_skills:
        summary_parts.append(
            "Skilled in " + ", ".join(matched_skills[:5])
        )

    summary = (
        ". ".join(summary_parts).rstrip(".") + "."
        if summary_parts
        else _shorten(_as_text(profile.get("summary")), 350) or None
    )

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
        "role": re.split(r"[.\n]", job_description.strip(), maxsplit=1)[0][:80],
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
    )

    prose = _llm_prose(job_description, facts)

    if prose:
        draft.headline = prose.headline.strip()
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
