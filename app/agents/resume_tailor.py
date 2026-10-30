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


class TailoredResumeDraft(BaseModel):
    """Proposed resume edits; these do not overwrite the original profile."""

    headline: str | None = None
    summary: str | None = None
    skills_to_highlight: list[str] = Field(default_factory=list)
    experience_edits: list[str] = Field(default_factory=list)
    projects_to_highlight: list[str] = Field(default_factory=list)
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


def _build_draft(profile: dict, job_description: str, evidence: str) -> tuple[TailoredResumeDraft, dict]:
    """Build the draft from the candidate's own data. Every item is a real skill, project or bullet
    from the resume that overlaps the job description, so nothing can be invented; anything the job
    asks for that the resume does not show is listed as a claim to confirm, never as a strength."""
    jd_stems = _terms(job_description)
    jd_set = set(jd_stems) - _ROLE_STEMS
    jd_words: dict[str, str] = {}
    for word in _WORD.findall(job_description):
        jd_words.setdefault(_norm(word), word.rstrip("."))
    show = lambda stems: ", ".join(jd_words.get(s, s) for s in stems)

    known = set(_tokens(f"{json.dumps(profile, default=str)}\n{evidence}"))

    # Skills the job asks for that the candidate really lists (every term of the skill must be in the job).
    listed: list[str] = []
    for field in ("skills", "programming_languages", "frameworks", "databases_tools"):
        for text in _as_list(profile.get(field)):
            listed += [item.strip() for item in re.split(r"[,;|]", text.split(":")[-1]) if item.strip()]
    skills = []
    for item in _dedupe(listed):
        terms = _terms(item)
        if terms and set(terms) <= jd_set:
            skills.append((min(jd_stems.index(t) for t in terms), item))
    matched_skills = [item for _, item in sorted(skills)][:MAX_ITEMS]

    # Projects, ranked by overlap with the job; only named projects (a name is never made up).
    projects = []
    for raw in _as_list(profile.get("projects")):
        entry = parse_entry(raw)
        meta = project_metadata(entry)
        overlap = [t for t in _terms(raw.replace("/n", "\n")) if t in jd_set]
        if overlap and "project_name" in meta:
            bonus = 0.5 if meta["project_name"].lower() in evidence.lower() else 0
            projects.append((len(overlap) + bonus, meta["project_name"], meta.get("technologies"), overlap))
    projects.sort(key=lambda p: -p[0])
    project_names = [name for _, name, _, _ in projects]
    projects_out = [
        f"{name} ({tech}): matches the job on {show(overlap)}" if tech else f"{name}: matches the job on {show(overlap)}"
        for _, name, tech, overlap in projects[:5]
    ]

    # Experience: for each role, the real bullet that best matches the job, quoted as written.
    roles, edits = [], []
    for raw in _as_list(profile.get("experience")):
        entry = parse_entry(raw)
        meta = experience_metadata(entry)
        label = f"{meta['job_title']} at {meta['company']}" if "company" in meta else _shorten(entry.header, 60)
        roles.append((meta.get("job_title"), meta.get("company")))
        best = max(
            ((len([t for t in _terms(line) if t in jd_set]), line) for line in entry.lines),
            default=(0, ""),
            key=lambda pair: pair[0],
        )
        if best[0]:
            bullet = best[1].lstrip("• ").strip()
            matched = show([t for t in _terms(bullet) if t in jd_set])
            edits.append((best[0], bullet, f"{label}: lead with “{_shorten(bullet, 180)}” (matches {matched})"))
    edits.sort(key=lambda e: -e[0])
    edits_out = [text for _, _, text in edits[:5]]

    # Headline and summary: real title, real employer, real skills and a real result.
    title, company = next(((t, c) for t, c in roles if t), (None, None))
    top = matched_skills[:3]
    headline = " | ".join(part for part in (title, ", ".join(top)) if part) or None
    sentences = []
    if title and company:
        lead = f"{title} at {company}"
        sentences.append(lead + (f" with experience in {', '.join(matched_skills[:4])}." if matched_skills else "."))
    elif _as_text(profile.get("summary")).strip():
        sentences.append(_shorten(_as_text(profile["summary"]), 200).rstrip(".") + ".")
    if edits:
        sentences.append(_shorten(edits[0][1], 180).rstrip(".") + ".")
    summary = _shorten(" ".join(sentences), 350) if sentences else None

    # What the job asks for but the resume does not show.
    claims = [
        f"{word}: the job asks for it but your resume doesn't show it. Add it only if you have really used it."
        for word in _absent_technical_terms(job_description, known)[:6]
    ]

    recommendations = []
    if matched_skills:
        recommendations.append(f"List {', '.join(matched_skills[:5])} first in your skills section; the job asks for them.")
    if project_names:
        recommendations.append(f"Put {project_names[0]} at the top of your projects; it is your closest match to this job.")
    if edits_out:
        recommendations.append("Open each role with the matching bullet listed in experience_edits and keep its numbers exactly as written.")
    if claims:
        recommendations.append("Do not add the skills in claims_requiring_confirmation unless you have actually used them.")
    if not (matched_skills or projects_out or edits_out):
        recommendations.append(
            "Your resume shows little overlap with this job description. Check whether the role fits, or "
            "add real experience you have that it asks for."
        )

    draft = TailoredResumeDraft(
        headline=_shorten(headline, 120) if headline else None,
        summary=summary,
        skills_to_highlight=matched_skills,
        experience_edits=edits_out,
        projects_to_highlight=projects_out,
        recommendations=recommendations[:MAX_ITEMS],
        claims_requiring_confirmation=claims,
    )
    facts = {  # the only material the model is given, and what its text is checked against
        "title": title,
        "company": company,
        "skills": matched_skills[:5],
        "bullets": [_shorten(e[1], 180) for e in edits[:2]],
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

    draft, facts = _build_draft(profile, job_description, evidence)
    prose = _llm_prose(job_description, facts)
    if prose:
        draft.headline, draft.summary = prose.headline.strip(), prose.summary.strip()

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
