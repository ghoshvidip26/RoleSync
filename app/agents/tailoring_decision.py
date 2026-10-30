from collections import Counter
import re

from app.agents.resume_tailor import _ROLE_STEMS  # job-title nouns such as "engineer" are not skills
from app.agents.review import _terms
from app.services.outreach_agent import experience_metadata, parse_entry, project_metadata

MAX_SAME_OPENER = 2     # the same opening word may start at most this many bullets in a resume
MIN_LEAD_KEYWORDS = 2   # the latest role should share at least this many keywords with the job


def _entries(profile: dict, field: str) -> list[str]:
    value = profile.get(field) or []
    if isinstance(value, str):
        value = [value]
    return [v for v in value if isinstance(v, str) and v.strip()]


def _bullets(profile: dict) -> list[str]:
    """Every bullet of every role and project, with wrapped lines already rejoined."""
    out: list[str] = []
    for field in ("experience", "projects"):
        for raw in _entries(profile, field):
            out += [line[2:].strip() for line in parse_entry(raw).lines if line.startswith("• ")]
    return out


def detect_repeated_action_verbs(bullets: list[str]) -> dict[str, int]:
    """Opening words that start more than MAX_SAME_OPENER bullets, e.g. {"built": 4}."""
    counts: Counter = Counter()
    for bullet in bullets:
        first = re.match(r"[^\w]*([A-Za-z]+)", bullet)   # skips a leading "•" but not a number such as "50%"
        if first:
            counts[first.group(1).lower()] += 1
    return {word: n for word, n in counts.items() if n > MAX_SAME_OPENER}


def find_buried_work(profile: dict, job_description: str) -> str | None:
    """A reason if the work that matches the job is not in the latest role, else None."""
    experience = _entries(profile, "experience")
    if not experience:
        return None  # nothing is "latest", so nothing can be buried behind it

    keywords = set(_terms(job_description)) - _ROLE_STEMS
    if len(keywords) < MIN_LEAD_KEYWORDS:
        return None

    def hits(raw: str) -> int:
        return len(keywords & set(_terms(raw.replace("/n", "\n"))))

    lead_hits = hits(experience[0])
    if lead_hits >= MIN_LEAD_KEYWORDS:
        return None

    best_hits, best_label = 0, ""
    for raw in experience[1:]:
        meta = experience_metadata(parse_entry(raw))
        label = f"{meta['job_title']} at {meta['company']}" if "company" in meta else "an earlier role"
        if hits(raw) > best_hits:
            best_hits, best_label = hits(raw), label
    for raw in _entries(profile, "projects"):
        name = project_metadata(parse_entry(raw)).get("project_name")
        if hits(raw) > best_hits:
            best_hits, best_label = hits(raw), f"your project {name}" if name else "one of your projects"

    if best_hits < MIN_LEAD_KEYWORDS:
        return None  # the resume has no matching work at all; tailoring cannot fix that
    return (
        f"Your latest role matches {lead_hits} of this job's keywords, but {best_label} matches {best_hits}. "
        "Lead with that work."
    )


def needs_tailoring(profile: dict, job_description: str) -> dict:
    """Whether tailoring would help: repeated opening words, or relevant work buried behind the latest role."""
    reasons: list[str] = []

    for word, count in detect_repeated_action_verbs(_bullets(profile)).items():
        reasons.append(f"'{word}' starts {count} bullets; use each opening word at most {MAX_SAME_OPENER} times.")

    buried = find_buried_work(profile, job_description)
    if buried:
        reasons.append(buried)

    return {"needed": bool(reasons), "reasons": reasons}
