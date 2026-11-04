"""Resume positioning: the headline, the professional summary and the skills to highlight.

Everything here is built from the candidate's own resume data and the target job, by rules. Nothing is
invented: each sentence of a summary is the resume's own wording, a headline names only a role the resume
evidences, and skills come from the candidate's profile.

Each of the three fields has several variants, so "regenerate" gives a different, equally grounded suggestion.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from app.agents.review import _tokens
from app.services.outreach_agent import _clause_cut

VARIANTS = 3
HEADLINE_MAX = 120
SUMMARY_MAX = 350            # what is generated; the editor allows up to 600

_ROLE_NOUN = r"(?:engineer|developer|programmer|architect|analyst|scientist|consultant|designer|sdet|tester)s?"
_ROLE_NOUNS = {"engineer", "developer", "programmer", "architect", "analyst", "scientist", "consultant", "designer", "sdet", "tester"}
# Seniority and level words are never claimed on the candidate's behalf; the resume's own titles carry that.
_SENIORITY = {"junior", "jr", "senior", "sr", "staff", "principal", "lead", "associate", "entry", "mid", "intern", "trainee", "graduate",
              "level", "entrylevel", "midlevel", "i", "ii", "iii", "iv", "v"}
_MANAGEMENT = {"manager", "director", "head", "vp", "chief", "president", "owner", "cto", "ceo"}
_STOP = {"the", "a", "an", "and", "of", "for", "to", "in", "at", "with", "on"}
# Words a posting puts in front of a title ("an experienced Python developer") that are not part of it.
_FLUFF = {"experienced", "talented", "skilled", "passionate", "motivated", "seasoned", "strong", "proven", "dedicated", "creative", "driven",
          "great", "amazing", "ambitious", "excellent", "outstanding", "hands-on", "handson", "enthusiastic", "capable", "smart", "new", "our", "your"}
_WORD = re.compile(r"[A-Za-z0-9+#.]+")


def _norm_word(word: str) -> str:
    return re.sub(r"[^a-z0-9+#]", "", word.lower())


# --------------------------------------------------------------------------------------------------------------
# The job's role

def _clean_title(text: str) -> str | None:
    """"Senior React Native Developer (Remote) - Bangalore" -> "Senior React Native Developer". None unless it
    ends in a role noun, so a sentence or a team name is never taken for a title."""
    text = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", text)
    text = re.split(r"\s[-–—|@]\s|,\s|\sat\s|\sin\s|\sto\s|\swho\s|\swith\s|\syou\b", text, maxsplit=1)[0]
    words = [w for w in _WORD.findall(text)]
    while words and _norm_word(words[0]) in _FLUFF:
        words.pop(0)
    while words and _norm_word(words[-1]) in {"i", "ii", "iii", "iv", "v", "1", "2", "3"}:
        words.pop()
    if not 1 <= len(words) <= 7 or _norm_word(words[-1]).rstrip("s") not in _ROLE_NOUNS:
        return None
    return " ".join(words)


def target_role(job_description: str, job_title: str | None = None) -> str | None:
    """The role the job is for: the title the user gave, else one found in the text, else None."""
    if job_title and job_title.strip():
        return _clean_title(job_title.strip())
    jd = job_description or ""
    patterns = (
        rf"\b(?:as|join us as|joining us as|become)\s+an?\s+((?:[A-Za-z0-9+#.&/-]+\s+){{0,5}}{_ROLE_NOUN})\b",
        rf"\b(?:hiring|looking for|seeking|recruiting)\s+(?:an?\s+)?((?:[A-Za-z0-9+#.&/-]+\s+){{0,5}}{_ROLE_NOUN})\b",
        r"(?im)^\s*(?:job title|role|position|title)\s*[:\-–]\s*([^\n]{3,80})$",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, jd, flags=re.IGNORECASE):
            title = _clean_title(match.group(1))
            if title:
                return title
    for line in jd.splitlines()[:6]:                       # a posting often starts with its title
        title = _clean_title(line.strip()) if 3 <= len(line.strip()) <= 70 else None
        if title:
            return title
    return None


def supported_role(role: str | None, candidate_text: str) -> str | None:
    """`role` without seniority words, if the resume evidences it; else None. A role is evidenced when every word
    that distinguishes it ("React Native" in "React Native Developer") appears in the candidate's own profile.
    A management title, or a role with no distinguishing word at all, gives None."""
    if not role:
        return None
    words = _WORD.findall(role)
    if any(_norm_word(w) in _MANAGEMENT for w in words):
        return None
    kept = [w for w in words if _norm_word(w) not in _SENIORITY]
    distinct = [w for w in kept if _norm_word(w).rstrip("s") not in _ROLE_NOUNS and _norm_word(w) not in _STOP]
    if not kept or not distinct:
        return None
    known = set(_tokens(candidate_text))
    if any(_tokens(w)[0] not in known for w in distinct if _tokens(w)):
        return None
    return " ".join(kept)


# --------------------------------------------------------------------------------------------------------------
# Headline

def _years_phrase(summary: str | None) -> str | None:
    match = re.search(r"\b(\d+\+?)\s*(years?)\s+of\s+(?:professional\s+)?experience\b", summary or "", flags=re.IGNORECASE)
    return f"{match.group(1)} {match.group(2).lower()} of experience" if match else None


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}" if items else ""


def build_headline(role: str | None, title: str | None, skills: list[str], years: str | None, variant: int) -> str | None:
    """A short, role-specific line. With a role the resume evidences it is "React Native Developer | JavaScript,
    Firebase"; without one it uses the candidate's real title. Skills that the role already names are not repeated."""
    lead = role or title
    if not lead:
        return ", ".join(skills[:3]) or None            # no real title and no evidenced role: just the matching skills
    covered = set(_tokens(lead))
    extra = [s for s in skills if not set(_tokens(s)) <= covered][:3] or []
    options: list[str] = []
    if extra:
        options.append(f"{lead} | {', '.join(extra)}")
        options.append(f"{lead} with {_join(extra[:2])}" if len(extra) >= 2 else f"{lead} with {extra[0]}")
        options.append(f"{_join(extra)} | {lead}")
    if years:
        options.append(f"{lead} | {years}" + (f", {extra[0]}" if extra else ""))
    if not options:
        options.append(lead)
    text = options[variant % len(options)]
    if len(text) > HEADLINE_MAX:
        text = _clause_cut(text, HEADLINE_MAX)
    return text


# --------------------------------------------------------------------------------------------------------------
# Summary

def _sentence(text: str) -> str:
    text = " ".join((text or "").split()).rstrip(" .;:,")
    return text[0].upper() + text[1:] + "." if text else ""


def _first_sentence(text: str | None, limit: int = 230) -> str:
    text = " ".join((text or "").split())
    first = re.split(r"(?<=[.!?])\s+", text)[0] if text else ""
    return first if first and len(first) <= limit and not re.match(r"(?i)^(?:i|i'm|i’m|we|my)\b", first) else ""


def _fit(sentences: list[str], limit: int = SUMMARY_MAX) -> str:
    """Whole sentences joined while they fit; the first is cut at a clause if even it is too long."""
    out: list[str] = []
    for s in sentences:
        if not s:
            continue
        if len(" ".join([*out, s])) <= limit:
            out.append(s)
        elif not out:
            out.append(_sentence(_clause_cut(s, limit - 1)))
    return " ".join(out)


def build_summary(*, profile_summary: str | None, title: str | None, company: str | None, skills: list[str],
                  accomplishments: list[str], project_lines: list[str], variant: int) -> str | None:
    """A professional summary made only of the resume's own statements: who the candidate is, one or two proofs
    that matter for the job, and the job-relevant skills not already mentioned. No sentence repeats another."""
    own = _first_sentence(profile_summary)
    role_line = _sentence(f"{title} at {company}" if title and company else (title or ""))
    intro = _sentence(own) if own else role_line
    proofs = [_sentence(a) for a in accomplishments if a]
    projects = [_sentence(p) for p in project_lines if p]

    def new_skills(text: str) -> str:
        said = set(_tokens(text))
        fresh = [s for s in skills if not set(_tokens(s)) <= said][:4]
        if not fresh:
            return ""
        return _sentence(f"{'Also hands-on with' if said & set(_tokens(' '.join(skills))) else 'Hands-on with'} {_join(fresh)}")

    nth = lambda items, i: items[i] if len(items) > i else ""
    SKILLS = None                                                # marks where the skills sentence goes
    plans: list[list[str | None]] = [
        [intro, nth(proofs, 0), SKILLS],                                         # who + the best proof + skills
        [intro, nth(projects, 0) or nth(proofs, 1), SKILLS],                     # who + a project + skills
        [role_line or intro, nth(proofs, 0), nth(proofs, 1)],                    # current role + two proofs
    ]
    chosen = plans[variant % len(plans)]
    parts: list[str] = []
    for item in chosen:
        if item is None:                                       # the skills sentence, built from what is already said
            item = new_skills(" ".join(parts))
        if item and not any(set(_tokens(item)) <= set(_tokens(p)) or set(_tokens(p)) <= set(_tokens(item)) for p in parts):
            parts.append(item)
    if not parts:
        parts = [intro or role_line]
    text = _fit(parts)
    return text or None


# --------------------------------------------------------------------------------------------------------------
# Putting it together

def accomplishment_lines(experience_items: list[Any], relevance: Callable[[str], int], limit: int = 2) -> list[str]:
    """The best real bullets: relevant to the job first, then ones with numbers, one per role, cut at a clause."""
    pool: list[tuple[tuple[int, int, int], str]] = []
    for index, item in enumerate(experience_items):
        bullets = [b.strip() for b in getattr(item, "bullets", []) or [] if b and b.strip()]
        if bullets:
            best = max(bullets, key=lambda b: (relevance(b), bool(re.search(r"\d", b))))
            pool.append(((relevance(best), int(bool(re.search(r"\d", best))), -index), best))
    pool.sort(key=lambda p: p[0], reverse=True)
    return [_clause_cut(text, 150) for _, text in pool[:limit]]


def project_lines(project_items: list[Any], relevance: Callable[[str], int], limit: int = 1) -> list[str]:
    """"<the project's best bullet> (<name>)" for the most relevant projects."""
    lines = []
    for item in project_items[:limit]:
        bullets = [b.strip() for b in getattr(item, "bullets", []) or [] if b and b.strip()]
        name = (getattr(item, "name", "") or "").strip()
        if bullets and name:
            best = max(bullets, key=lambda b: (relevance(b), bool(re.search(r"\d", b))))
            lines.append(f"{_clause_cut(best, 130)} ({name})")
    return lines
