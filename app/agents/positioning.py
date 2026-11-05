"""Resume positioning: the headline, the professional summary and the skills to highlight.

Everything here is built from the candidate's own resume data and the target job, by rules. Nothing is
invented: each sentence of a summary is the resume's own wording, a headline names only a role the resume
evidences, and skills come from the candidate's profile.

Each of the three fields has several variants, so "regenerate" gives a different, equally grounded suggestion.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from app.agents.review import _WORD as _REVIEW_WORD, _norm as _review_norm, _terms, _tokens
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
# Skill order: what the job wants most comes first

_OPTIONAL_HEAD = re.compile(r"nice[\s-]*to[\s-]*haves?|good[\s-]*to[\s-]*haves?|preferred|bonus|desirable|optional|extra points", re.IGNORECASE)
_PLUS = re.compile(r"\b(?:is|are|would be|will be)\s+(?:a\s+)?(?:big\s+)?(?:plus|bonus|advantage)\b|\ba (?:plus|bonus)\b", re.IGNORECASE)
_REQUIRED_HEAD = re.compile(
    r"^\W*(?:requirements?|required|must[\s-]*haves?|qualifications?|what you(?:'|’)ll need|what we(?:'|’)re looking for|responsibilit\w+|"
    r"what you(?:'|’)ll do|key skills|skills)\b", re.IGNORECASE)
_STRONG = re.compile(r"\b(?:required|must|essential|proficien\w+|strong|expert\w*|solid|hands-on)\b", re.IGNORECASE)


def _optional_ranges(text: str) -> list[tuple[int, int]]:
    """Character ranges of the job description that are "nice to have": a section under such a heading, the rest of a line after
    "Nice to have:", and the sentence before "... is a plus"."""
    ranges: list[tuple[int, int]] = []
    pos, in_optional = 0, False
    for line in text.splitlines(keepends=True):
        start, end = pos, pos + len(line)
        pos = end
        stripped = line.strip()
        heading = 0 < len(stripped) <= 45
        marker = _OPTIONAL_HEAD.search(line)
        if heading and not marker and _REQUIRED_HEAD.match(stripped):
            in_optional = False
        if marker and heading:
            remainder = line[marker.end():].strip(" :-–\n\r\t")
            if remainder:
                ranges.append((start + marker.end(), end))        # "Nice to have: Kubernetes, Docker." (the list is on the heading line)
            else:
                in_optional = True                                # a heading such as "Nice to have" opens an optional section
            continue
        if marker:
            ranges.append((start + marker.start(), end))          # "... Docker. Nice to have: Kubernetes."
        elif in_optional:
            ranges.append((start, end))
        for plus in _PLUS.finditer(line):
            before = max(line.rfind(".", 0, plus.start()), line.rfind("!", 0, plus.start()), line.rfind("?", 0, plus.start()))
            ranges.append((start + before + 1, start + plus.end()))
    return ranges


def rank_skills(skills: list[str], job_description: str, job_title: str | None = None) -> list[str]:
    """`skills` ordered by how much the job wants each: a skill in the title or repeated in the requirements comes first, one
    asked for as "required/must/strong" next, and one that is only "nice to have" or "a plus" after them. Ties keep the order
    the job lists them in. Skills the job never mentions keep their order at the end. Nothing is added or removed."""
    jd = job_description or ""
    words = [(m.start(), _review_norm(m.group(0))) for m in _REVIEW_WORD.finditer(jd)]
    optional = _optional_ranges(jd)
    is_optional = lambda offset: any(a <= offset < b for a, b in optional)
    title_terms = set(_tokens(job_title or ""))

    def score(skill: str) -> tuple[float, int]:
        terms = _terms(skill)
        if not terms:
            return (0.0, 10**9)
        hits = [off for i, (off, t) in enumerate(words)
                if t == terms[0] and set(terms) <= {x for _, x in words[i : i + len(terms) + 2]}]
        if not hits:
            return (0.0, 10**9)
        total = sum(0.5 if is_optional(off) else 2.0 for off in hits[:4])
        for off in hits[:4]:
            ends = [i for i in (jd.find(".", off), jd.find("\n", off)) if i >= 0]
            sentence = jd[max(jd.rfind(".", 0, off), jd.rfind("\n", 0, off)) + 1 : min(ends) if ends else len(jd)]
            if not is_optional(off) and _STRONG.search(sentence):
                total += 1.0
                break
        if set(terms) <= title_terms:
            total += 4.0
        return (total, hits[0])

    scored = [(skill, score(skill)) for skill in skills]
    return [skill for skill, _ in sorted(scored, key=lambda item: (-item[1][0], item[1][1]))]


# --------------------------------------------------------------------------------------------------------------
# The job's focus: backend, frontend, full-stack, mobile or AI. A summary is written for it, so the same resume reads
# differently for a backend job than for a frontend one. It is used only when the resume itself evidences that focus.

_FOCUS_WORDS = {
    "backend": "backend back-end server-side server api apis rest restful graphql grpc microservices microservice node.js express django "
               "flask fastapi spring laravel rails postgresql postgres mysql mongodb redis kafka rabbitmq sql database databases orm "
               "authentication caching queue queues",
    "frontend": "frontend front-end ui ux react next.js vue angular svelte html css tailwind redux responsive accessibility components "
                "browser webpack vite figma dom",
    "mobile": "mobile android ios flutter kotlin swift expo native",
    "ai": "ai ml llm llms rag langchain embeddings nlp pytorch tensorflow generative faiss vector",
}
FOCUS_TERMS = {focus: set(_tokens(words)) for focus, words in _FOCUS_WORDS.items()}
FOCUS_NOUN = {"backend": "Backend", "frontend": "Frontend", "fullstack": "Full-stack", "mobile": "Mobile"}   # no noun for AI: skills carry it
_FULLSTACK_RE = re.compile(r"full[\s-]?stack", re.IGNORECASE)


def _focus_counts(text: str) -> dict[str, int]:
    tokens = _tokens(text)
    return {focus: sum(1 for t in tokens if t in terms) for focus, terms in FOCUS_TERMS.items()}


def job_focus(job_description: str, job_title: str | None = None) -> str | None:
    """What kind of job this is, by what it asks for: "backend", "frontend", "fullstack", "mobile", "ai" or None when it is
    not clear. The title counts several times over; a job that asks for both sides about equally is full-stack."""
    jd, title = job_description or "", job_title or ""
    counts = _focus_counts(jd)
    for focus, extra in _focus_counts(title).items():
        counts[focus] += 4 * extra
    if _FULLSTACK_RE.search(title) or len(_FULLSTACK_RE.findall(jd)) >= 2 or _FULLSTACK_RE.search(jd[:300]):
        return "fullstack"
    back, front, mobile, ai = counts["backend"], counts["frontend"], counts["mobile"], counts["ai"]
    if mobile >= 3 and mobile >= front:
        return "mobile"
    names_both = {"backend", "frontend"} <= set(_tokens(jd))              # the posting itself names both sides
    if back >= 3 and front >= 3 and (names_both or min(back, front) >= 0.6 * max(back, front)):
        return "fullstack"
    top = max(("backend", "frontend", "ai"), key=lambda f: counts[f])
    return top if counts[top] >= 3 else None


def candidate_focus_hits(text: str) -> dict[str, set[str]]:
    """Which focus terms the candidate's own resume text contains, per focus."""
    tokens = set(_tokens(text))
    return {focus: tokens & terms for focus, terms in FOCUS_TERMS.items()}


def supported_focus(focus: str | None, candidate_text: str) -> str | None:
    """`focus` if the resume evidences it (at least two of its terms; both sides for full-stack), else None: a backend
    summary is never written for someone whose resume shows no backend work."""
    if not focus:
        return None
    hits = candidate_focus_hits(candidate_text)
    needs = ("backend", "frontend") if focus == "fullstack" else (focus,)
    return focus if all(len(hits[f]) >= 2 for f in needs) else None


def _families(focus: str) -> tuple[str, ...]:
    return ("backend", "frontend") if focus == "fullstack" else (focus,)


def focus_skills(focus: str | None, skills: list[str], matched: list[str], limit: int = 5) -> dict[str, list[str]]:
    """The candidate's OWN skills that belong to the focus, the job's own words first: {"backend": [...], "frontend": [...]}
    (one key, or two for full-stack). Empty lists are left in so the caller knows nothing matched."""
    ordered = list(dict.fromkeys([*matched, *skills]))
    out: dict[str, list[str]] = {}
    for family in _families(focus or ""):
        out[family] = [s for s in ordered if set(_tokens(s)) & FOCUS_TERMS[family]][:limit]
    return out


def _focus_score(text: str, relevance: Callable[[str], int], families: tuple[str, ...]) -> tuple[int, int, int]:
    tokens = set(_tokens(text))
    hits = sum(len(tokens & FOCUS_TERMS[f]) for f in families)
    return (hits * 3 + relevance(text), hits, int(bool(re.search(r"\d", text))))


def focused_accomplishments(experience_items: list[Any], relevance: Callable[[str], int], focus: str, limit: int = 2) -> list[str]:
    """The real bullets that show the job's focus, best first: for full-stack one for each side. One bullet per role."""
    bullets = [(index, b.strip()) for index, item in enumerate(experience_items)
               for b in (getattr(item, "bullets", []) or []) if b and b.strip()]
    picked: list[tuple[int, str]] = []

    def take(families: tuple[str, ...]) -> None:
        pool = [(index, b) for index, b in bullets if index not in {i for i, _ in picked} and b not in {x for _, x in picked}]
        if pool:
            best = max(pool, key=lambda ib: _focus_score(ib[1], relevance, families))
            if _focus_score(best[1], relevance, families)[0] > 0:
                picked.append(best)

    if focus == "fullstack":
        take(("backend",))
        take(("frontend",))
    while len(picked) < limit:
        before = len(picked)
        take(_families(focus))
        if len(picked) == before:
            break
    return [_clause_cut(text, 150) for _, text in picked[:limit]]


def focused_projects(project_items: list[Any], relevance: Callable[[str], int], focus: str, limit: int = 1) -> list[str]:
    """"<the project's best bullet> (<name>)" for the project that shows the focus best."""
    scored = []
    for item in project_items:
        bullets = [b.strip() for b in getattr(item, "bullets", []) or [] if b and b.strip()]
        name = (getattr(item, "name", "") or "").strip()
        if bullets and name:
            text = " ".join([name, *getattr(item, "technologies", []), *bullets])
            best = max(bullets, key=lambda b: _focus_score(b, relevance, _families(focus)))
            scored.append((_focus_score(text, relevance, _families(focus)), f"{_clause_cut(best, 130)} ({name})"))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [line for _, line in scored[:limit]]


def check_focus(text: str, focus: str) -> None:
    """Raise ValueError unless `text` is about the job's focus: it names something from each side the focus needs, and for a
    backend-only or frontend-only job it does not lean on the other side."""
    counts = _focus_counts(text)
    for family in _families(focus):
        if counts[family] == 0:
            raise ValueError(f"the summary does not mention anything {family}-specific, which this job is about")
    if focus in ("backend", "frontend"):
        other = "frontend" if focus == "backend" else "backend"
        if counts[other] >= counts[focus]:
            raise ValueError(f"the summary leans on {other} work, but this job is {focus}")


# --------------------------------------------------------------------------------------------------------------
# Projects: the ones that show what the job asks for

# For choosing projects only (the summary's focus detection is unchanged): AI projects also show up as agents, retrieval, chat...
_PROJECT_AI = set(_tokens("agent agents agentic multi-agent orchestration retrieval semantic chatbot openai gemini claude embedding transformer "
                          "fine-tuning inference"))
PROJECT_TERMS = {**FOCUS_TERMS, "ai": FOCUS_TERMS["ai"] | _PROJECT_AI}
# Which kinds of work a job of each focus wants to see in a project. A full-stack job also shows the candidate's AI work.
PROJECT_FAMILIES = {"backend": ("backend",), "frontend": ("frontend",), "fullstack": ("backend", "frontend", "ai"),
                    "mobile": ("mobile", "frontend"), "ai": ("ai",)}
PROJECT_MIN_SCORE = 1.0


def project_score(project: Any, ranked_skills: list[str], focus: str | None) -> float:
    """How well one project shows what the job wants: each of the job's skills that the project names counts (the ones the job
    wants most count more), and so does each kind of work the job's focus asks for. 0 means it shows none of it."""
    text = " ".join([getattr(project, "name", "") or "", *(getattr(project, "technologies", []) or []), *(getattr(project, "bullets", []) or [])])
    tokens = set(_tokens(text)) | set(_tokens(re.sub(r"[-/]", " ", text)))        # "5-agent" is also "5" and "agent"
    n = max(len(ranked_skills), 1)
    score = sum(1 + (n - i) / n for i, skill in enumerate(ranked_skills) if _terms(skill) and set(_terms(skill)) <= tokens)
    terms: set[str] = set()
    for family in PROJECT_FAMILIES.get(focus or "", ()):
        terms |= PROJECT_TERMS[family]
    return score + 0.5 * len(tokens & terms)


def select_projects(items: list[tuple[int, Any]], ranked_skills: list[str], focus: str | None) -> tuple[list[tuple[int, Any]], list[str]]:
    """(the projects to show, best first; the names left out). A project is kept if it shows the job's skills or focus. If none
    does, nothing is left out: a resume is never emptied of projects. Order among equals is the order they came in."""
    scored = [(project_score(project, ranked_skills, focus), index, entry) for index, entry in enumerate(items) for project in [entry[1]]]
    kept = [x for x in scored if x[0] >= PROJECT_MIN_SCORE]
    if not kept:
        return items, []
    kept.sort(key=lambda x: (-x[0], x[1]))
    kept_ids = {id(x[2]) for x in kept}
    return [x[2] for x in kept], [getattr(entry[1], "name", "") for entry in items if id(entry) not in kept_ids]


# A summary states expertise; it does not quote the experience section. A line that opens with an ownership or duty phrase, or names
# an internal product ("the Sales app"), reads as pasted resume text and is left out of it.
_WEAK_OPENER = re.compile(r"^\W*(?:owned|own|responsible|worked|helped|assisted|participated|contributed|involved|tasked|handled|did|made)\b", re.IGNORECASE)


def summary_worthy(line: str, known_terms: set[str], allow_names: bool = False) -> bool:
    """True if `line` can stand in a professional summary: it does not open with a duty phrase ("Owned...", "Worked on...") and,
    unless `allow_names` (project lines carry their project's name on purpose), it names no capitalised product or company
    that is not one of the candidate's own skills (acronyms such as REST or COD are fine)."""
    text = (line or "").strip()
    if not text or _WEAK_OPENER.match(text):
        return False
    if allow_names:
        return True
    for word in _WORD.findall(text)[1:]:
        bare = word.strip(".")
        if bare[:1].isupper() and not re.fullmatch(r"[A-Z]{2,}s?", bare) and _review_norm(bare) not in known_terms:   # REST, APIs: acronyms
            return False
    return True


# --------------------------------------------------------------------------------------------------------------
# Summary

def _sentence(text: str) -> str:
    text = " ".join((text or "").split()).rstrip(" .;:,")
    return text[0].upper() + text[1:] + "." if text else ""


def _first_sentence(text: str | None, limit: int = 230) -> str:
    text = " ".join((text or "").split())
    first = re.split(r"(?<=[.!?])\s+", text)[0] if text else ""
    # a "summary" of a few words ("Mobile developer.") is a title, not a statement: build a real one instead
    # ...and neither is the text the upload step generates when a resume has no summary ("<Title> | <Company>. Skilled in ...")
    generated = "|" in first or re.search(r"\bSkilled in\b|\bBuilt projects including\b", first)
    return first if first and len(first.split()) >= 6 and len(first) <= limit and not generated and not re.match(r"(?i)^(?:i|i'm|i’m|we|my)\b", first) else ""


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


def _identity(role: str | None, title: str | None, skills: list[str], years: str | None,
              focus: str | None = None, groups: dict[str, list[str]] | None = None, variant: int = 0) -> str:
    """Who the candidate is, as a professional summary says it: a level-free role noun, the years the resume itself states,
    and the skills. Never "<job title> at <company>": that line belongs to the experience section.

    The noun is the job's role when the resume evidences it (`role`, e.g. "Backend Engineer"), else the job's focus
    ("Backend developer", "Full-stack developer"), else the bare noun of the candidate's real title ("Developer").
    With a focus the skills are the candidate's own skills of that kind: backend skills for a backend job, and for a
    full-stack job both, each on its own side."""
    suffix = "developer"
    if title:
        last = _norm_word((_WORD.findall(title) or [""])[-1]).rstrip("s")
        suffix = last if last in _ROLE_NOUNS else suffix
    if role:
        suffix = _norm_word(_WORD.findall(role)[-1]).rstrip("s") if _WORD.findall(role) else suffix
    noun = role or (f"{FOCUS_NOUN[focus]} {suffix}" if focus in FOCUS_NOUN else None)
    if not noun and title:
        last = _norm_word((_WORD.findall(title) or [""])[-1]).rstrip("s")
        noun = last.capitalize() if last in _ROLE_NOUNS else None            # the bare noun: no seniority, no company

    named = set(_tokens(noun or ""))                                         # "React Native Developer": React Native is said already
    skills = [s for s in skills if not set(_tokens(s)) <= named]
    groups = {family: [s for s in items if not set(_tokens(s)) <= named] for family, items in (groups or {}).items()}
    if focus == "fullstack" and groups.get("frontend") and groups.get("backend"):
        sides = f"{_join(groups['frontend'][:3])} on the front end and {_join(groups['backend'][:3])} on the back end"
        lead = f"{noun or 'Full-stack developer'}" + (f" with {years}" if years else "")
        return _sentence(f"{lead}, working across the stack with {sides}")
    focused = [s for family in _families(focus or "") for s in groups.get(family, [])]
    top = _join((focused or skills)[:4])
    if variant % 3 == 2 and noun and top:                                 # the third suggestion words the opening differently
        return _sentence(f"{noun} with {years}, focused on {top}" if years else f"{noun} focused on {top}")
    if noun and years and top:
        return _sentence(f"{noun} with {years}, building with {top}")
    if noun and top:
        return _sentence(f"{noun} with hands-on experience in {top}")
    if noun and years:
        return _sentence(f"{noun} with {years}")
    if top:
        return _sentence(f"Hands-on experience in {top}")
    return _sentence(noun or "")


def build_summary(*, profile_summary: str | None, title: str | None, company: str | None, skills: list[str],
                  accomplishments: list[str], project_lines: list[str], variant: int,
                  role: str | None = None, years: str | None = None, focus: str | None = None,
                  focus_groups: dict[str, list[str]] | None = None) -> str | None:
    """A professional summary made only of the resume's own statements, in the usual order: who the candidate is and
    what they work with, then one or two proofs that matter for the job. No sentence repeats another, and the opening
    never states the job title and company (that is what the experience section is for). `company` is accepted for
    older callers and is deliberately unused."""
    own = _first_sentence(profile_summary)
    # A job with a clear focus gets a summary written for it; the resume's generic own summary would read the same for every job.
    intro = _sentence(own) if own and not focus else _identity(role, title, skills, years, focus, focus_groups, variant)
    proofs = [_sentence(a) for a in accomplishments if a]
    projects = [_sentence(p) for p in project_lines if p]

    def new_skills(text: str) -> str:
        said = set(_tokens(text))
        fresh = [s for s in skills if not set(_tokens(s)) <= said][:4]
        if not fresh:
            return ""
        return _sentence(f"{'Also works with' if said & set(_tokens(' '.join(skills))) else 'Works with'} {_join(fresh)}")

    nth = lambda items, i: items[i] if len(items) > i else ""
    SKILLS = None                                                # marks where the skills sentence goes
    plans: list[list[str | None]] = [
        [intro, nth(proofs, 0), SKILLS],                                        # who + the best proof + the rest of the skills
        [intro, nth(projects, 0) or nth(proofs, 1), nth(proofs, 0)],            # who + a project + the best proof
        [intro, nth(proofs, 0), nth(proofs, 1) or nth(projects, 0)],            # who + two proofs
    ]
    chosen = plans[variant % len(plans)]
    parts: list[str] = []
    for item in chosen:
        if item is None:                                       # the skills sentence, built from what is already said
            item = new_skills(" ".join(parts))
        if item and not any(set(_tokens(item)) <= set(_tokens(p)) or set(_tokens(p)) <= set(_tokens(item)) for p in parts):
            parts.append(item)
    if not parts:
        parts = [intro]
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
