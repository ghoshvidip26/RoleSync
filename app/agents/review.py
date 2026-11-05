import json
import logging
import re
from typing import Annotated, Any, Literal

from langchain_ollama import ChatOllama

from app.services import llm_fallback
from app.services.llm_fallback import FallbackChat
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.services.outreach_agent import CHAT_MODEL, experience_metadata, get_retriever, parse_entry, project_metadata

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 2
MAX_REQUIREMENTS = 10
MAX_EVIDENCE = 10  # retrieved chunks passed to the model and returned as evidence_used

# Bounded text and list sizes. Besides validating the response, these limits go to Ollama in
# the JSON schema, so a small model cannot loop forever inside one string and get aborted.
ShortText = Annotated[str, Field(max_length=300)]
TextList = Annotated[list[ShortText], Field(max_length=MAX_REQUIREMENTS)]


class ReviewError(Exception):
    """The review model failed or kept returning an unusable response."""


# --- Public response schema (unchanged) -------------------------------------------------


class FitAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["strong_match", "partial_match", "weak_match"]
    reasoning: Annotated[str, Field(max_length=600)]
    matching_requirements: TextList
    gaps: TextList
    uncertain_requirements: TextList


class ResumeUpdates(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required: bool
    title: Annotated[str | None, Field(max_length=120)]
    summary: Annotated[str | None, Field(max_length=500)]
    skills_to_highlight: TextList
    projects_to_highlight: TextList
    recommendations: TextList


class ReviewResult(BaseModel):
    """Exactly two top-level keys; anything else is rejected."""

    model_config = ConfigDict(extra="forbid")

    fit_assessment: FitAssessment
    resume_updates: ResumeUpdates


# --- Internal schemas for the model's three small tasks ----------------------------------


class RequirementList(BaseModel):
    requirements: Annotated[list[Annotated[str, Field(max_length=120)]], Field(max_length=MAX_REQUIREMENTS)]


class RequirementAssessment(BaseModel):
    index: Annotated[int, Field(ge=1)]
    status: Literal["matching", "gap", "uncertain"]
    evidence_quote: Annotated[str | None, Field(max_length=200)]


class Assessments(BaseModel):
    assessments: Annotated[list[RequirementAssessment], Field(max_length=MAX_REQUIREMENTS)]


class ResumeDraft(BaseModel):
    reasoning: Annotated[str, Field(max_length=600)]
    title: Annotated[str | None, Field(max_length=120)]
    summary: Annotated[str | None, Field(max_length=500)]
    skills_to_highlight: TextList
    projects_to_highlight: TextList
    recommendations: TextList


# temperature=0 (greedy decoding) makes this model fall into repetition loops that Ollama
# aborts, so use light sampling plus a repeat penalty. The output schema is passed per call.
review_agent = FallbackChat(ChatOllama(
    model=CHAT_MODEL,           # OLLAMA_CHAT_MODEL switches it, same as the tailoring and email code
    temperature=0.3,
    repeat_penalty=1.15,
    num_ctx=8192,
), label="review model")        # if a call fails, Groq answers (see app/services/llm_fallback.py)

REQUIREMENTS_PROMPT = """
You extract the requirements from a job description.

List the distinct, specific requirements the job description states: skills,
technologies, tools, experience and responsibilities. Include every technology and tool it
names, one per item. Copy the wording from the job description and keep each item short.
Do not add anything the job description does not say. Never add numbers or years of
experience that the job description does not state.
"""

ASSESSMENT_PROMPT = """
You compare a candidate against numbered job requirements, one requirement at a time.

You receive the CANDIDATE PROFILE (the complete stored profile, the source of truth) and
RETRIEVED RESUME EVIDENCE (read-only excerpts). For each numbered requirement, give:
- status "matching": the profile or evidence clearly shows it. Copy a short exact quote
  from the profile or evidence into evidence_quote.
- status "gap": the job requires it and nothing in the profile or evidence shows it.
  Search the whole profile before choosing this.
- status "uncertain": the profile or evidence is not enough to decide. evidence_quote is
  null.
Prefer evidence from work experience and projects over the skills list. Never invent
evidence. Judge each requirement on its own. Return one assessment per
requirement, using its number as `index`.
"""

DRAFT_PROMPT = """
You write the final part of a candidate review. You receive the job description, an
already-verified assessment of each requirement, and the candidate's stored profile.

Return:
- reasoning: 2-3 sentences for this job. Name the specific employer, job title or project
  (as written in the assessment) that supports each strength, and mention the main gap if
  there is one.
  Do not give generic statements about the review process.
- title: a better resume headline, only if it improves alignment with this job AND uses
  only words from the candidate's profile. Otherwise null.
- summary: a better professional summary, only if it improves alignment with this job AND
  every skill, tool, number and achievement in it appears in the candidate's profile.
  Otherwise null.
- skills_to_highlight: skills the candidate really has that this job asks for.
- projects_to_highlight: names of the candidate's own projects that are relevant to this job.
- recommendations: concrete edits to the existing resume. Never suggest adding a skill,
  tool or experience that is not in the candidate's profile.
Leave a field null or empty when nothing meaningful applies.
"""

# Profile fields the review is allowed to see; everything else stored on the row is ignored.
PROFILE_FIELDS = (
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

# Words that carry no skill information. They are ignored when deciding whether a
# requirement, skill or claim is backed by the candidate's own text.
GENERIC_WORDS = frozenset("""
a an and or of in on at to for with without the as by from into over per via vs is are be
been being was were it its this that these those their our your we you they he she
experience experienced years year months strong good great solid excellent proven demonstrated
knowledge ability able skill skills proficiency proficient familiarity familiar understanding
working work worked use using used build building built develop developing development
design designing designed implement implementing implemented
team teams plus preferred required requirement requirements must should have has had including
include such etc eg ie other related least minimum role position candidate ideal bonus nice
responsible responsibilities responsibility hands handson track record tools tool technologies
technology systems system software applications application products product solutions
senior junior lead ensure support across within also well new
apply now description job jobs about looking closely own ll ve re colleague colleagues play along alongside
""".split())

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+#_-]*")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_INVENTED_QUANTITY = re.compile(
    r"\b\d+(?:\.\d+)?\+?\s*(?:years?|yrs?|months?)?\s*(?:of\s+)?(?:experience\s*)?(?:with\s+|in\s+|using\s+)?",
    re.I,
)
_SECTION_PREFIX = re.compile(r"^(?:Work Experience|Project|Education|Professional Summary|Quantifiable Achievement):\s*")
_JUNK = re.compile(r'[\[\]{}"“”]')                               # JSON/quote debris from a small model
_THIRD_PERSON = re.compile(r"\b(?:the|this) candidate\b", re.I)    # a resume summary is not written about "the candidate"
_NEGATION = re.compile(r"\b(?:lack\w*|no|not|without|missing|gap\w*|absent|never|n't|unproven)\b", re.I)


_SUFFIXES = (("ions", 6), ("ion", 6), ("ings", 5), ("ing", 5),
             ("ers", 5), ("er", 5), ("ed", 5), ("es", 5), ("s", 3))


def _norm(token: str) -> str:
    """Lowercase, drop punctuation and crude-stem so word forms compare equal."""
    t = re.sub(r"[._-]", "", token.casefold())
    for suffix, min_stem in _SUFFIXES:
        if t.endswith(suffix) and not t.endswith("ss") and len(t) - len(suffix) >= min_stem:
            t = t[: -len(suffix)]
            break
    return t.rstrip("e") if len(t) > 4 else t


_GENERIC = frozenset(_norm(w) for w in GENERIC_WORDS)


def _tokens(text: str) -> list[str]:
    return [_norm(w) for w in _WORD.findall(text)]


def _terms(text: str) -> list[str]:
    """Distinctive terms of a text, in order and de-duplicated (no filler words or bare numbers)."""
    out: list[str] = []
    for word in _WORD.findall(text):
        # "must-have", "nice-to-have": a compound made only of filler words is filler too, not an unknown skill
        if "-" in word and all(_norm(part) in _GENERIC for part in word.split("-") if part):
            continue
        t = _norm(word)
        if len(t) >= 2 and not re.fullmatch(r"\d+\+?", t) and t not in _GENERIC and t not in out:
            out.append(t)
    return out


def _squash(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.casefold())


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for item in items:
        key = _squash(item)
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


# --- Text normalization -----------------------------------------------------------------


def _clean_text(text: str) -> str:
    """Undo the literal "/n" that older uploads stored in place of a newline."""
    return text.replace("\r\n", "\n").replace("/n", "\n")


def _as_text_list(value: Any) -> list[str]:
    """Plain strings only. Anything else (e.g. a stray {content, metadata} object) is dropped."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [_clean_text(item).strip() for item in value if isinstance(item, str) and item.strip()]


def _profile_sections(profile: dict[str, Any]) -> str:
    """Render the stored profile as text. Read-only: the stored dict is never modified."""
    parts = []
    for field in PROFILE_FIELDS:
        items = _as_text_list(profile.get(field))
        if items:
            parts.append(f"{field.upper()}:\n" + "\n---\n".join(items))
    return "\n\n".join(parts) or "(no structured profile data)"


def _profile_is_complete(profile: dict[str, Any]) -> bool:
    """True when the stored profile has real content, so absence from it means something."""
    return any(
        _as_text_list(profile.get(f))
        for f in ("experience", "projects", "skills", "programming_languages", "frameworks")
    )


def _restore_hyphens(text: str, source: str) -> str:
    """Rejoin hyphenated terms the model split ("end- to- end") when the source spells them joined."""
    source = source.casefold()

    def fix(match: re.Match) -> str:
        joined = re.sub(r"-\s+", "-", match.group(0))
        return joined if joined.casefold() in source else match.group(0)

    return re.sub(r"\w+(?:-\s+\w+)+", fix, text)


def _restore_hyphens_in(value: Any, source: str) -> Any:
    if isinstance(value, str):
        return _restore_hyphens(value, source)
    if isinstance(value, list):
        return [_restore_hyphens_in(v, source) for v in value]
    if isinstance(value, dict):
        return {k: _restore_hyphens_in(v, source) for k, v in value.items()}
    return value


# --- Model calls --------------------------------------------------------------------------


def _generate(schema: type[BaseModel], messages: list[tuple[str, str]]) -> BaseModel:
    """Call the model with a schema-constrained output; retry once if the call or output fails."""
    last_error = "unknown error"
    attempt = 0
    while True:
        attempt += 1
        # A call that fails is answered by Groq inside `review_agent`. When the local model only ever returned
        # unusable JSON, one last try goes to Groq directly.
        use_groq = attempt > MAX_ATTEMPTS
        if use_groq and not (last_error.startswith("invalid response") and llm_fallback.groq_available() and attempt == MAX_ATTEMPTS + 1):
            break
        try:
            if use_groq:
                logger.warning("Review model gave unusable output twice; asking Groq once.")
                content = llm_fallback.groq_invoke(messages, schema.model_json_schema()).content
            else:
                content = review_agent.invoke(messages, format=schema.model_json_schema()).content
        except Exception as exc:  # Ollama down, model missing, generation aborted, ...
            last_error = f"model call failed: {exc}"
            logger.warning("Review attempt %d failed: %s", attempt, exc)
            continue

        try:
            return schema.model_validate(json.loads(content))
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = f"invalid response: {str(exc).splitlines()[0]}"
            logger.warning("Review attempt %d returned an invalid response: %s", attempt, exc)
            messages = messages + [
                ("assistant", content),
                ("human", f"That response was invalid ({last_error}). Return only the requested JSON."),
            ]
    raise ReviewError(f"Profile Review Agent failed after {attempt - 1} attempts: {last_error}")


def _traceable_requirements(items: list[str], job_description: str) -> list[str]:
    """Keep only requirements that really come from the job description. Invented quantities and
    invented elaborations are cut back to the words the job description actually uses."""
    jd_all = set(_tokens(job_description))
    jd_terms = set(_terms(job_description))
    jd_squashed = _squash(job_description)
    jd_numbers = set(_NUMBER.findall(job_description))
    kept: list[str] = []
    for req in items:
        req = req.strip()
        if any(num not in jd_numbers for num in _NUMBER.findall(req)):
            # The model made up a quantity ("1 year of experience with X"); keep only the requirement.
            req = _INVENTED_QUANTITY.sub("", req).strip(" ,;:-")
        terms = _terms(req)
        if not terms:
            if req and _squash(req) in jd_squashed:
                kept.append(req)
            continue
        if len(set(terms) & jd_terms) / len(terms) < 0.6:
            # Mostly invented wording ("Redux, a powerful state management tool for ..."): keep only
            # the words the job description itself uses.
            req = " ".join(w for w in _WORD.findall(req) if _norm(w) in jd_all)
            terms = _terms(req)
        if req and terms and set(terms) & jd_terms:
            kept.append(req)
    # Drop duplicates, including items whose terms are already covered by an earlier item.
    out: list[str] = []
    covered: list[set[str]] = []
    for req in _dedupe(kept):
        terms = set(_terms(req))
        if not (terms and any(terms <= c for c in covered)):
            out.append(req)
            covered.append(terms)
    return out


def _extract_requirements(job_description: str) -> list[str]:
    """Ask the model for the job's requirements; a second pass picks up ones the first missed."""
    first = _generate(RequirementList, [("system", REQUIREMENTS_PROMPT), ("human", job_description)])
    kept = _traceable_requirements(first.requirements, job_description)
    if not kept:
        raise ReviewError("Could not extract any requirements from the job description.")

    if len(kept) < MAX_REQUIREMENTS:
        listed = "\n".join(f"- {r}" for r in kept)
        try:
            more = _generate(
                RequirementList,
                [
                    ("system", REQUIREMENTS_PROMPT),
                    (
                        "human",
                        f"{job_description}\n\nAlready listed:\n{listed}\n\n"
                        "List only the requirements from the job description that are NOT already listed. "
                        "Return an empty list if none are missing.",
                    ),
                ],
            )
        except ReviewError:
            more = RequirementList(requirements=[])  # the first pass is still usable
        covered = [set(_terms(r)) for r in kept]
        for req in _traceable_requirements(more.requirements, job_description):
            terms = set(_terms(req))
            if len(kept) < MAX_REQUIREMENTS and not any(terms and terms <= c for c in covered):
                kept.append(req)
                covered.append(terms)
    return kept


# --- Grounding: every claim is checked against the candidate's own text ------------------


class _Unit:
    """One checkable piece of the candidate's text and where it came from."""

    __slots__ = ("text", "tokens", "kind", "label")

    def __init__(self, text: str, kind: str, label: str | None = None):
        self.text, self.tokens, self.kind, self.label = text, set(_tokens(text)), kind, label

    @property
    def priority(self) -> int:  # work and project evidence outranks everything else
        # A summary line usually just restates skills, so it ranks below an actual skills entry.
        return {"work": 0, "project": 0, "tech": 0, "other": 1, "skill": 2, "summary": 3}[self.kind]


class _Best:
    def __init__(self, present: list[str], unit: "_Unit | None", has_work_evidence: bool):
        self.present, self.unit, self.has_work_evidence = present, unit, has_work_evidence

    @property
    def excerpt(self) -> str | None:
        u = self.unit
        if u is None:
            return None
        if u.kind == "skill":
            return f"listed in skills: {u.text}"
        if u.kind == "tech":
            return f"technologies include {u.text}"
        return u.text if len(u.text) <= 160 else u.text[:160].rsplit(" ", 1)[0] + "…"

    @property
    def source(self) -> str | None:
        """Where the evidence comes from, e.g. 'Backend Engineer at PayLoop' or 'project StackSearch'."""
        if self.unit is None:
            return None
        return "skills list" if self.unit.kind == "skill" else self.unit.label


class _Candidate:
    """The candidate's complete text (profile + retrieved evidence) and lookups over it."""

    def __init__(self, profile: dict[str, Any], profile_text: str, evidence: list[dict], evidence_text: str, complete: bool):
        self.text = f"{profile_text}\n{evidence_text}"
        self.tokens = set(_tokens(self.text))
        self.squashed = _squash(self.text)
        self.numbers = set(_NUMBER.findall(self.text))
        self.complete = complete
        self.units: list[_Unit] = []
        self.project_names: list[str] = []
        self.jobs: list[tuple[str, str]] = []          # the candidate's real (job title, company) pairs
        self._add_profile_units(profile)
        for item in evidence:  # retrieved chunks can hold text the structured fields lost
            meta = item["metadata"]
            if meta.get("source_section") == "skills":
                # A skills line is a list of separate skills, not one phrase: split it like the saved skills fields.
                for line in item["content"].splitlines():
                    for part in re.split(r"[,;|]", line.split(":")[-1]):
                        if part.strip():
                            self.units.append(_Unit(part.strip(), "skill"))
                continue
            label = _label_from_metadata(meta)
            kind = {"experience": "work", "projects": "project"}.get(meta.get("source_section"), "other")
            if meta.get("project_name") and meta["project_name"] not in self.project_names:
                self.project_names.append(meta["project_name"])
            self._add_lines(item["content"], kind, label if kind != "other" else None)

    @property
    def has_work(self) -> bool:
        return any(u.kind in ("work", "project", "tech") for u in self.units)

    def _add_lines(self, text: str, kind: str, label: str | None) -> None:
        for line in text.splitlines():
            line = _SECTION_PREFIX.sub("", line.strip(" \t•●▪·*-—"))
            if len(line) >= 3:
                self.units.append(_Unit(line, kind, label))

    def _add_profile_units(self, profile: dict[str, Any]) -> None:
        for raw in _as_text_list(profile.get("experience")):
            entry = parse_entry(raw)
            meta = experience_metadata(entry)
            label = f"{meta['job_title']} at {meta['company']}" if "company" in meta else entry.header[:60]
            if "company" in meta:
                self.jobs.append((meta["job_title"], meta["company"]))
            self._add_lines(entry.header, "work", label)
            self._add_lines("\n".join(entry.lines), "work", label)
        for raw in _as_text_list(profile.get("projects")):
            entry = parse_entry(raw)
            meta = project_metadata(entry)
            label = f"project {meta['project_name']}" if "project_name" in meta else "an unnamed project"
            if "project_name" in meta:
                self.project_names.append(meta["project_name"])
            for item in re.split(r"[,;]", meta.get("technologies", "")):
                if item.strip():
                    self.units.append(_Unit(item.strip(), "tech", label))
            self._add_lines("\n".join(entry.lines) or entry.header, "project", label)
        for field, label in (("summary", "summary"), ("achievements", "achievements"), ("education", "education")):
            for text in _as_text_list(profile.get(field)):
                self._add_lines(text, "summary" if field == "summary" else "other", label)
        for field in ("skills", "programming_languages", "frameworks", "databases_tools"):
            for text in _as_text_list(profile.get(field)):
                for item in re.split(r"[,;|]", text.split(":")[-1]):
                    if item.strip():
                        self.units.append(_Unit(item.strip(), "skill"))

    def mixes_up_jobs(self, text: str) -> bool:
        """True if `text` says "<Title> at <Company>" with a title that is not that company's real one,
        e.g. the title from one job pinned on the company of another."""
        title_words = {_norm(w) for title, _ in self.jobs for w in _WORD.findall(title)}
        for _, company in self.jobs:
            for m in re.finditer(rf"((?:[A-Z][\w&/.-]*\s+)*[A-Z][\w&/.-]*)\s+(?:at|@)\s+{re.escape(company)}", text):
                run = m.group(1)
                if {_norm(w) for w in _WORD.findall(run)} & title_words and not any(
                    _squash(run) == _squash(t) for t, c in self.jobs if c == company
                ):
                    return True
        return False

    def best_unit(self, terms: list[str], kinds: tuple[str, ...] | None = None) -> _Best:
        """The line/item showing the most of `terms` together. Ties go to work experience and
        projects, then other text, and only last to the skills list. `kinds` limits the search."""
        best, best_hits, work_hit = None, [], False
        for u in self.units:
            if kinds is not None and u.kind not in kinds:
                continue
            hits = [t for t in terms if t in u.tokens]
            if hits and u.kind in ("work", "project", "tech"):
                work_hit = True
            if len(hits) > len(best_hits) or (hits and len(hits) == len(best_hits) and u.priority < best.priority):
                best, best_hits = u, hits
        return _Best(best_hits, best, work_hit)

    def supports_quote(self, quote: str | None) -> bool:
        q = _squash(quote or "")
        return len(q) >= 6 and q in self.squashed


def _label_from_metadata(meta: dict[str, Any]) -> str | None:
    if meta.get("job_title") and meta.get("company"):
        return f"{meta['job_title']} at {meta['company']}"
    if meta.get("project_name"):
        return f"project {meta['project_name']}"
    return None


def _classify(req: str, a: RequirementAssessment | None, cand: _Candidate) -> tuple[str, str, str | None]:
    """Decide matching / gap / uncertain for one requirement from checked facts, not from the
    model's say-so. Returns (list name, explanation, where the evidence came from)."""
    status = a.status if a else "uncertain"
    quote_ok = bool(a and cand.supports_quote(a.evidence_quote))
    terms = _terms(req)
    words = {_norm(w): w.strip(".").lower() for w in _WORD.findall(req)}
    label = lambda ts: ", ".join(words.get(t, t) for t in ts)   # show words, not stems
    anywhere = [t for t in terms if t in cand.tokens]
    best = cand.best_unit(terms)          # terms found together in one line/item
    present = best.present
    has_number = bool(re.search(r"\d", req))  # years / quantities cannot be verified lexically

    # Show work or project evidence when it covers most of the requirement, even if a skills-list
    # line happens to contain a few more of its words.
    evidence = best
    if best.unit is not None and best.unit.kind in ("skill", "other", "summary"):
        work = cand.best_unit(terms, kinds=("work", "project", "tech"))
        if work.unit is not None and len(work.present) >= max(1, -(-len(terms) * 6 // 10)):
            evidence = work
    source = evidence.source

    def cite(shown: str, shown_best: _Best | None = None) -> str:
        """Say where the evidence is from; flag evidence that is only a skills-list entry."""
        b = shown_best or evidence
        if b.unit is None or shown != b.excerpt:
            return shown
        if b.unit.kind == "skill":
            return shown if b.has_work_evidence else f"{shown} (skills list only; no work or project evidence found)"
        return f"{b.unit.label}: {shown}" if b.unit.label else shown

    # "JavaScript and React Native": a list of separate skills. Each item is judged on its own and all must
    # be present (a phrase such as "data pipelines" has no "and", so it still needs its words together).
    items = [i for i in re.split(r",|;|\band\b|&", req, flags=re.I) if _terms(i)]
    if len(items) >= 2 and not has_number and not re.search(r"\bor\b", req, re.I):
        found_items = [(item, cand.best_unit(_terms(item))) for item in items]
        if all(f.unit is not None and len(f.present) == len(_terms(item)) for item, f in found_items):
            shown = list(dict.fromkeys(cite(f.excerpt, f) for _, f in found_items))
            lead = min((f for _, f in found_items), key=lambda f: f.unit.priority)
            return "matching", f"{req} — {'; '.join(shown)}", lead.source

    # "Expo, Firebase, or native modules": any one of the options is enough.
    if re.search(r"\bor\b", req, re.I) and not has_number:
        options = [o for o in re.split(r",|;|\bor\b", req, flags=re.I) if _terms(o)]
        met = []
        for option in options:
            option_terms = _terms(option)
            found = cand.best_unit(option_terms)
            if found.unit is not None and len(found.present) == len(option_terms):
                met.append((found.unit.priority, found))
        if len(options) >= 2 and met:
            _, found = min(met, key=lambda m: m[0])
            return "matching", f"{req} — {cite(found.excerpt, found)} (any one of the options is enough)", found.source

    if not terms:
        if status == "matching" and quote_ok:
            return "matching", f"{req} — {a.evidence_quote.strip()}", None
        return "uncertain", f"{req} — cannot be verified from the profile or resume text", None

    coverage = len(present) / len(terms)
    if coverage > 0.5:
        shown = evidence.excerpt or (a.evidence_quote.strip() if quote_ok else label(present))
        if status == "matching" and (quote_ok or not has_number):
            return "matching", f"{req} — {cite(shown)}", source
        if status != "matching" and coverage == 1 and not has_number:
            return "matching", f"{req} — {cite(shown)}", source
        if has_number:
            return "uncertain", f"{req} — relevant work is shown ({cite(shown)}), but the required amount of experience cannot be confirmed from the profile", None
        return "uncertain", f"{req} — related evidence exists ({cite(shown)}) but it does not clearly meet the requirement", None
    if anywhere:
        return "uncertain", (
            f"{req} — only partly shown (found: {label(anywhere)}; "
            f"{'not found: ' + label([t for t in terms if t not in anywhere]) if len(anywhere) < len(terms) else 'but never together in one place'})"
        ), None
    if status == "matching" and quote_ok:
        return "uncertain", f"{req} — evidence cited but its wording does not match the requirement", None
    if cand.complete:
        return "gap", f"{req} — not found anywhere in the profile or resume evidence", None
    return "uncertain", f"{req} — the stored profile is too sparse to confirm", None


def _verdict(matching: int, gaps: int, uncertain: int) -> str:
    """Derived from the verified lists so it can never contradict them. Uncertain requirements
    count for half: they are neither shown nor ruled out."""
    total = matching + gaps + uncertain
    score = (matching + 0.5 * uncertain) / total
    if score >= 0.7 and gaps / total <= 0.15:
        return "strong_match"
    if score < 0.35 or gaps / total >= 0.5:
        return "weak_match"
    return "partial_match"


def _unsupported_terms(text: str, cand: _Candidate, jd_terms: set[str]) -> list[str]:
    """Skills/proper nouns in `text` that the candidate's own text does not contain."""
    unsupported = []
    sentence_start = True
    for m in re.finditer(r"[A-Za-z0-9][A-Za-z0-9.+#_-]*|[.!?]", text):
        word = m.group(0)
        if word in ".!?":
            sentence_start = True
            continue
        first, sentence_start = sentence_start, False
        word = word.rstrip(".")
        n = _norm(word)
        if len(n) < 2 or n in _GENERIC or n in cand.tokens or re.fullmatch(r"\d+\+?", n):
            continue
        technical = bool(re.search(r"[A-Z].*[A-Z]|[0-9+#.]", word)) or (word[:1].isupper() and not first)
        if n in jd_terms or technical:
            unsupported.append(word)
    if any(num not in cand.numbers for num in _NUMBER.findall(text)):
        unsupported.append("<number not in profile>")
    return unsupported


def _sanitize_updates(draft: ResumeDraft, cand: _Candidate, jd_terms: set[str], matched: list[str]) -> ResumeUpdates:
    """Keep only suggestions the candidate's own text supports; `required` follows from the result."""
    title = (draft.title or "").strip() or None
    if title and (_JUNK.search(title) or not title[0].isalpha()):
        title = None
    if title:
        words = [_norm(w) for w in _WORD.findall(title) if _norm(w) not in {"and", "or", "of", "the", "for", "at", "in", "a", "an"}]
        aligned = bool(set(_terms(title)) & jd_terms)
        already_there = _squash(title) in cand.squashed  # a verbatim copy is the existing title, not an update
        if not words or not aligned or already_there or any(w not in cand.tokens for w in words):
            title = None

    summary = (draft.summary or "").strip() or None
    if summary and (
        _JUNK.search(summary) or _THIRD_PERSON.search(summary) or not summary[0].isalpha()
        or "|" in summary or cand.mixes_up_jobs(summary)      # header debris / a title pinned on the wrong employer
    ):
        summary = None
    if summary:
        sentences = [x for x in re.split(r"(?<=[.!?])\s+", summary) if len(_squash(x)) >= 6]
        pasted = bool(sentences) and all(_squash(x) in cand.squashed for x in sentences)  # copied, not rewritten
        aligned = len(set(_terms(summary)) & jd_terms & cand.tokens) >= min(2, len(jd_terms & cand.tokens))
        if pasted or not aligned or _unsupported_terms(summary, cand, jd_terms):
            summary = None

    # A skill must be one of the candidate's own listed skills or a requirement shown to match.
    known_skills = {_squash(u.text) for u in cand.units if u.kind == "skill"} | {_squash(m) for m in matched}
    skills = []
    for item in draft.skills_to_highlight:
        terms = _terms(item)
        if terms and _squash(item) in known_skills and set(terms) & jd_terms:
            skills.append(item.strip())

    # Only the candidate's own named projects; with none known, suggest none rather than guess.
    names = [_squash(n) for n in cand.project_names]
    projects = [
        p.strip() for p in draft.projects_to_highlight
        if len(_squash(p)) >= 3 and any(_squash(p) in n for n in names)
    ]

    recommendations = [
        r.strip() for r in draft.recommendations
        if len(r.split()) >= 4 and r.strip()[0].isalpha() and not _JUNK.search(r)
        and _squash(r) not in cand.squashed and not _unsupported_terms(r, cand, jd_terms)
    ]

    skills, projects, recommendations = _dedupe(skills)[:6], _dedupe(projects)[:6], _dedupe(recommendations)[:6]
    required = bool(title or summary or skills or projects or recommendations)
    return ResumeUpdates(
        required=required,
        title=title,
        summary=summary,
        skills_to_highlight=skills,
        projects_to_highlight=projects,
        recommendations=recommendations,
    )


def _word_safe(text: str, limit: int) -> str:
    """`text` cut to `limit` characters at a sentence or word boundary, never in the middle of a word."""
    if len(text) <= limit:
        return text
    head = text[:limit]
    sentence = max(head.rfind(". "), head.rfind("; "))
    return head[: sentence + 1] if sentence > limit // 2 else head.rsplit(" ", 1)[0].rstrip(",;:( ") + "…"


def _reasoning(draft_text: str, verdict: str, lists: dict[str, list[str]], cand: _Candidate, sources: list[str | None]) -> str:
    """Use the model's reasoning only when it talks about this job's requirements, names the job or
    project that backs a strength, and does not present a gap as a strength. Otherwise state the
    verified result, with its sources, plainly."""
    text = (draft_text or "").strip()
    heads = {key: [i.split(" — ")[0] for i in items] for key, items in lists.items()}
    req_terms = {t for items in heads.values() for item in items for t in _terms(item)}
    mentioned = req_terms & set(_terms(text))
    specific = len(mentioned) >= min(2, len(req_terms))
    work_sources = [src for src in sources if src and src != "skills list"]
    source_terms = {t for src in work_sources for t in _terms(src)}
    cites_evidence = not source_terms or bool(source_terms & set(_terms(text)))
    gap_terms = {t for item in heads["gaps"] for t in _terms(item)}
    claims_gap = bool(gap_terms & set(_tokens(text))) and not _NEGATION.search(text)
    if text and specific and cites_evidence and not claims_gap:
        return text

    def short(item: str, limit: int = 70) -> str:
        """`item` cut at a word boundary (never mid-word), with an ellipsis if it was cut."""
        item = " ".join(item.split()).rstrip(".")
        return item if len(item) <= limit else item[:limit].rsplit(" ", 1)[0].rstrip(",;:( ") + "…"

    def names(items: list[str], limit: int = 3) -> str:
        shown = "; ".join(items[:limit])
        return shown + (f"; and {len(items) - limit} more" if len(items) > limit else "")

    supported = [f"{short(h)} ({src})" if src else short(h) for h, src in zip(heads["matching"], sources)]
    total = sum(len(v) for v in lists.values())
    covered, unclear = len(lists["matching"]), len(lists["uncertain"])
    sentences = [
        f"{verdict.replace('_', ' ').capitalize()}: "
        + (f"{covered} of {total} job requirements are clearly covered." if covered else f"none of the {total} job requirements is clearly covered.")
    ]
    if supported:
        sentences.append(f"Covered: {names(supported, 2)}.")
    sentences.append(f"Missing: {names([short(g) for g in heads['gaps']])}." if heads["gaps"] else "Nothing is clearly missing.")
    if unclear:
        sentences.append(f"{unclear} other{'' if unclear == 1 else 's'} {'is' if unclear == 1 else 'are'} only partly shown (see the unclear list).")
    if not cand.has_work:
        sentences.append("The profile has no work experience or project entries to check against.")
    return " ".join(sentences)


def _section(doc) -> str | None:
    return doc.metadata.get("source_section")


def _first(docs: list, sections: tuple[str, ...], n: int) -> list:
    return [d for d in docs if _section(d) in sections][:n]


# --- Entry point ------------------------------------------------------------------------


def orchestrate(profile_id: str, profile: dict, job_description: str) -> dict:
    retriever = get_retriever(profile_id)

    if retriever is None:
        raise ValueError(
            f"No FAISS index found for profile {profile_id}"
        )

    query = f"""
    Evaluate this candidate against the following job description.
    Find relevant skills, work experience, projects, and achievements.

    Job description: {job_description}
    """

    documents = retriever.invoke(query)
    if not documents:
        raise ValueError(
            "No relevant resume evidence was retrieved."
        )

    requirements = _extract_requirements(job_description)

    # The skills chunk mentions every technology, so a plain search is dominated by it. Search
    # for work experience and projects explicitly (the indexed entries start with "Work
    # Experience:" / "Project:"), for the job as a whole and for each requirement.
    documents += _first(retriever.invoke(f"Work Experience: {job_description}"), ("experience",), 2)
    documents += _first(retriever.invoke(f"Project: {job_description}"), ("projects",), 2)
    for req in requirements:
        plain = retriever.invoke(req)
        documents += _first(retriever.invoke(f"Work Experience: {req}") + plain, ("experience",), 1)
        documents += _first(retriever.invoke(f"Project: {req}") + plain, ("projects",), 1)
        documents += plain[:1]
    # Experience and project entries first, so the cap never crowds them out.
    documents.sort(key=lambda d: 0 if _section(d) in ("experience", "projects") else 1)
    evidence, seen = [], set()
    for doc in documents:
        content = _clean_text(doc.page_content)
        if _squash(content) not in seen and len(evidence) < MAX_EVIDENCE:
            seen.add(_squash(content))
            evidence.append({"content": content, "metadata": doc.metadata})

    profile_text = _profile_sections(profile)
    evidence_text = "\n\n".join(f"[{i}] {item['content']}" for i, item in enumerate(evidence, 1))
    cand = _Candidate(profile, profile_text, evidence, evidence_text, complete=_profile_is_complete(profile))
    jd_terms = set(_terms(job_description))

    numbered = "\n".join(f"{i}. {req}" for i, req in enumerate(requirements, 1))
    assessed = _generate(
        Assessments,
        [
            ("system", ASSESSMENT_PROMPT),
            (
                "human",
                f"REQUIREMENTS:\n{numbered}\n\n"
                f"CANDIDATE PROFILE (stored, source of truth):\n{profile_text}\n\n"
                f"RETRIEVED RESUME EVIDENCE (read-only excerpts):\n{evidence_text}",
            ),
        ],
    )
    by_index = {a.index: a for a in assessed.assessments if 1 <= a.index <= len(requirements)}

    lists: dict[str, list[str]] = {"matching": [], "gaps": [], "uncertain": []}
    sources: list[str | None] = []          # where each matching requirement's evidence came from
    for i, req in enumerate(requirements, 1):
        bucket, explanation, source = _classify(req, by_index.get(i), cand)
        lists["gaps" if bucket == "gap" else bucket].append(explanation[:300])
        if bucket == "matching":
            sources.append(source)
    verdict = _verdict(len(lists["matching"]), len(lists["gaps"]), len(lists["uncertain"]))

    assessment_text = "\n".join(
        f"- {label}: {item}"
        for label, key in (("MATCHING", "matching"), ("GAP", "gaps"), ("UNCERTAIN", "uncertain"))
        for item in lists[key]
    )
    draft = _generate(
        ResumeDraft,
        [
            ("system", DRAFT_PROMPT),
            (
                "human",
                f"JOB DESCRIPTION:\n{job_description}\n\n"
                f"VERIFIED ASSESSMENT (verdict: {verdict}):\n{assessment_text}\n\n"
                f"CANDIDATE PROFILE:\n{profile_text}",
            ),
        ],
    )

    result = ReviewResult(
        fit_assessment=FitAssessment(
            verdict=verdict,
            reasoning=_word_safe(_reasoning(draft.reasoning, verdict, lists, cand, sources), 600),
            matching_requirements=lists["matching"],
            gaps=lists["gaps"],
            uncertain_requirements=lists["uncertain"],
        ),
        resume_updates=_sanitize_updates(draft, cand, jd_terms, [i.split(" — ")[0] for i in lists["matching"]]),
    )

    # Fix spacing artifacts only where the candidate's own text proves the original spelling.
    verdict_out = _restore_hyphens_in(result.model_dump(), cand.text)

    return {
        "verdict": verdict_out,
        "evidence_used": evidence,
    }
