import logging
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

from app.models.company import Company
from app.services.llm_fallback import FallbackChat
from app.models.profile import Profile
from app.models.recipient import Recipient

logger = logging.getLogger(__name__)

# Anchored to the project root, not the current directory, so indexes are found (and written to)
# the same place however the app is launched. INDEX_DIR overrides it.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = Path(os.getenv("INDEX_DIR", PROJECT_ROOT / "data" / "indexes"))
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen:latest")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:latest")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# In-memory vector store cache to avoid reloading from disk on every email generation
_VECTOR_STORE_CACHE: dict[str, FAISS] = {}


def clear_index_cache(profile_id: str | None = None) -> None:
    """Clear cached FAISS vector stores (all or for a specific profile)."""
    if profile_id:
        _VECTOR_STORE_CACHE.pop(profile_id, None)
    else:
        _VECTOR_STORE_CACHE.clear()


@lru_cache
def get_embeddings() -> OllamaEmbeddings:
    """Returns Ollama embeddings configured with the specified model."""
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


@lru_cache
def get_llm(temperature: float = 0.0) -> FallbackChat:
    """The chat model with the given temperature: local first, Groq if a call fails (see llm_fallback.py)."""
    return FallbackChat(ChatOllama(model=CHAT_MODEL, temperature=temperature), label="email model")


# A single experience/project entry stays one document up to this size. Longer entries are
# split between bullets, never inside one, and each part repeats the entry's header.
MAX_ENTRY_CHARS = 1500

_BULLET = re.compile(r"^\s*[•●▪◦·∙‣⁃*\-–—]\s+")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_LINK_LABELS = {"github", "gitlab", "live", "demo", "link", "website", "code", "source", "repo", "app", "web"}
# Line-aligned splitting for unstructured resume text: chunks start at the beginning of a line.
_raw_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=0, separators=["\n\n", "\n"])


@dataclass
class ParsedEntry:
    """One experience or project entry: its header, its date line, and complete body lines."""

    header: str
    dates: str | None = None
    lines: list[str] = field(default_factory=list)   # bullets are rendered as "• text"

    def render(self, lines: list[str] | None = None) -> str:
        head = [self.header] + ([self.dates] if self.dates else [])
        return "\n".join(head + (self.lines if lines is None else lines))


def normalize_entry_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("/n", "\n")


def _is_date_line(line: str) -> bool:
    return len(line) <= 45 and bool(_YEAR.search(line)) and bool(re.search(r"[–—·-]|present|current|\bto\b", line, re.I))


def parse_entry(text: str) -> ParsedEntry:
    """Split an entry into header / dates / body. A bullet that wraps onto the next line is joined
    back into one complete bullet."""
    raw = [l.strip() for l in normalize_entry_text(text).splitlines() if l.strip()]
    entry = ParsedEntry(header=raw[0])
    last_is_bullet = False
    for line in raw[1:]:
        if _BULLET.match(line):
            entry.lines.append("• " + _BULLET.sub("", line).strip())
            last_is_bullet = True
        elif last_is_bullet:
            entry.lines[-1] += " " + line          # wrapped continuation of the previous bullet
        elif entry.dates is None and not entry.lines and _is_date_line(line):
            entry.dates = line
        else:
            entry.lines.append(line)
    return entry


def experience_metadata(entry: ParsedEntry) -> dict[str, str]:
    """job_title / company / dates, only when the header really says so (never guessed)."""
    header, meta = entry.header, {}
    parts = [p.strip() for p in header.split("|")]
    if len(parts) >= 2 and parts[0] and parts[1]:
        meta["job_title"], meta["company"] = parts[0], parts[1].split(",")[0].strip()
        if "," in parts[1] and parts[1].split(",", 1)[1].strip():
            meta["location"] = parts[1].split(",", 1)[1].strip()
    elif re.search(r"\s+at\s+", header, re.I):
        title, rest = re.split(r"\s+at\s+", header, maxsplit=1, flags=re.I)
        company = re.sub(r"\s*[(:].*$", "", rest).split(",")[0].strip()
        if title.strip() and company:
            meta["job_title"], meta["company"] = title.strip(), company
    in_parens = re.search(r"\(([^)]*\d{4}[^)]*)\)", header)
    dates = entry.dates or (in_parens.group(1).strip() if in_parens else None)
    if dates:
        meta["dates"] = dates
    return meta



def project_metadata(entry: ParsedEntry) -> dict[str, str]:
    """Extract project name and technologies from a project heading."""
    parts = [part.strip() for part in entry.header.split("|")]

    name = parts[0]

    # Remove prefixes such as "Project:" before extracting the name.
    name = re.sub(r"^\s*project\s*:\s*", "", name, flags=re.IGNORECASE)
    name = name.strip()

    # Handle headings such as "StreamLite: Open-source engine"
    if len(parts) == 1 and ":" in name:
        name = name.split(":", 1)[0].strip()

    meta: dict[str, str] = {}

    if name and len(name.split()) <= 8 and not _BULLET.match(name):
        meta["project_name"] = name

        technologies = [
            part
            for part in parts[1:]
            if part.lower() not in _LINK_LABELS
            and not _is_date_line(part)
        ]

        if technologies:
            meta["technologies"] = ", ".join(technologies)

    return meta

def _entry_documents(raw_entries: list[str], label: str, section: str, meta_fn) -> list[Document]:
    """One Document per complete entry, prefixed with `label`, with accurate metadata. An entry
    longer than MAX_ENTRY_CHARS is split between bullets and every part repeats the header."""
    docs: list[Document] = []
    for raw in raw_entries:
        if not raw or not raw.strip():
            continue
        entry = parse_entry(raw)
        meta = {"document_type": "profile", "source_section": section, **meta_fn(entry)}
        parts = [entry.lines]
        if len(entry.render()) > MAX_ENTRY_CHARS and len(entry.lines) > 1:
            parts, current = [], []
            for line in entry.lines:
                if current and len(entry.render(current + [line])) > MAX_ENTRY_CHARS:
                    parts.append(current)
                    current = []
                current.append(line)
            parts.append(current)
        for i, lines in enumerate(parts, 1):
            docs.append(
                Document(
                    page_content=f"{label}: {entry.render(lines)}",
                    metadata={**meta, "presplit": True, **({"part": f"{i}/{len(parts)}"} if len(parts) > 1 else {})},
                )
            )
    return docs


def build_documents_from_profile(profile: Profile, raw_text: str | None = None) -> list[Document]:
    """Build structured Documents with rich metadata from a validated profile and optional raw resume text."""
    docs: list[Document] = []

    def single(content: str, section: str) -> Document:
        # Short items stay whole; only an unusually long one is left for index_resume to split.
        return Document(
            page_content=content,
            metadata={"document_type": "profile", "source_section": section, "presplit": len(content) <= MAX_ENTRY_CHARS},
        )

    if profile.summary and profile.summary.strip():
        docs.append(single(f"Professional Summary: {normalize_entry_text(profile.summary)}", "summary"))

    all_skills = []
    if profile.programming_languages:
        all_skills.extend([f"Programming Languages: {', '.join(profile.programming_languages)}"])
    if profile.frameworks:
        all_skills.extend([f"Frameworks & Libraries: {', '.join(profile.frameworks)}"])
    if profile.databases_tools:
        all_skills.extend([f"Databases & Tools: {', '.join(profile.databases_tools)}"])
    if profile.skills:
        all_skills.extend([f"Skills: {', '.join(profile.skills)}"])
    if all_skills:
        docs.append(single("\n".join(all_skills), "skills"))

    docs.extend(_entry_documents(profile.experience or [], "Work Experience", "experience", experience_metadata))
    docs.extend(_entry_documents(profile.projects or [], "Project", "projects", project_metadata))

    for ach in profile.achievements or []:
        if ach and ach.strip():
            docs.append(single(f"Quantifiable Achievement: {normalize_entry_text(ach)}", "achievements"))

    for edu in profile.education or []:
        if edu and edu.strip():
            docs.append(single(f"Education: {normalize_entry_text(edu)}", "education"))

    if raw_text and raw_text.strip():
        for chunk in _raw_splitter.create_documents(
            [raw_text.strip()],
            metadatas=[{"document_type": "resume", "source_section": "raw_resume", "presplit": True}],
        ):
            docs.append(chunk)

    return docs


def index_resume(profile_id: str, documents: list[Document]) -> None:
    """Chunk documents and persist a FAISS index for this profile.
    
    Builds the index only when valid non-empty documents exist.
    """
    valid_docs = [doc for doc in documents if doc.page_content and doc.page_content.strip()]
    if not valid_docs:
        logger.warning("No valid documents provided to index for profile %s; skipping index build.", profile_id)
        return

    # Entries built by build_documents_from_profile are already complete units; splitting them
    # again at 500 characters is what cut experience and project entries in half.
    whole = [doc for doc in valid_docs if doc.metadata.get("presplit")]
    document_chunks = whole + splitter.split_documents([d for d in valid_docs if not d.metadata.get("presplit")])
    if not document_chunks:
        return

    vector_store = FAISS.from_documents(document_chunks, get_embeddings())
    out_dir = INDEX_DIR / profile_id
    out_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(out_dir))
    _VECTOR_STORE_CACHE[profile_id] = vector_store
    logger.info("Indexed %d chunks for profile %s at %s", len(document_chunks), profile_id, out_dir)


@lru_cache
def _embedding_dimension() -> int:
    """Length of the vectors the current embedding model produces (probed once per process)."""
    return len(get_embeddings().embed_query("dimension probe"))


def _built_with_current_model(vector_store: FAISS, profile_id: str) -> bool:
    """False if the index holds vectors of a different size than the current embedding model makes,
    i.e. it was built with another model and cannot be searched (FAISS would just assert)."""
    try:
        expected = _embedding_dimension()
    except Exception:  # embedding service unreachable: skip the check, the search will report it
        return True
    if vector_store.index.d == expected:
        return True
    logger.warning(
        "Index for profile %s holds %d-dimensional vectors but %s produces %d-dimensional ones. "
        "It was built with a different embedding model; re-upload the resume to rebuild it.",
        profile_id, vector_store.index.d, EMBEDDING_MODEL, expected,
    )
    return False


def get_retriever(profile_id: str):
    """Retriever over this profile's resume, or None if it has no usable resume index."""
    if profile_id in _VECTOR_STORE_CACHE:
        cached = _VECTOR_STORE_CACHE[profile_id]
        if not _built_with_current_model(cached, profile_id):
            return None
        return cached.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},
        )

    path = INDEX_DIR / profile_id
    if not path.exists():
        return None

    try:
        vector_store = FAISS.load_local(
            str(path), get_embeddings(), allow_dangerous_deserialization=True
        )
        if not _built_with_current_model(vector_store, profile_id):
            return None
        _VECTOR_STORE_CACHE[profile_id] = vector_store
        return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},
        )
    except Exception as exc:
        logger.warning("Failed to load FAISS index for profile %s: %s", profile_id, exc)
        return None


@dataclass
class DraftEmail:
    subject: str
    body: str


class _Draft(BaseModel):
    subject: str
    body: str


SYSTEM_PROMPT = (
    "You write the middle of a short, genuine cold email from a job seeker to a founder or hiring manager. "
    "The greeting, closing, sign-off and links are added by someone else: write ONLY these two short paragraphs, "
    "separated by a blank line:\n"
    "Paragraph 1 (1-2 sentences): who the candidate is, in plain words (role or focus, and where they most recently worked).\n"
    "Paragraph 2 (2-3 sentences): the candidate's expertise and skill sets, naming specific tools or projects from the evidence.\n"
    "Rules:\n"
    "1. Ground every claim strictly in the candidate's resume excerpts and evidence. Never fabricate experience, metrics, or skills.\n"
    "2. Do not write a greeting, a closing line, a sign-off, a signature, links or bullet points.\n"
    "3. About 60 to 110 words in total. Plain words, no buzzwords, no filler such as 'I hope this email finds you well'.\n"
    "4. Do not use placeholders, brackets like [Your Name] or [Link], or hypothetical details. If you lack a fact, leave it out.\n"
    "5. This is a cold email, not a job application: never mention an application, an open position, a job posting or an attachment.\n"
    "6. Say nothing about the company beyond its name. Also write a short subject line (under 60 characters)."
)

ANSWER_PROMPT = (
    "You answer one question that a founder or hiring manager asked the candidate, in the first person, in 2-3 sentences "
    "(under 80 words). Say what the candidate can do for the role, and back it with one concrete thing from the evidence. "
    "Rules: ground every claim strictly in the evidence; never fabricate experience, metrics or skills; no greeting, "
    "sign-off, bullet points, placeholders or links; say nothing about the company beyond its name."
)


# --- Verification: the model proposes the email, code checks it, a grounded template is the fallback ---

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+#_-]*")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
_TECHNICAL_RE = re.compile(r"[A-Z].*[A-Z]|\d|[+#]")  # CamelCase, ALLCAPS, digits, C++ / C#
_PLACEHOLDER_RE = re.compile(r"\[[^\]]*\]|\{[^}]*\}|<[^>]*>|\byour name\b|\bcompany name\b", re.I)
_ATTACHMENT_RE = re.compile(r"\b(?:attach\w*|enclos\w*)\b", re.I)          # nothing is attached to these emails
_COMPANY_NEWS_RE = re.compile(          # claims about the company that nobody has given us
    r"\byour\s+(?:recent|latest|new)\b|\bcongratulat\w*|\b(?:funding|series\s+[a-e])\b"
    r"|\brecent(?:ly)?\s+(?:raised|announced|launched)\b",
    re.I,
)
_URL_RE = re.compile(
    r"https?://[^\s)>\]]+|www\.[^\s)>\]]+|\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|io|ai|dev|app|co|net|org|me|xyz|tech|in|uk)\b(?:/[^\s)>\]]*)?",
    re.I,
)
_SPLIT_DOMAIN_RE = re.compile(r"(\w)\. (com|io|ai|dev|app|co|net|org|me|xyz|tech|in|uk)\b", re.I)
_PROCESS_CLAIM_RE = re.compile(          # a cold email is not an application, and we know nothing about open roles
    r"\b(?:my|this|the)\s+application\b|\bi\s+(?:have\s+)?applied\b|\bthe\s+(?:open\s+)?(?:position|opening|vacancy)\b",
    re.I,
)
_GREETING_RE = re.compile(r"^\s*(?:hi|hello|hey|dear)\s+([^,\n]+),", re.I)
_GENERIC_GREETINGS = {"there", "team", "hiring manager", "hiring team", "founder", "founders"}


def _stem(word: str) -> str:
    w = re.sub(r"[._-]", "", word.lower())
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def _stems(text: str) -> set[str]:
    return {_stem(w) for w in _WORD_RE.findall(text)}


def _squash(text: str) -> str:
    return re.sub(r"\W+", "", text.lower())


def _unsupported(text: str, allowed: str) -> list[str]:
    """Technical-looking or capitalised words, and numbers, in `text` that `allowed` does not contain."""
    known, numbers = _stems(allowed), set(_NUMBER_RE.findall(allowed))
    out, sentence_start = [], True
    for m in re.finditer(r"[A-Za-z0-9][A-Za-z0-9.+#_-]*|[.!?\n]", text):
        word = m.group(0)
        if word in ".!?\n":
            sentence_start = True
            continue
        first, sentence_start = sentence_start, word.endswith(".")   # a glued period ends the sentence
        word = word.rstrip(".")
        if len(word) < 2 or _stem(word) in known:
            continue
        if _TECHNICAL_RE.search(word) or (word[:1].isupper() and not first):
            out.append(word)
    out += [n for n in _NUMBER_RE.findall(text) if n not in numbers]
    return list(dict.fromkeys(out))


def _candidate_context(profile: Profile, evidence: str) -> str:
    """Everything the email may draw on about the candidate: the saved profile and the retrieved evidence."""
    return f"{profile.model_dump_json()}\n{evidence}\nGitHub Portfolio LinkedIn".replace("/n", "\n")


def _anchor_terms(profile: Profile) -> tuple[set[str], set[str]]:
    """(identity, substance) words from the candidate's real background. Identity: job titles and
    employers. Substance: skills, project names and technologies; an email must contain some."""
    identity: list[str] = []
    substance: list[str] = []
    for raw in profile.experience or []:
        meta = experience_metadata(parse_entry(raw))
        identity += [meta.get("job_title", ""), meta.get("company", "")]
    for raw in profile.projects or []:
        meta = project_metadata(parse_entry(raw))
        substance += [meta.get("project_name", ""), meta.get("technologies", "")]
    for field in (profile.skills, profile.programming_languages, profile.frameworks, profile.databases_tools):
        for item in (field or [])[:30]:
            substance += [p.strip() for p in re.split(r"[,;|]", item.split(":")[-1])]
    stems = lambda parts: {s for part in parts for s in _stems(part) if len(s) > 2}
    return stems(identity), stems(substance)


def _verify_email(draft: "_Draft", profile: Profile, company: Company, recipient: Recipient, evidence: str) -> None:
    """Raise ValueError unless every claim in the email is supported by the candidate's resume or by the
    company/recipient details we were given."""
    subject, body = draft.subject.strip(), draft.body.strip()
    signs = list(_SIGN_OFF_RE.finditer(body))
    # The sign-off ("Thanks & Regards," and the name) is not a claim; checking it would flag "Regards" as an unknown word.
    text = f"{subject}\n{body[: signs[-1].start()] if signs else body}"
    context = _candidate_context(profile, evidence)
    given = f"{profile.name} {company.name} {company.website or ''} {recipient.name or ''} {recipient.role or ''}"
    allowed = f"{context}\n{given}"

    if not 3 <= len(subject) <= 100:
        raise ValueError("subject is missing or too long")
    words = len(body.split())
    if not 25 <= words <= 180:
        raise ValueError(f"body has {words} words (expected 25-180)")
    if _PLACEHOLDER_RE.search(text):
        raise ValueError("contains a placeholder such as [Name] or {Company}")
    if _PROCESS_CLAIM_RE.search(text):
        raise ValueError("refers to an application or an open position that nobody mentioned")
    if _ATTACHMENT_RE.search(text):
        raise ValueError("mentions an attachment, but nothing is attached")
    if _COMPANY_NEWS_RE.search(text):
        raise ValueError("makes a claim about the company's news or funding that we have no source for")

    for link in _URL_RE.findall(text):
        if link.rstrip(".,;:!?").lower() not in allowed.lower():
            raise ValueError(f"contains a link that is neither the candidate's nor the company's: {link}")

    greeting = _GREETING_RE.match(body)
    if greeting:
        name = greeting.group(1).strip().lower()
        if recipient.name:
            if name.split()[0] not in {w.lower() for w in recipient.name.split()}:
                raise ValueError(f"greets '{greeting.group(1).strip()}' but the recipient is {recipient.name}")
        elif name not in _GENERIC_GREETINGS:
            raise ValueError(f"greets '{greeting.group(1).strip()}', a name nobody gave us")

    bad = _unsupported(text, allowed)
    if bad:
        raise ValueError(f"mentions things not in the resume or company details: {', '.join(bad[:5])}")

    identity, substance = _anchor_terms(profile)
    if not (_stems(body) & substance):
        raise ValueError(
            "is generic: it names no skill or project from the candidate's resume"
            if (_stems(body) & identity) or not substance
            else "does not mention anything from the candidate's real background"
        )

    candidate_sq = _squash(context)
    sentences = [x for x in re.split(r"(?<=[.!?])\s+|\n+", body) if len(_squash(x)) >= 25]
    if sentences and all(_squash(x) in candidate_sq for x in sentences):
        raise ValueError("pasted from the resume instead of written for this recipient")
    lowered = [w.lower().strip(".,;:()") for w in body.split()]
    trigrams = [tuple(lowered[i : i + 3]) for i in range(len(lowered) - 2)]
    if len(set(lowered)) / len(lowered) < 0.45 or len(trigrams) != len(set(trigrams)):
        raise ValueError("repetitive")


def _tailored_facts(tailored: dict | None) -> dict:
    """What an approved tailored resume contributes: skills and a result to emphasise, and projects.
    Its recommendations and claims_requiring_confirmation are left out on purpose: those are things the
    resume does NOT show, so an email may not claim them."""
    empty = {"skills": [], "bullet": None, "bullets": [], "projects": [], "lines": [], "text": ""}
    if not tailored:
        return empty
    texts = lambda key: [t.strip() for t in (tailored.get(key) or []) if isinstance(t, str) and t.strip()]
    items = lambda key: [i for i in (tailored.get(key) or []) if isinstance(i, dict)]
    first_bullet = lambda item: next((b.strip() for b in item.get("bullets") or [] if isinstance(b, str) and b.strip()), None)

    # Two draft shapes exist. The current one holds whole roles and projects (most relevant first); older
    # saved drafts hold "skills_to_highlight", quoted "experience_edits" and "projects_to_highlight".
    skills = (texts("skills_to_highlight") or texts("skills"))[:5]
    bullets: list[str] = []
    for edit in texts("experience_edits"):
        quoted = re.search(r"“(.+?)”", edit)
        if quoted:
            bullets.append(quoted.group(1).rstrip("…").strip())
    if not bullets:
        # the draft puts the most relevant bullet of the most relevant role (then project) first
        bullets = [b for b in (first_bullet(i) for i in items("experience")[:3] + items("projects")[:2]) if b]
    bullet = bullets[0] if bullets else None
    projects = [re.split(r"[:(]", p)[0].strip() for p in texts("projects_to_highlight")][:2]
    if not projects:
        projects = [p["name"].strip() for p in items("projects") if isinstance(p.get("name"), str) and p["name"].strip()][:2]
    headline = (tailored.get("headline") or "").strip()
    summary = (tailored.get("summary") or "").strip()
    lines = []
    if headline:
        lines.append(f"Headline: {headline}")
    if skills:
        lines.append(f"Skills to emphasise: {', '.join(skills)}")
    if bullet:
        lines.append(f"A result to mention: {bullet}")
    if projects:
        lines.append(f"Projects to mention: {', '.join(projects)}")
    return {
        "skills": skills, "bullet": bullet, "bullets": bullets[:4], "projects": projects, "lines": lines,
        "text": "\n".join([headline, summary, *skills, *bullets[:4], *projects]).strip(),
    }


_BREAK_WORDS = (" to ", " using ", " by ", " for ", " with ", " via ", " across ", " while ", " so ", " that ", " and ")


def _clause_cut(text: str, limit: int) -> str:
    """`text` cut at a natural break before `limit`, without a final period or an ellipsis. In order of preference:
    the last comma or semicolon, then the last " to / using / by / for / with ..." (so the rest is dropped as a
    whole clause), then a word. Breaks inside parentheses are never used, and a parenthesis left open by the cut
    is dropped."""
    text = " ".join(text.split()).rstrip(".")
    if len(text) <= limit:
        return text
    head, depth, comma, clause = text[:limit], 0, -1, -1
    for i, ch in enumerate(head):
        depth += (ch == "(") - (ch == ")" and depth > 0)
        if depth:
            continue
        if head[i:i + 2] in (", ", "; "):
            comma = i
        elif any(head.startswith(w, i) for w in _BREAK_WORDS):
            clause = i
    if comma > limit // 2:
        out = head[:comma]
    elif clause > limit // 2:
        out = head[:clause]
    else:
        out = head.rsplit(" ", 1)[0]
    if out.count("(") > out.count(")"):                       # "...flows (pickup, delivery" -> "...flows"
        out = out[: out.rindex("(")]
    return out.rstrip(",;: ")


def _tidy_note(note: str) -> str:
    """The personal line as the candidate wrote it, made safe to put in front of the intro: stray trailing
    symbols are dropped and it ends like a sentence ("I'm looking for SWE-1/" -> "I'm looking for SWE-1.")."""
    note = " ".join(note.split()).rstrip("/\\|,;:-– ")
    if not note:
        return ""
    note = note[0].upper() + note[1:]
    return note if note.endswith((".", "!", "?")) else note + "."


# --- The email: greeting, 2 written paragraphs, proof of work, closing. Code builds everything but the paragraphs. ---

CLOSING_REGARDS = "Thanks & Regards,"
PROOF_HEADING = "A few relevant things I've worked on:"
MAX_PROOF_CHARS = 230     # one whole proof line: project or employer, the resume's own sentence, and the tech stack

# Words that signal a field, and the words in a resume that show work in it. Plain data: add a line to teach the
# ranking a new field. It only decides WHICH of the candidate's real projects and roles to show first; none of
# these words is ever written into an e-mail.
DOMAIN_HINTS = {
    "ai": "llm llms rag agent agents embedding embeddings langchain langgraph faiss gemini openai prompt ml nlp vector retrieval semantic qdrant",
    "ml": "llm rag embedding model training pytorch tensorflow nlp vector retrieval ai",
    "data": "pipeline etl analytics sql postgresql mongodb warehouse kafka spark",
    "fintech": "payment payments razorpay upi wallet cashfree stripe checkout settlement",
    "payments": "payment razorpay upi wallet cashfree stripe checkout settlement",
    "mobile": "react native android ios firebase fcm expo geofence location",
    "cloud": "aws gcp azure docker kubernetes terraform serverless",
    "devops": "docker kubernetes ci cd github actions terraform sre incident deployment",
    "security": "auth authentication oauth jwt encryption vulnerability",
    "health": "patient clinical hipaa",
}


_NOT_A_NAME = {"hi", "hello", "hey", "dear", "team", "there", "sir", "madam", "mr", "mrs", "ms", "dr", "mr.", "mrs.", "ms.", "dr."}


def _first_name(recipient: Recipient) -> str | None:
    """The recipient's first name, or None when what was entered is not a usable name (e.g. "Hi", "hr@x.com",
    "Hello Team"): a greeting word is skipped, and anything that is not a plain word gives None."""
    words = [w.strip(",.;:") for w in (recipient.name or "").split()]
    while words and words[0].lower() in _NOT_A_NAME:        # "Hi Miles" -> "Miles"; "Dr Dyson" keeps "Dyson" too
        words.pop(0)
    first = words[0] if words else ""
    if not re.fullmatch(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*", first):   # letters only (hyphen/apostrophe inside is fine)
        return None
    return first.capitalize() if first.islower() or first.isupper() else first


def _greeting(recipient: Recipient, company: Company) -> str:
    first = _first_name(recipient)
    return f"Hi {first}," if first else f"Hi {company.name} Team,"


def _current_role(profile: Profile) -> tuple[str | None, str | None]:
    """(job title, employer) of the first experience entry that states both."""
    title, employer, _ = _current_role_with_dates(profile)
    return title, employer


_STILL_THERE = re.compile(r"\b(?:present|current|currently|now|ongoing|today)\b", re.I)


def _current_role_with_dates(profile: Profile) -> tuple[str | None, str | None, bool]:
    """(job title, employer, is that job still going on?) of the first experience entry that states both."""
    for raw in profile.experience or []:
        meta = experience_metadata(parse_entry(raw))
        if meta.get("job_title") and meta.get("company"):
            return meta["job_title"], meta["company"], bool(_STILL_THERE.search(meta.get("dates", "")))
    return None, None, False


def _subject(profile: Profile, company: Company, title: str | None) -> str:
    """Short and specific (the guidance is under about 60 characters): who, what, and where."""
    for text in (f"{profile.name}: {title} interested in {company.name}" if title else "",
                 f"{profile.name}: interested in {company.name}"):
        if text and len(text) <= 70:
            return text
    return f"{profile.name}: interested in {company.name}"[:100]


def _skill_list(profile: Profile) -> list[str]:
    """Languages, frameworks and tools first; the general `skills` list also holds concepts such as "OOP"."""
    skills: list[str] = []
    for field in (profile.programming_languages, profile.frameworks, profile.databases_tools, profile.skills):
        for item in field or []:
            skills += [p.strip() for p in re.split(r"[,;|]", item.split(":")[-1]) if p.strip()]
    return list(dict.fromkeys(skills))


MAX_EXPERTISE_SKILLS = 6


def _expertise_skills(profile: Profile, tailored: dict | None) -> list[str]:
    """What the expertise paragraph names: the approved tailored resume's skills first, then the candidate's
    other languages, frameworks and tools. General concepts (OOP, ...) are used only if there are too few tools."""
    tools: list[str] = []
    for field in (profile.programming_languages, profile.frameworks, profile.databases_tools):
        for item in field or []:
            tools += [p.strip() for p in re.split(r"[,;|]", item.split(":")[-1]) if p.strip()]
    ordered = list(dict.fromkeys([*_tailored_facts(tailored)["skills"], *tools]))
    if len(ordered) < 3:
        ordered = list(dict.fromkeys([*ordered, *_skill_list(profile)]))
    return ordered[:MAX_EXPERTISE_SKILLS]


def _join_list(items: list[str]) -> str:
    """"A", "A and B", "A, B and C"."""
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}" if items else ""


_WORDS = re.compile(r"[^A-Za-z0-9+#]+")
_ADDRESS_NOISE = {"http", "https", "www", "com", "org", "net", "io", "co", "in", "app", "dev"}      # "ai" is deliberately not noise


def _wanted_terms(company: Company, recipient: Recipient, tailored: dict | None, note: str | None) -> set[str]:
    """Stems that say what this email is for: the company (name and web address, so "gnani.ai" gives "ai"),
    the recipient's role, the candidate's own personal line and the approved tailored resume's skills and headline.
    A field word such as "ai" also brings in the words that show work in that field (DOMAIN_HINTS)."""
    facts = _tailored_facts(tailored)
    raw = " ".join([company.name, company.website or "", recipient.role or "", note or "", " ".join(facts["skills"]),
                    str((tailored or {}).get("headline") or "")])
    stems = {_stem(w) for w in _WORDS.split(raw) if len(w) > 1 and w.lower() not in _ADDRESS_NOISE}
    for field, hint in DOMAIN_HINTS.items():
        if _stem(field) in stems:
            stems |= _stems(hint)
    return stems


def _evidence(profile: Profile, tailored: dict | None) -> list[dict]:
    """The candidate's real projects and roles as {kind, label, tech, bullets}. From the approved tailored
    resume when it has them (the candidate approved those words, in order of relevance), else from the profile."""
    items: list[dict] = []
    clean = lambda xs: [str(x).strip() for x in xs or [] if isinstance(x, str) and str(x).strip()]
    t = tailored or {}
    t_work = [e for e in t.get("experience") or [] if isinstance(e, dict)]
    t_projects = [e for e in t.get("projects") or [] if isinstance(e, dict)]
    if t_work or t_projects:
        for e in t_work:
            if clean(e.get("bullets")):
                items.append({"kind": "work", "label": str(e.get("company") or "").strip(), "tech": [], "bullets": clean(e["bullets"])})
        for e in t_projects:
            if clean(e.get("bullets")) and str(e.get("name") or "").strip():
                items.append({"kind": "project", "label": str(e["name"]).strip(), "tech": clean(e.get("technologies")), "bullets": clean(e["bullets"])})
        return items
    for raw in profile.experience or []:
        entry = parse_entry(raw)
        bullets = [l.lstrip("• ").strip() for l in entry.lines if l.lstrip("• ").strip()]
        if bullets:
            items.append({"kind": "work", "label": experience_metadata(entry).get("company", ""), "tech": [], "bullets": bullets})
    for raw in profile.projects or []:
        entry = parse_entry(raw)
        meta = project_metadata(entry)
        bullets = [l.lstrip("• ").strip() for l in entry.lines if l.lstrip("• ").strip()]
        if bullets and meta.get("project_name"):
            items.append({"kind": "project", "label": meta["project_name"], "tech": [t.strip() for t in meta.get("technologies", "").split(",") if t.strip()], "bullets": bullets})
    return items


def _proof_line(item: dict, bullet: str) -> str:
    """"StackSearch: Achieved sub-100 ms retrieval over 2,500+ documentation pages using Node.js, LangChain and FAISS"
    or "At Aforro: Delivered 6 rider order flows". The sentence is the resume's own, cut at a clause; the stack is
    the project's own technology list (the first five)."""
    prefix = f"{item['label']}: " if item["kind"] == "project" else (f"At {item['label']}: " if item["label"] else "")
    suffix = f" using {_join_list(item['tech'][:5])}" if item["tech"] else ""
    room = max(60, MAX_PROOF_CHARS - len(prefix) - len(suffix))
    return f"{prefix}{_clause_cut(bullet, room)}{suffix}"


def _proof_points(profile: Profile, company: Company, recipient: Recipient, tailored: dict | None,
                  limit: int = 3, skip_text: str = "", exclude: list[str] | None = None, note: str | None = None) -> list[str]:
    """The best proof of work: the candidate's real projects (with their tech stacks) and roles, ranked by how
    well they fit this company, the recipient's role, the personal line and the tailored resume. Two projects and
    a role by default, projects first. Each line is a sentence from the resume, cut at a clause, never rewritten."""
    wanted = _wanted_terms(company, recipient, tailored, note)
    covered = _stems(skip_text)

    def best_bullet(item: dict) -> str:
        return max(enumerate(item["bullets"]), key=lambda ib: (len(_stems(ib[1]) & wanted), bool(_NUMBER_RE.search(ib[1])), -ib[0]))[1]

    ranked: dict[str, list[tuple[dict, str]]] = {"project": [], "work": []}
    for index, item in sorted(enumerate(_evidence(profile, tailored)),
                              key=lambda ie: (-len(_stems(" ".join([ie[1]["label"], *ie[1]["tech"], *ie[1]["bullets"]])) & wanted), ie[0])):
        ranked[item["kind"]].append((item, best_bullet(item)))

    order: list[tuple[dict, str]] = []                      # best project, best role, second project, second role, ...
    for i in range(max(len(ranked["project"]), len(ranked["work"]))):
        order += [ranked[k][i] for k in ("project", "work") if i < len(ranked[k])]

    chosen: list[tuple[dict, str]] = []
    for item, bullet in order:
        words = _stems(bullet)
        line = _proof_line(item, bullet)
        if words and len(words & covered) / len(words) > 0.7:      # the written paragraphs already say this
            continue
        if line in (exclude or []) or any(line == l for _, l in chosen):
            continue
        chosen.append((item, line))
        if len(chosen) == limit:
            break
    chosen.sort(key=lambda pair: pair[0]["kind"] != "project")       # projects first, then the role (stable otherwise)
    return [line for _, line in chosen]


def _summary_sentence(profile: Profile) -> str:
    """The resume's own summary as a first-person sentence: "Software engineer with 1 year of experience shipping
    mobile and AI features." -> "I have 1 year of experience shipping mobile and AI features." Empty when the
    summary does not have that shape (nothing is guessed)."""
    text = " ".join((profile.summary or "").split())
    first = re.split(r"(?<=[.!?])\s+", text)[0].rstrip(".!?") if text else ""
    if re.match(r"(?i)^(?:i|i'm|i’m|i've|i’ve|we|my|our)\b", first):        # already first person: "I build things with care"
        return ""
    m = re.match(r"^(?:an?\s+)?[A-Za-z][A-Za-z /&-]{2,40}?\s+with\s+(?P<rest>.{15,})$", first)
    return f"I have {m['rest']}." if m else ""


def _template_paragraphs(profile: Profile, tailored: dict | None) -> list[str]:
    """Intro and expertise paragraphs built only from the candidate's real background."""
    title, employer, ongoing = _current_role_with_dates(profile)
    first = profile.name.split()[0]
    article = "an" if title and title[0].lower() in "aeiou" else "a"
    if title and employer and ongoing:
        intro = f"I'm {first}, currently {article} {title} at {employer}."
    elif title and employer:
        intro = f"I'm {first}. Most recently I've worked as {article} {title} at {employer}."
    else:
        intro = f"I'm {first}."
    summary = _summary_sentence(profile)
    if summary:
        intro += f" {summary}"
    skills = _expertise_skills(profile, tailored)     # the approved tailored resume decides what to lead with
    paragraphs = [intro]
    if skills:
        paragraphs.append(f"I work mainly with {_join_list(skills)}.")
    return paragraphs


def _assemble(profile: Profile, company: Company, recipient: Recipient, paragraphs: list[str],
              proof: list[str], tailored: bool, sections: list[tuple[str, str, list[str]]] | None = None,
              note: str | None = None) -> str:
    """greeting, paragraphs, proof of work, any answered questions, closing, sign-off."""
    paragraphs = [p.strip() for p in paragraphs if p and p.strip()]
    note = _tidy_note(note or "")
    if note:                                    # the candidate's own words, with only the ending tidied
        paragraphs = [f"{note} {paragraphs[0]}" if paragraphs else note, *paragraphs[1:]]
    out = [_greeting(recipient, company), ""]
    for p in paragraphs:
        out += [p, ""]
    if proof:
        out += [PROOF_HEADING, *[f"• {b}" for b in proof], ""]
    for question, answer, bullets in sections or []:
        out += [question.strip(), answer.strip()]
        if bullets:
            out += ["", PROOF_HEADING, *[f"• {b}" for b in bullets]]
        out.append("")
    closing = RESUME_NOTE + " " if tailored else ""
    closing += f"I'd welcome a short conversation about opportunities at {company.name}, and I look forward to hearing from you."
    out += [closing, "", CLOSING_REGARDS, profile.name]
    return "\n".join(out)


def _template_email(profile: Profile, company: Company, recipient: Recipient, tailored: dict | None = None,
                    sections: list[tuple[str, str, list[str]]] | None = None, note: str | None = None) -> "DraftEmail":
    """A plain email built only from the candidate's real background and the details we were given."""
    title, _ = _current_role(profile)
    paragraphs = _template_paragraphs(profile, tailored)
    proof = _proof_points(profile, company, recipient, tailored, skip_text=" ".join(paragraphs), note=note)
    body = _assemble(profile, company, recipient, paragraphs, proof, bool(tailored), sections, note)
    return DraftEmail(subject=_subject(profile, company, title), body=body)


def _template_answer(profile: Profile, company: Company, recipient: Recipient, tailored: dict | None,
                     shown: list[str] | None = None) -> tuple[str, list[str]]:
    """A grounded answer to "why you?": what the candidate works with, then real proof of work not already shown."""
    title, _ = _current_role(profile)
    skills = _expertise_skills(profile, tailored)[:4]
    first = profile.name.split()[0]
    if title:
        article = "an" if title[0].lower() in "aeiou" else "a"
        answer = f"I'm {first}, {article} {title}" + (f" with hands-on experience in {_join_list(skills)}." if skills else ".")
    else:
        answer = f"I have hands-on experience in {_join_list(skills)}." if skills else f"I'm {first}."
    answer += f" I'd be glad to put that experience to work at {company.name}."
    return answer, _proof_points(profile, company, recipient, tailored, limit=2, exclude=shown)


RESUME_NOTE = "I've attached my resume for reference."
_SIGN_OFF_RE = re.compile(
    r"^[ \t]*(?:thanks[ \t]*(?:&|and)[ \t]*regards|best regards|kind regards|warm regards|best wishes|best|regards|sincerely"
    r"|thanks|thank you|cheers)[ \t]*,?[ \t]*$",
    re.I | re.M,
)


def with_resume_note(body: str) -> str:
    """The body with one sentence saying the resume is attached, placed before the sign-off.

    Added by code, after verification: the model may not mention an attachment itself (nothing it can
    see is attached), but an email linked to a tailored resume really is sent with that PDF.
    """
    if re.search(r"\battach\w*", body, re.I):
        return body
    matches = list(_SIGN_OFF_RE.finditer(body))
    if matches:
        at = matches[-1].start()
        return f"{body[:at].rstrip()}\n\n{RESUME_NOTE}\n\n{body[at:]}"
    return f"{body.rstrip()}\n\n{RESUME_NOTE}"


class _Answer(BaseModel):
    answer: str


def _middle(body: str) -> str:
    """The written paragraphs of a model email: its own greeting and sign-off (if it added any) are dropped,
    because the greeting and closing are built by code."""
    text = body.replace("\r\n", "\n").strip()
    text = re.sub(r"^\s*(?:hi|hello|hey|dear)\b[^\n,]{0,40},[ \t]*\n*", "", text, count=1, flags=re.I)
    signs = list(_SIGN_OFF_RE.finditer(text))
    if signs:
        text = text[: signs[-1].start()]
    return text.strip()


class OutreachAgent:
    """Writes the outreach email, grounded in the candidate's indexed resume (RAG).

    The email is: greeting, two written paragraphs (who the candidate is; their expertise), a few real
    pieces of proof of work, the answers to any questions asked, a closing line, and the sign-off.
    The model writes only the two paragraphs (and the answers); everything else is built by code, and
    anything the model writes is verified against the resume first.
    """

    def run(self, profile: Profile, company: Company, recipient: Recipient, tailored: dict | None = None,
            questions: list[str] | None = None, personal_note: str | None = None) -> DraftEmail:
        retriever = get_retriever(profile.id)
        retrieved_texts: list[str] = []

        query = f"Relevant experience, technical projects, and skills for {company.name} {recipient.role or ''}".strip()
        if retriever:
            try:
                retrieved_docs = retriever.invoke(query)
                retrieved_texts = [doc.page_content for doc in retrieved_docs if doc.page_content.strip()]
            except Exception as exc:
                logger.warning("Retrieval invocation failed for profile %s: %s", profile.id, exc)

        # Fallback evidence from profile attributes if retrieval returned nothing
        if not retrieved_texts:
            fallback_parts = []
            if profile.summary:
                fallback_parts.append(f"Summary: {profile.summary}")
            if profile.skills:
                fallback_parts.append(f"Skills: {', '.join(profile.skills)}")
            if profile.programming_languages:
                fallback_parts.append(f"Languages: {', '.join(profile.programming_languages)}")
            if profile.frameworks:
                fallback_parts.append(f"Frameworks: {', '.join(profile.frameworks)}")
            if profile.projects:
                fallback_parts.append(f"Projects: {'; '.join(profile.projects[:2])}")
            if profile.experience:
                fallback_parts.append(f"Experience: {'; '.join(profile.experience[:2])}")
            if fallback_parts:
                retrieved_texts = fallback_parts

        context = "\n\n".join(retrieved_texts) if retrieved_texts else "(no specific resume evidence available)"
        evidence = "\n\n".join(retrieved_texts)
        emphasis = _tailored_facts(tailored)
        if emphasis["text"]:
            evidence = f"{evidence}\n{emphasis['text']}"      # derived from the resume, so the email may use it

        # Prepare candidate links
        links_list: list[str] = []
        if profile.github:
            links_list.append(f"GitHub: {profile.github}")
        if profile.portfolio:
            links_list.append(f"Portfolio: {profile.portfolio}")
        if profile.linkedin:
            links_list.append(f"LinkedIn: {profile.linkedin}")

        tailored_block = (
            "Approved tailored resume highlights (lead with these):\n" + "\n".join(emphasis["lines"]) + "\n\n"
            if emphasis["lines"]
            else ""
        )

        user_prompt = (
            f"Candidate: {profile.name}\n"
            f"Candidate Evidence:\n{context}\n\n"
            f"{tailored_block}"
            f"Recipient: {recipient.name or 'Team'} ({recipient.role or 'Hiring Contact'}) "
            f"at {company.name} {company.website or ''}\n\n"
            "Write the two paragraphs and a short subject line."
        )

        llm = get_llm()
        draft: _Draft | None = None

        # 1. Primary: Pydantic structured output
        try:
            draft = llm.with_structured_output(_Draft).invoke(
                [("system", SYSTEM_PROMPT), ("human", user_prompt)]
            )
        except Exception as exc:
            logger.warning("Structured output invocation failed: %s; falling back to direct prompt parsing", exc)

        # 2. Fallback: Direct generation with subject/body extraction
        if draft is None or not draft.subject or not draft.body:
            try:
                fallback_prompt = (
                    f"{user_prompt}\n\n"
                    "Output format:\n"
                    "Subject: <Subject Line>\n\n"
                    "Body:\n<the two paragraphs>\n"
                )
                response = llm.invoke(
                    [("system", SYSTEM_PROMPT), ("human", fallback_prompt)]
                )
                raw_text = response.content if hasattr(response, "content") else str(response)

                subject = f"Connecting regarding opportunities at {company.name}"
                body = raw_text.strip()
                if "Subject:" in raw_text:
                    parts = raw_text.split("Subject:", 1)[1]
                    if "\nBody:" in parts:
                        sub_part, body_part = parts.split("\nBody:", 1)
                        subject = sub_part.strip()
                        body = body_part.strip()
                    elif "\n\n" in parts:
                        sub_part, body_part = parts.split("\n\n", 1)
                        subject = sub_part.strip()
                        body = body_part.strip()

                draft = _Draft(subject=subject, body=body)
            except Exception as exc:
                logger.warning("Email generation failed (%s); using the grounded template instead", exc)
                draft = None

        # The model proposes; code verifies. Text with any unsupported claim is replaced by paragraphs built
        # only from the candidate's real background (the email still goes to human review).
        paragraphs: list[str] = []
        subject = None
        if draft is not None:
            draft = _Draft(subject=_SPLIT_DOMAIN_RE.sub(r"\1.\2", draft.subject), body=_SPLIT_DOMAIN_RE.sub(r"\1.\2", draft.body))
            try:
                _verify_email(draft, profile, company, recipient, evidence)
                paragraphs = [x.strip() for x in re.split(r"\n\s*\n", _middle(draft.body)) if x.strip()]
                subject = draft.subject.strip() if len(draft.subject.strip()) <= 70 else None
            except ValueError as exc:
                logger.warning("Model-written email not used (grounded template used instead): %s", exc)
        title, _ = _current_role(profile)
        if not paragraphs:
            paragraphs = _template_paragraphs(profile, tailored)
        subject = subject or _subject(profile, company, title)

        proof = _proof_points(profile, company, recipient, tailored, skip_text=" ".join(paragraphs), note=personal_note)
        sections, shown = [], list(proof)
        for q in (questions or []):
            answer, bullets = self._answer(llm, q, profile, company, recipient, tailored, evidence, context,
                                           tailored_block, shown)
            sections.append((q, answer, bullets))
            shown += bullets
        body = _assemble(profile, company, recipient, paragraphs, proof, bool(tailored), sections, personal_note)

        # Candidate links go under the signature
        if links_list:
            missing_links = [l for l in links_list if l.split(": ")[-1] not in body]
            if missing_links:
                body = body + "\n" + "\n".join(missing_links)

        return DraftEmail(subject=subject, body=body)

    def _answer(self, llm, question: str, profile: Profile, company: Company, recipient: Recipient,
                tailored: dict | None, evidence: str, context: str, tailored_block: str,
                shown: list[str] | None = None) -> tuple[str, list[str]]:
        """(answer, proof bullets) for one question such as "Why should you choose me?". The model proposes,
        code verifies, and a grounded answer is the fallback."""
        prompt = (
            f"Question: {question}\n\nCandidate: {profile.name}\nCandidate Evidence:\n{context}\n\n{tailored_block}"
            f"Company: {company.name} {company.website or ''}\nRole asked about: {recipient.role or 'not stated'}"
        )
        try:
            out = llm.with_structured_output(_Answer).invoke([("system", ANSWER_PROMPT), ("human", prompt)])
            text = _SPLIT_DOMAIN_RE.sub(r"\1.\2", (out.answer or "").strip())
            if "\n" in text or not text:
                raise ValueError("answer must be a single paragraph")
            _verify_email(_Draft(subject="answer", body=text), profile, company, recipient, evidence)
            return text, _proof_points(profile, company, recipient, tailored, limit=2, skip_text=text, exclude=shown)
        except Exception as exc:
            logger.warning("Model-written answer not used (grounded answer used instead): %s", exc)
            return _template_answer(profile, company, recipient, tailored, shown)
