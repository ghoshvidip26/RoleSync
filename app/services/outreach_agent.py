import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

from app.models.company import Company
from app.models.profile import Profile
from app.models.recipient import Recipient

logger = logging.getLogger(__name__)

INDEX_DIR = Path(os.getenv("INDEX_DIR", "data/indexes"))
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen:latest")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

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
def get_llm(temperature: float = 0.0) -> ChatOllama:
    """Returns ChatOllama configured with the given temperature."""
    return ChatOllama(model=CHAT_MODEL, temperature=temperature)


def build_documents_from_profile(profile: Profile, raw_text: str | None = None) -> list[Document]:
    """Build structured Documents with rich metadata from a validated profile and optional raw resume text."""
    docs: list[Document] = []

    if profile.summary and profile.summary.strip():
        docs.append(
            Document(
                page_content=f"Professional Summary: {profile.summary.strip()}",
                metadata={"document_type": "profile", "source_section": "summary"},
            )
        )

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
        docs.append(
            Document(
                page_content="\n".join(all_skills),
                metadata={"document_type": "profile", "source_section": "skills"},
            )
        )

    if profile.experience:
        for exp in profile.experience:
            if exp and exp.strip():
                docs.append(
                    Document(
                        page_content=f"Work Experience: {exp.strip()}",
                        metadata={"document_type": "profile", "source_section": "experience"},
                    )
                )

    if profile.projects:
        for proj in profile.projects:
            if proj and proj.strip():
                proj_name = proj.split(":")[0].strip() if ":" in proj else "project"
                docs.append(
                    Document(
                        page_content=f"Project: {proj.strip()}",
                        metadata={
                            "document_type": "profile",
                            "source_section": "projects",
                            "project_name": proj_name,
                        },
                    )
                )

    if profile.achievements:
        for ach in profile.achievements:
            if ach and ach.strip():
                docs.append(
                    Document(
                        page_content=f"Quantifiable Achievement: {ach.strip()}",
                        metadata={"document_type": "profile", "source_section": "achievements"},
                    )
                )

    if profile.education:
        for edu in profile.education:
            if edu and edu.strip():
                docs.append(
                    Document(
                        page_content=f"Education: {edu.strip()}",
                        metadata={"document_type": "profile", "source_section": "education"},
                    )
                )

    if raw_text and raw_text.strip():
        raw_chunks = splitter.create_documents(
            [raw_text.strip()],
            metadatas=[{"document_type": "resume", "source_section": "raw_resume"}],
        )
        docs.extend(raw_chunks)

    return docs


def index_resume(profile_id: str, documents: list[Document]) -> None:
    """Chunk documents and persist a FAISS index for this profile.
    
    Builds the index only when valid non-empty documents exist.
    """
    valid_docs = [doc for doc in documents if doc.page_content and doc.page_content.strip()]
    if not valid_docs:
        logger.warning("No valid documents provided to index for profile %s; skipping index build.", profile_id)
        return

    document_chunks = splitter.split_documents(valid_docs)
    if not document_chunks:
        return

    vector_store = FAISS.from_documents(document_chunks, get_embeddings())
    out_dir = INDEX_DIR / profile_id
    out_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(out_dir))
    _VECTOR_STORE_CACHE[profile_id] = vector_store
    logger.info("Indexed %d chunks for profile %s at %s", len(document_chunks), profile_id, out_dir)


def get_retriever(profile_id: str):
    """Retriever over this profile's resume, or None if it has no resume index."""
    if profile_id in _VECTOR_STORE_CACHE:
        return _VECTOR_STORE_CACHE[profile_id].as_retriever(
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
    "You write short, genuine, highly personalized cold outreach emails from a job seeker to a founder or hiring manager. "
    "Rules:\n"
    "1. Ground all claims strictly in the candidate's resume excerpts and evidence. Never fabricate experience, metrics, or skills.\n"
    "2. Mention only 1 or 2 relevant skills or achievements that match the recipient's role or company.\n"
    "3. Keep the email body concise (under 150 words) with a clear, low-pressure call to action.\n"
    "4. Sign off with the candidate's name and include their provided GitHub, Portfolio, and LinkedIn links in the signature if available.\n"
    "5. Do not use placeholders, brackets like [Your Name] or [Link], or hypothetical details. If you lack a fact, leave it out."
)


class OutreachAgent:
    """Writes the outreach email, grounded in the candidate's indexed resume (RAG)."""

    def run(self, profile: Profile, company: Company, recipient: Recipient) -> DraftEmail:
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

        # Prepare candidate links
        links_list: list[str] = []
        if profile.github:
            links_list.append(f"GitHub: {profile.github}")
        if profile.portfolio:
            links_list.append(f"Portfolio: {profile.portfolio}")
        if profile.linkedin:
            links_list.append(f"LinkedIn: {profile.linkedin}")
        links_str = "\n".join(links_list)

        signature_expected = f"Best regards,\n{profile.name}"
        if links_list:
            signature_expected += "\n" + "\n".join(links_list)

        user_prompt = (
            f"Candidate: {profile.name}\n"
            f"Candidate Links:\n{links_str or '(none provided)'}\n\n"
            f"Candidate Evidence:\n{context}\n\n"
            f"Recipient: {recipient.name or 'Team'} ({recipient.role or 'Hiring Contact'}) "
            f"at {company.name} {company.website or ''}\n\n"
            f"Required Sign-off:\n{signature_expected}\n\n"
            "Write the cold outreach email with a compelling subject line and concise body."
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
                    "Body:\n<Email Body>\n"
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
                raise RuntimeError(f"Email generation failed: {exc}") from exc

        body = draft.body.strip()

        # Ensure candidate links are in the signature if not already present
        if links_list:
            missing_links = [l for l in links_list if l.split(": ")[-1] not in body]
            if missing_links:
                body = body + "\n\n" + "\n".join(missing_links)

        return DraftEmail(subject=draft.subject.strip(), body=body)
