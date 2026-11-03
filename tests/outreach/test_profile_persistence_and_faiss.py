from unittest.mock import MagicMock, patch
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.services import outreach_agent
from app.models.profile import Profile, ProfileCreate
from app.services import resume_service


class DummyEmbeddings(Embeddings):
    """Deterministic dummy embeddings for fast, isolated vector store testing."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t) % 10), 0.1, 0.2, 0.3] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text) % 10), 0.1, 0.2, 0.3]


@pytest.fixture
def dummy_embeddings():
    dummy = DummyEmbeddings()
    with patch("app.services.outreach_agent.get_embeddings", return_value=dummy):
        yield dummy


def test_build_documents_preserves_metadata():
    profile = Profile(
        id="user-123",
        name="Developer A",
        email="dev@example.com",
        summary="Experienced Python developer",
        programming_languages=["Python", "SQL"],
        frameworks=["FastAPI"],
        databases_tools=["PostgreSQL"],
        experience=["Software Engineer at Corp X: built APIs"],
        projects=["TaskRunner: distributed job runner in Python"],
        achievements=["Boosted system uptime to 99.99%"],
        education=["B.S. in Computer Science"],
    )

    docs = outreach_agent.build_documents_from_profile(profile, raw_text="Full resume text content")
    assert len(docs) >= 6

    # Verify sections and metadata
    metadata_types = {d.metadata.get("source_section") for d in docs}
    assert "summary" in metadata_types
    assert "skills" in metadata_types
    assert "experience" in metadata_types
    assert "projects" in metadata_types
    assert "achievements" in metadata_types
    assert "education" in metadata_types

    # Verify project name metadata
    project_doc = next(d for d in docs if d.metadata.get("source_section") == "projects")
    assert project_doc.metadata.get("project_name") == "TaskRunner"


def test_empty_documents_handling(dummy_embeddings):
    # Should not raise exception and should not create index
    outreach_agent.index_resume("empty-profile", [])
    assert outreach_agent.get_retriever("empty-profile") is None


def test_index_creation_and_cache_reuse(dummy_embeddings):
    profile_id = "test-prof-1"
    docs = [
        Document(page_content="Built payment gateway in FastAPI", metadata={"source_section": "experience"}),
        Document(page_content="Expertise in PostgreSQL and Redis", metadata={"source_section": "skills"}),
    ]

    outreach_agent.index_resume(profile_id, docs)

    # First retrieval call
    retriever_1 = outreach_agent.get_retriever(profile_id)
    assert retriever_1 is not None

    # Cached instance should be reused
    assert profile_id in outreach_agent._VECTOR_STORE_CACHE
    cached_store = outreach_agent._VECTOR_STORE_CACHE[profile_id]

    # Second retrieval call should return retriever from cached store
    retriever_2 = outreach_agent.get_retriever(profile_id)
    assert retriever_2.vectorstore is cached_store

    results = retriever_2.invoke("payment gateway")
    assert len(results) > 0
    assert any("payment gateway" in r.page_content for r in results)


def test_profile_isolation(dummy_embeddings):
    """Ensure user profile A cannot access or retrieve user profile B's documents."""
    profile_a = "user-a"
    profile_b = "user-b"

    docs_a = [Document(page_content="Secret project Apple by Candidate A", metadata={"user": "A"})]
    docs_b = [Document(page_content="Secret project Zebra by Candidate B", metadata={"user": "B"})]

    outreach_agent.index_resume(profile_a, docs_a)
    outreach_agent.index_resume(profile_b, docs_b)

    retriever_a = outreach_agent.get_retriever(profile_a)
    retriever_b = outreach_agent.get_retriever(profile_b)

    results_a = retriever_a.invoke("Secret project")
    results_b = retriever_b.invoke("Secret project")

    # A only gets A's documents
    assert all("Candidate A" in r.page_content for r in results_a)
    assert not any("Candidate B" in r.page_content for r in results_a)

    # B only gets B's documents
    assert all("Candidate B" in r.page_content for r in results_b)
    assert not any("Candidate A" in r.page_content for r in results_b)


def test_confirm_and_persist_profile_builds_index(dummy_embeddings):
    profile_data = ProfileCreate(
        name="Candidate One",
        email="candidate.one@example.com",
        github="https://github.com/cone",
        summary="Specialist in cloud architectures.",
        programming_languages=["Python", "Rust"],
        experience=["Lead Architect at TechCorp"],
    )

    with patch("app.db.models.insert") as mock_insert:
        mock_insert.side_effect = lambda table, data: dict(data)

        saved_profile = resume_service.confirm_and_persist_profile(profile_data)
        assert saved_profile.id is not None
        assert saved_profile.name == "Candidate One"

        # Index should now exist for saved_profile.id
        retriever = outreach_agent.get_retriever(saved_profile.id)
        assert retriever is not None
        results = retriever.invoke("cloud architectures")
        assert len(results) > 0
