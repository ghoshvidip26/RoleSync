import os
import pytest
from app.services import outreach_agent
from app.models.company import Company
from app.models.profile import Profile
from app.models.recipient import Recipient
from app.services import resume_service
from tests.conftest import create_sample_pdf


def test_live_end_to_end_resume_rag_outreach():
    """Live integration test connecting PyMuPDF, Ollama qwen:latest, nomic-embed-text FAISS, and OutreachAgent."""
    resume_text = (
        "Alex Rivera\n"
        "Email: alex.rivera@example.com\n"
        "GitHub: https://github.com/arivera\n"
        "LinkedIn: https://linkedin.com/in/arivera\n\n"
        "Summary:\nBackend software engineer with 4 years of experience building high-throughput event-driven microservices.\n\n"
        "Technical Skills:\n"
        "Languages: Python, Go\n"
        "Frameworks: FastAPI, Celery\n"
        "Databases & Tools: PostgreSQL, Kafka, Redis, Docker\n\n"
        "Work Experience:\n"
        "Senior Backend Engineer at FinTech Global (2022-Present):\n"
        "- Built financial event streaming pipeline with Kafka and FastAPI processing 50,000 events/sec.\n"
        "- Reduced database query latency by 45% via Redis caching layer.\n\n"
        "Projects:\n"
        "StreamLite: Open-source real-time event processing engine in Go and Python.\n"
    )
    pdf_bytes = create_sample_pdf(resume_text)

    # 1. Extract and analyze resume via PyMuPDF + Ollama
    extracted = resume_service.extract_and_analyze_resume(pdf_bytes)
    assert extracted.name == "Alex Rivera"
    assert "alex.rivera@example.com" in extracted.email
    assert extracted.github == "https://github.com/arivera"

    # 2. Persist profile and build FAISS index with nomic-embed-text
    profile = Profile(
        id="test-live-user-1",
        **extracted.model_dump(exclude={"id"}),
    )
    docs = outreach_agent.build_documents_from_profile(profile, raw_text=extracted.raw_resume_text)
    outreach_agent.index_resume(profile.id, docs)

    # 3. Verify retrieval
    retriever = outreach_agent.get_retriever(profile.id)
    assert retriever is not None
    results = retriever.invoke("event streaming Kafka microservices")
    assert len(results) > 0

    # 4. Generate email with OutreachAgent
    company = Company(id="c1", name="StreamScale", website="https://streamscale.io")
    recipient = Recipient(id="r1", name="David Kim", email="david@streamscale.io", role="VP of Engineering")

    agent = outreach_agent.OutreachAgent()
    draft = agent.run(profile, company, recipient)

    assert draft.subject
    assert len(draft.body) > 30
    assert "Alex Rivera" in draft.body
    assert "https://github.com/arivera" in draft.body
