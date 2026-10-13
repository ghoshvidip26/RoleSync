import io
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.profile import ExtractedProfile
from app.services import resume_service
from tests.conftest import create_blank_pdf, create_sample_pdf

client = TestClient(app)


def test_upload_non_pdf_rejected():
    response = client.post(
        "/api/profiles/resume",
        files={"file": ("resume.txt", b"plain text", "text/plain")},
    )
    assert response.status_code == 415
    assert "PDF" in response.json()["detail"]


def test_upload_empty_pdf_rejected():
    response = client.post(
        "/api/profiles/resume",
        files={"file": ("resume.pdf", b"", "application/pdf")},
    )
    assert response.status_code == 422
    assert "empty" in response.json()["detail"].lower()


def test_upload_oversized_pdf_rejected():
    oversized = b"%PDF-1.4\n" + b"0" * (5 * 1024 * 1024 + 10)
    response = client.post(
        "/api/profiles/resume",
        files={"file": ("large_resume.pdf", oversized, "application/pdf")},
    )
    assert response.status_code == 413
    assert "5 MB" in response.json()["detail"]


def test_upload_corrupt_pdf_rejected():
    corrupt_data = b"%PDF-1.4\n" + b"not a valid pdf content at all corrupt junk data"
    response = client.post(
        "/api/profiles/resume",
        files={"file": ("corrupt.pdf", corrupt_data, "application/pdf")},
    )
    assert response.status_code == 422
    assert "corrupt" in response.json()["detail"].lower() or "not a valid pdf" in response.json()["detail"].lower()


def test_upload_scanned_pdf_no_text_rejected():
    blank_pdf = create_blank_pdf()
    response = client.post(
        "/api/profiles/resume",
        files={"file": ("scanned.pdf", blank_pdf, "application/pdf")},
    )
    assert response.status_code == 422
    assert "no extractable text" in response.json()["detail"].lower()


def test_successful_resume_upload_and_extraction():
    sample_text = (
        "Alice Smith\n"
        "Email: alice@example.com\n"
        "GitHub: https://github.com/alicesmith\n"
        "LinkedIn: https://linkedin.com/in/alicesmith\n\n"
        "Summary:\nExperienced backend engineer specializing in distributed systems and FastAPI.\n\n"
        "Skills:\nPython, Go, PostgreSQL, Docker\n\n"
        "Experience:\nSenior Engineer at CloudCorp (2021-Present): Scaled microservices handling 2M QPS.\n\n"
        "Projects:\nFastQueue: High-throughput distributed message queue in Go.\n"
    )
    pdf_bytes = create_sample_pdf(sample_text)

    mock_profile = ExtractedProfile(
        name="Alice Smith",
        email="alice@example.com",
        github="https://github.com/alicesmith",
        linkedin="https://linkedin.com/in/alicesmith",
        summary="Experienced backend engineer specializing in distributed systems and FastAPI.",
        programming_languages=["Python", "Go"],
        databases_tools=["PostgreSQL", "Docker"],
        skills=["Python", "Go", "PostgreSQL", "Docker"],
        experience=["Senior Engineer at CloudCorp (2021-Present): Scaled microservices handling 2M QPS."],
        projects=["FastQueue: High-throughput distributed message queue in Go."],
    )

    with patch("app.services.resume_service.outreach_agent.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = mock_profile
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        response = client.post(
            "/api/profiles/resume",
            files={"file": ("resume.pdf", pdf_bytes, "application/pdf")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Alice Smith"
        assert data["email"] == "alice@example.com"
        assert data["github"] == "https://github.com/alicesmith"
        assert "Python" in data["skills"]
        assert len(data["projects"]) == 1


def test_resume_extraction_fallback_when_structured_output_fails():
    sample_text = "Bob Jones\nEmail: bob@example.com\nSummary: Passionate developer."
    pdf_bytes = create_sample_pdf(sample_text)

    with patch("app.services.resume_service.outreach_agent.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = Exception("Structured output unsupported by model")
        mock_llm.with_structured_output.return_value = mock_structured

        # Fallback text invoke returns raw text
        mock_response = MagicMock()
        mock_response.content = "Name: Bob Jones\nEmail: bob@example.com\nSummary: Passionate developer."
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        response = client.post(
            "/api/profiles/resume",
            files={"file": ("bob_resume.pdf", pdf_bytes, "application/pdf")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Bob Jones"
        assert data["email"] == "bob@example.com"


def test_resume_extraction_llm_failure_returns_502():
    sample_text = "Bob Jones\nEmail: bob@example.com"
    pdf_bytes = create_sample_pdf(sample_text)

    with patch("app.services.resume_service.outreach_agent.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = Exception("Ollama connection refused")
        mock_llm.with_structured_output.return_value = mock_structured
        mock_llm.invoke.side_effect = Exception("Ollama connection refused")
        mock_get_llm.return_value = mock_llm

        response = client.post(
            "/api/profiles/resume",
            files={"file": ("resume.pdf", pdf_bytes, "application/pdf")},
        )

        assert response.status_code == 502
        assert "profile extraction failed" in response.json()["detail"].lower()
