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
        "/profiles/resume",
        files={"file": ("resume.txt", b"plain text", "text/plain")},
    )
    assert response.status_code == 415
    assert "PDF" in response.json()["detail"]


def test_upload_empty_pdf_rejected():
    response = client.post(
        "/profiles/resume",
        files={"file": ("resume.pdf", b"", "application/pdf")},
    )
    assert response.status_code == 422
    assert "empty" in response.json()["detail"].lower()


def test_upload_oversized_pdf_rejected():
    oversized = b"%PDF-1.4\n" + b"0" * (5 * 1024 * 1024 + 10)
    response = client.post(
        "/profiles/resume",
        files={"file": ("large_resume.pdf", oversized, "application/pdf")},
    )
    assert response.status_code == 413
    assert "5 MB" in response.json()["detail"]


def test_upload_corrupt_pdf_rejected():
    corrupt_data = b"%PDF-1.4\n" + b"not a valid pdf content at all corrupt junk data"
    response = client.post(
        "/profiles/resume",
        files={"file": ("corrupt.pdf", corrupt_data, "application/pdf")},
    )
    assert response.status_code == 422
    assert "corrupt" in response.json()["detail"].lower() or "not a valid pdf" in response.json()["detail"].lower()


def test_upload_scanned_pdf_no_text_rejected():
    blank_pdf = create_blank_pdf()
    response = client.post(
        "/profiles/resume",
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
    response = client.post(
        "/profiles/resume",
        files={"file": ("resume.pdf", create_sample_pdf(sample_text), "application/pdf")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Alice Smith"
    assert data["email"] == "alice@example.com"
    assert data["github"] == "github.com/alicesmith"                       # handles are normalised, without the scheme
    assert data["linkedin"] == "linkedin.com/in/alicesmith"
    assert data["skills"] == ["Python", "Go", "PostgreSQL", "Docker"]      # a plain skills list is read as skills
    assert data["summary"] == "Experienced backend engineer specializing in distributed systems and FastAPI."
    assert len(data["experience"]) == 1 and len(data["projects"]) == 1


def test_extraction_does_not_need_the_language_model():
    """Parsing is deterministic Python, so an unavailable model must not affect it."""
    pdf_bytes = create_sample_pdf("Bob Jones\nEmail: bob@example.com\nSummary: Passionate developer.")

    with patch("app.services.resume_service.outreach_agent.get_llm", side_effect=Exception("Ollama connection refused")):
        response = client.post(
            "/profiles/resume",
            files={"file": ("bob_resume.pdf", pdf_bytes, "application/pdf")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Bob Jones"
    assert data["email"] == "bob@example.com"


# --- uploading the real sample resume (tests/fixtures/Vidip_Resume.pdf) ----------------------------------------------

from pathlib import Path as _Path

_SAMPLE = _Path(__file__).resolve().parents[1] / "fixtures" / "Vidip_Resume.pdf"


@pytest.mark.skipif(not _SAMPLE.exists(), reason="sample resume not present")
def test_the_real_sample_resume_uploads_and_is_read():
    """Regression: its "AI/ML" skills label used to crash the upload with a KeyError (a 500)."""
    from fastapi.testclient import TestClient
    from app import main
    response = TestClient(main.app).post("/profiles/resume", files={"file": ("Vidip_Resume.pdf", _SAMPLE.read_bytes(), "application/pdf")})
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Vidip Ghosh" and body["email"] == "ghoshvidip26@gmail.com"
    assert "LLMs" in body["databases_tools"] and "React Native" in body["frameworks"] and "Python" in body["programming_languages"]
    assert len(body["projects"]) == 2 and len(body["achievements"]) == 3 and len(body["education"]) == 1


@pytest.mark.skipif(not _SAMPLE.exists(), reason="sample resume not present")
def test_each_role_of_the_sample_resume_becomes_its_own_experience_entry():
    from app.services import resume_service
    from app.services.outreach_agent import experience_metadata, parse_entry
    profile = resume_service.extract_and_analyze_resume(_SAMPLE.read_bytes())
    metas = [experience_metadata(parse_entry(e)) for e in profile.experience]
    assert [(m["job_title"], m["company"], m.get("location"), m["dates"]) for m in metas] == [
        ("Junior Mobile Application Developer", "Aforro", "Bengaluru", "Jul 2026 – Present"),
        ("React Native Intern", "Aforro", "Bengaluru", "Apr 2026 – Jul 2026"),
        ("SDE Intern", "TechKareer", "Bengaluru", "Jan 2026 – Mar 2026"),
        ("Assistant System Engineer", "Tata Consultancy Services", "Noida", "Sep 2024 – Nov 2025")]
    assert [len(parse_entry(e).lines) for e in profile.experience] == [3, 3, 3, 2]


# --- experience laid out as company / location / title / dates (no "|") ---------------------------------------------

LAYOUT_B = """Acme Corp
Berlin
Senior Engineer
Jan 2022 - Present
• Built payment APIs serving two million users across Europe
with strict latency budgets.
• Led a team of four engineers
Tech Lead
Jan 2020 - Dec 2021
• Ran the platform guild
Beta Inc
Software Engineer
Jun 2018 - Dec 2019
• Shipped the mobile checkout
"""


def test_company_location_title_dates_layout_is_split_into_one_entry_per_role():
    from app.services.resume_service import extract_experience
    entries = extract_experience(LAYOUT_B)
    assert entries == [
        "Senior Engineer | Acme Corp, Berlin\nJan 2022 - Present\n"
        "• Built payment APIs serving two million users across Europe with strict latency budgets.\n• Led a team of four engineers",
        "Tech Lead | Acme Corp, Berlin\nJan 2020 - Dec 2021\n• Ran the platform guild",     # same company: inherited
        "Software Engineer | Beta Inc\nJun 2018 - Dec 2019\n• Shipped the mobile checkout"]  # no location line


def test_a_wrapped_bullet_line_is_not_taken_for_a_heading():
    from app.services.resume_service import extract_experience
    text = ("Acme\nBerlin\nEngineer\nJan 2022 - Present\n• Did a long thing that wrapped\nonto the next line without a marker.\n"
            "• Did another thing")
    (entry,) = extract_experience(text)
    assert entry.endswith("• Did a long thing that wrapped onto the next line without a marker.\n• Did another thing")


def test_the_title_pipe_company_layout_is_unchanged():
    from app.services.resume_service import extract_experience
    text = ("Backend Engineer | PayLoop, Pune\nJun 2020 – Dec 2021\n• Cut latency 45%.\n"
            "Intern | Acme, Delhi\nJan 2019 – May 2019\n• Wrote scripts.")
    assert extract_experience(text) == ["Backend Engineer | PayLoop, Pune\nJun 2020 – Dec 2021\n• Cut latency 45%.",
                                        "Intern | Acme, Delhi\nJan 2019 – May 2019\n• Wrote scripts."]


@pytest.mark.parametrize("text", ["", "   ", "Just some prose about my career with no dates at all.",
                                  "• one bullet\n• another bullet"])
def test_text_without_dated_headings_falls_back_to_the_old_behaviour(text):
    from app.services.resume_service import extract_experience
    assert extract_experience(text) == resume_service.splitEntries(text, lambda l: "|" in l) if text.strip() else extract_experience(text) == []


def test_year_only_and_open_ended_ranges_are_dates():
    from app.services.resume_service import extract_experience
    for rng in ("2021 - 2023", "2021 – Present", "Mar 2021 – current"):
        (entry,) = extract_experience(f"Acme\nEngineer\n{rng}\n• Did things")
        assert entry.startswith("Engineer | Acme\n" + rng)
