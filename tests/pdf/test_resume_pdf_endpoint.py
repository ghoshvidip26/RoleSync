from unittest.mock import patch

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app import main
from app.services.resume_pdf_renderers import original_resume
from test_entry_grouper import Page
from test_pdf_preserving_renderer import design_c, exp

client = TestClient(main.app)

PROFILE = {"id": "p1", "name": "Alex Doe", "email": "alex@example.com"}
APPROVED_DRAFT = {
    "summary": "Platform engineer focused on payments reliability.",
    "skills": ["Kafka"],
    "experience": [exp("Acme Corp", "Senior Engineer", ["Rebuilt the ledger service", "Mentored engineers"],
                       duration="Jan 2022 - Present", location="Berlin"),
                   exp("Beta Inc", "Software Engineer", ["Cut checkout errors by half"],
                       duration="Jun 2019 - Dec 2021", location="London")],
    "education": [{"details": "BSc Computer Science | University of Leeds\n2016 - 2019"}],
}


class FakeDB:
    def __init__(self, status="approved"):
        self.rows = {"d1": {"id": "d1", "profile_id": "p1", "job_description": "jd", "draft": APPROVED_DRAFT,
                            "status": status, "created_at": "2026-10-01T10:00:00+00:00", "reviewed_at": None}}

    def fetch(self, table, id_):
        if table == "profiles":
            return PROFILE if id_ == "p1" else None
        return self.rows.get(id_)


@pytest.fixture(autouse=True)
def resume_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ORIGINAL_RESUME_DIR", str(tmp_path / "resumes"))


@pytest.fixture
def db(request):
    fake = FakeDB(getattr(request, "param", "approved"))
    with patch.object(main.db, "fetch", fake.fetch):
        yield fake


def two_column_pdf():
    p = Page((612, 792), left=40)
    for i in range(6):
        p.page.insert_text((40, 60 + i * 40), f"Left column paragraph number {i} with enough words in it", fontsize=9)
        p.page.insert_text((330, 74 + i * 40), f"Right column paragraph number {i} with enough words in it", fontsize=9)
    return p.pdf()


def upload(pdf, profile="p1", name="resume.pdf"):
    return client.post(f"/profiles/{profile}/original-resume", files={"file": (name, pdf, "application/pdf")})


def text_of(content):
    with pymupdf.open(stream=content, filetype="pdf") as d:
        return " ".join(p.get_text() for p in d)


# ---------------------------------------------------------------- gallery

def test_gallery_lists_four_templates_with_previews(db):
    body = client.get("/resume-templates").json()
    assert [t["id"] for t in body["templates"]] == ["classic", "modern", "harvard", "compact"]
    assert all(t["preview_url"].endswith("/preview") and t["ats_notes"] for t in body["templates"])
    assert body["original"] is None


def test_gallery_reports_whether_the_users_own_design_is_usable(db):
    before = client.get("/resume-templates", params={"profile_id": "p1"}).json()["original"]
    assert before["available"] is False and "uploaded" in before["reason"].lower()
    assert upload(design_c()).status_code == 200
    after = client.get("/resume-templates", params={"profile_id": "p1"}).json()["original"]
    assert after["available"] and after["preserve_supported"] and after["ats"]["risk"] == "low"


def test_template_preview_png_and_unknown_template(db):
    ok = client.get("/resume-templates/classic/preview")
    assert ok.status_code == 200 and ok.headers["content-type"] == "image/png" and ok.content.startswith(b"\x89PNG")
    assert client.get("/resume-templates/fancy/preview").status_code == 404


# ----------------------------------------------------- uploading the original

def test_upload_stores_the_pdf_and_assesses_it(db, tmp_path):
    pdf = design_c()
    body = upload(pdf).json()
    assert body["stored"] and body["preserve_supported"] and body["ats"]["risk"] == "low"
    assert original_resume.load("p1") == pdf
    status = client.get("/profiles/p1/original-resume")
    assert status.status_code == 200 and status.json()["preserve_supported"]


def test_upload_of_an_unsupported_design_is_kept_but_marked_unsupported(db):
    body = upload(two_column_pdf()).json()
    assert body["stored"] and body["preserve_supported"] is False
    assert "single-column" in body["reason"] and body["ats"]["risk"] in {"medium", "high"}


def test_upload_rejects_non_pdf_and_unknown_profiles(db):
    assert upload(b"hello", name="resume.txt").status_code == 415
    assert upload(b"%PDF-1.4 this is not really a pdf", name="r.pdf").status_code == 422
    assert upload(design_c(), profile="nope").status_code == 404
    assert client.get("/profiles/p1/original-resume").status_code == 404


def test_original_store_rejects_path_tricks():
    for bad in ("../x", "a/b", "", "x" * 80):
        with pytest.raises(ValueError):
            original_resume.save(bad, b"%PDF-1.4")


# ------------------------------------------------------------ downloading

@pytest.mark.parametrize("db", ["pending_review"], indirect=True)
def test_pdf_needs_an_approved_draft(db):
    assert client.get("/resume-drafts/d1/pdf", params={"template": "classic"}).status_code == 409


def test_without_an_original_or_template_the_user_is_asked_to_choose(db):
    r = client.get("/resume-drafts/d1/pdf")
    body = r.json()
    assert r.status_code == 409 and body["action"] == "choose_template"
    assert [t["id"] for t in body["templates"]] == ["classic", "modern", "harvard", "compact"]
    assert "No original resume" in body["detail"]


@pytest.mark.parametrize("tid", ["classic", "modern", "harvard", "compact"])
def test_every_gallery_template_downloads(db, tid):
    r = client.get("/resume-drafts/d1/pdf", params={"template": tid})
    assert r.status_code == 200 and r.content.startswith(b"%PDF")
    assert r.headers["x-resume-renderer"] == f"template:{tid}"
    text = text_of(r.content)
    assert "Alex Doe" in text and "Rebuilt the ledger service" in text and "Platform engineer" in text


def test_unknown_template_is_a_422(db):
    r = client.get("/resume-drafts/d1/pdf", params={"template": "fancy"})
    assert r.status_code == 422 and "Unknown template" in r.json()["detail"]


def test_asking_for_original_without_one_offers_the_gallery(db):
    r = client.get("/resume-drafts/d1/pdf", params={"template": "original"})
    assert r.status_code == 409 and r.json()["action"] == "choose_template"


def test_original_design_is_used_by_default_when_it_works(db):
    upload(design_c())
    r = client.get("/resume-drafts/d1/pdf")
    assert r.status_code == 200
    assert r.headers["x-resume-renderer"] == "original" and r.headers["x-resume-validation"] in {"pass", "warn"}
    text = " ".join(text_of(r.content).split())
    assert "Rebuilt the ledger service" in text and "Built payment APIs" not in text
    assert "Berlin" in text                                    # facts the draft left blank stay as in the original
    assert r.headers["x-resume-pages"] == "1"


def test_a_template_can_still_be_chosen_when_an_original_exists(db):
    upload(design_c())
    r = client.get("/resume-drafts/d1/pdf", params={"template": "harvard"})
    assert r.status_code == 200 and r.headers["x-resume-renderer"] == "template:harvard"


def test_unsupported_original_falls_back_to_choosing_a_template(db):
    upload(two_column_pdf())
    r = client.get("/resume-drafts/d1/pdf")
    body = r.json()
    assert r.status_code == 409 and body["action"] == "choose_template" and "single-column" in body["detail"]
    assert client.get("/resume-drafts/d1/pdf", params={"template": "compact"}).status_code == 200


def test_a_rebuild_that_fails_its_checks_is_never_returned(db):
    upload(design_c())
    failing = {"passed": False, "summary": {"pass": 5, "warn": 0, "fail": 1},
               "checks": [{"name": "text_complete", "status": "fail", "detail": "1 approved text(s) are missing"}]}
    with patch.object(main, "validate_rendering", return_value=failing):
        r = client.get("/resume-drafts/d1/pdf")
    body = r.json()
    assert r.status_code == 409 and body["action"] == "choose_template"
    assert body["failed_checks"][0]["name"] == "text_complete"
