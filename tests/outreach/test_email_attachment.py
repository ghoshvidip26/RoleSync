import email as email_lib
import threading
from email import policy
from unittest.mock import MagicMock, patch

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app import main
from app.models.email import EmailStatus
from app.services import email_service
from app.services import outreach_agent as oa
from test_outreach_uses_tailored_resume import COMPANY, PROFILE, RECIPIENT, TAILORED, run_agent
from test_pdf_preserving_renderer import design_c, exp

client = TestClient(main.app)
PDF = b"%PDF-1.4 tiny"


# ---------------------------------------------------------------- email_service

@pytest.fixture
def smtp(monkeypatch):
    monkeypatch.setenv("SMTP_EMAIL", "me@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "secret")
    monkeypatch.setattr(email_service, "SMTP_EMAIL", "me@example.com")
    monkeypatch.setattr(email_service, "SMTP_PASSWORD", "secret")
    server = MagicMock()
    with patch.object(email_service.smtplib, "SMTP") as factory:
        factory.return_value.__enter__.return_value = server
        yield server


def sent_message(server):
    return server.send_message.call_args[0][0]


def test_send_email_without_attachments_is_unchanged(smtp):
    email_service.send_email("to@example.com", "Hello", "Body text")
    msg = sent_message(smtp)
    assert msg["To"] == "to@example.com" and msg["Subject"] == "Hello" and not list(msg.iter_attachments())
    assert msg.get_content().strip() == "Body text"


def test_send_email_attaches_the_pdf(smtp):
    email_service.send_email("to@example.com", "Hello", "Body text",
                             attachments=[("Alex_Doe_Resume.pdf", PDF, "application/pdf")])
    msg = email_lib.message_from_bytes(sent_message(smtp).as_bytes(), policy=policy.default)
    parts = list(msg.iter_attachments())
    assert len(parts) == 1
    assert parts[0].get_filename() == "Alex_Doe_Resume.pdf" and parts[0].get_content_type() == "application/pdf"
    assert parts[0].get_content() == PDF
    assert msg.get_body(preferencelist=("plain",)).get_content().strip() == "Body text"


@pytest.mark.parametrize("bad", [("../x.pdf", PDF, "application/pdf"), ("a/b.pdf", PDF, "application/pdf"),
                                 ("", PDF, "application/pdf"), ("r.pdf", b"", "application/pdf"),
                                 ("r.pdf", "text", "application/pdf")])
def test_unsafe_or_empty_attachments_are_refused_before_anything_is_sent(smtp, bad):
    with pytest.raises(ValueError):
        email_service.send_email("to@example.com", "Hello", "Body", attachments=[bad])
    smtp.send_message.assert_not_called()


def test_oversized_attachments_are_refused(smtp, monkeypatch):
    monkeypatch.setattr(email_service, "MAX_ATTACHMENT_BYTES", 10)
    with pytest.raises(ValueError, match="too large"):
        email_service.send_email("to@example.com", "Hello", "Body", attachments=[("r.pdf", PDF, "application/pdf")])
    smtp.send_message.assert_not_called()


# ------------------------------------------------------------- the email text

def test_note_goes_before_the_sign_off():
    body = "Hi Miles,\n\nI build things.\n\nBest regards,\nSarah"
    out = oa.with_resume_note(body)
    assert out == "Hi Miles,\n\nI build things.\n\nI've attached my resume for reference.\n\nBest regards,\nSarah"
    assert oa.with_resume_note(out) == out                               # idempotent


@pytest.mark.parametrize("sign_off", ["Thanks,", "Sincerely", "Kind regards,", "Cheers"])
def test_note_before_other_sign_offs(sign_off):
    out = oa.with_resume_note(f"Hi,\n\nText here.\n\n{sign_off}\nSam")
    assert out.index(oa.RESUME_NOTE) < out.index(sign_off)


def test_note_is_appended_when_there_is_no_sign_off():
    assert oa.with_resume_note("Hi,\n\nText.").endswith("\n\n" + oa.RESUME_NOTE)


def test_note_only_when_the_email_is_tied_to_a_tailored_resume():
    with_tailored, _ = run_agent(TAILORED)
    without, _ = run_agent(None)
    assert oa.RESUME_NOTE in with_tailored.body
    assert "attach" not in without.body.lower()


def test_the_model_still_may_not_claim_an_attachment_itself():
    from app.services.outreach_agent import _Draft, _verify_email
    with pytest.raises(ValueError, match="attachment"):
        _verify_email(_Draft(subject="Hello from Sarah", body="Hi Miles, " + "I work with Python and Redis. " * 6 +
                             "I have attached my resume."), PROFILE, COMPANY, RECIPIENT, "")


# ------------------------------------------------- approving sends the resume

APPROVED_DRAFT = {
    "summary": "Platform engineer focused on payments reliability.",
    "experience": [exp("Acme Corp", "Senior Engineer", ["Rebuilt the ledger service"], duration="Jan 2022 - Present",
                       location="Berlin"),
                   exp("Beta Inc", "Software Engineer", ["Cut checkout errors by half"], duration="Jun 2019 - Dec 2021",
                       location="London")],
}
PROFILE_ROW = {"id": "p1", "name": "Alex Doe", "email": "alex@example.com"}
RECIPIENT_ROW = {"id": "r1", "name": "Miles", "email": "miles@cyberdyne.ai"}


class World:
    """In-memory rows plus a recording stand-in for SMTP."""

    def __init__(self, resume_draft_id="d1", draft_status="approved", draft_profile="p1"):
        self.emails = {"e1": {"id": "e1", "profile_id": "p1", "company_id": "c1", "recipient_id": "r1",
                              "subject": "Hello", "body": "Hi Miles,\n\nText.\n\nBest regards,\nAlex",
                              "status": "pending_review", "resume_draft_id": resume_draft_id}}
        self.drafts = {"d1": {"id": "d1", "profile_id": draft_profile, "job_description": "jd", "draft": APPROVED_DRAFT,
                              "status": draft_status, "created_at": "2026-10-01T10:00:00+00:00", "reviewed_at": None}}
        self.sent = []
        self.lock = threading.Lock()

    def fetch(self, table, id_):
        return {"emails": self.emails, "resume_drafts": self.drafts,
                "profiles": {"p1": PROFILE_ROW}, "recipients": {"r1": RECIPIENT_ROW}}.get(table, {}).get(id_)

    def update(self, table, id_, data):
        row = self.emails[id_]
        row.update({k: getattr(v, "value", v) for k, v in data.items()})
        return row

    def claim(self, table, id_, from_status, to_status, data=None):
        """Atomic like the database: only a row still in `from_status` moves."""
        with self.lock:
            row = self.emails[id_]
            if row["status"] != getattr(from_status, "value", from_status):
                return None
            row.update({**(data or {}), "status": getattr(to_status, "value", to_status)})
            return row

    def send(self, **kwargs):
        self.sent.append(kwargs)


@pytest.fixture
def world(request, tmp_path, monkeypatch):
    monkeypatch.setenv("ORIGINAL_RESUME_DIR", str(tmp_path / "resumes"))
    w = World(**getattr(request, "param", {}))
    with patch.object(main.db, "fetch", w.fetch), patch.object(main.db, "update", w.update), \
         patch.object(main.db, "claim", w.claim), patch.object(main.email_service, "send_email", w.send):
        yield w


def pdf_text(content):
    with pymupdf.open(stream=content, filetype="pdf") as d:
        return " ".join(p.get_text() for p in d)


def test_approving_sends_the_chosen_template_as_an_attachment(world):
    r = client.post("/emails/e1/approve", params={"template": "classic"})
    assert r.status_code == 200 and r.json()["status"] == "sent"
    (call,) = world.sent
    (name, content, mime), = call["attachments"]
    assert name == "Alex_Doe_Resume.pdf" and mime == "application/pdf" and content.startswith(b"%PDF")
    text = pdf_text(content)
    assert "Alex Doe" in text and "Rebuilt the ledger service" in text and "Platform engineer" in text
    assert call["to_email"] == "miles@cyberdyne.ai" and call["from_email"] == "alex@example.com"


def test_without_a_template_the_user_is_asked_and_nothing_is_sent(world):
    r = client.post("/emails/e1/approve")
    body = r.json()
    assert r.status_code == 409 and body["action"] == "choose_template" and len(body["templates"]) == 4
    assert world.sent == [] and world.emails["e1"]["status"] == "pending_review"


def test_the_users_own_design_is_attached_when_it_works(world):
    from app.services.resume_pdf_renderers import original_resume
    pdf = design_c()
    original_resume.save("p1", pdf)
    original_resume.save_assessment("p1", original_resume.assess(pdf))
    r = client.post("/emails/e1/approve")
    assert r.status_code == 200
    content = world.sent[0]["attachments"][0][1]
    assert "Rebuilt the ledger service" in " ".join(pdf_text(content).split())


def test_an_email_without_a_tailored_resume_is_sent_exactly_as_before():
    w = World(resume_draft_id=None)
    with patch.object(main.db, "fetch", w.fetch), patch.object(main.db, "update", w.update), \
         patch.object(main.db, "claim", w.claim), patch.object(main.email_service, "send_email", w.send):
        r = client.post("/emails/e1/approve", params={"send_without_attachment": "true"})
    assert r.status_code == 200
    assert w.sent == [{"to_email": "miles@cyberdyne.ai", "subject": "Hello", "body": w.emails["e1"]["body"],
                       "from_email": "alex@example.com"}]


@pytest.mark.parametrize("world", [{"draft_status": "pending_review"}], indirect=True)
def test_a_resume_that_is_no_longer_approved_blocks_sending(world):
    assert client.post("/emails/e1/approve", params={"template": "classic"}).status_code == 409
    assert world.sent == [] and world.emails["e1"]["status"] == "pending_review"


@pytest.mark.parametrize("world", [{"draft_profile": "someone-else"}], indirect=True)
def test_a_resume_from_another_profile_is_never_attached(world):
    r = client.post("/emails/e1/approve", params={"template": "classic"})
    assert r.status_code == 409 and world.sent == []


def test_an_unknown_template_does_not_send_or_fail_the_email(world):
    r = client.post("/emails/e1/approve", params={"template": "fancy"})
    assert r.status_code == 422 and world.sent == [] and world.emails["e1"]["status"] == "pending_review"


def test_a_smtp_failure_still_marks_the_email_failed(world):
    def boom(**kwargs):
        raise RuntimeError("SMTP down")
    with patch.object(main.email_service, "send_email", boom):
        r = client.post("/emails/e1/approve", params={"template": "classic"})
    assert r.json()["status"] == "failed" and "SMTP down" in r.json()["error"]


def test_a_failed_resume_check_blocks_the_send(world):
    from app.services.resume_pdf_renderers import original_resume
    pdf = design_c()
    original_resume.save("p1", pdf)
    failing = {"passed": False, "summary": {"pass": 1, "warn": 0, "fail": 1},
               "checks": [{"name": "text_complete", "status": "fail", "detail": "missing"}]}
    with patch.object(main, "validate_rendering", return_value=failing):
        r = client.post("/emails/e1/approve")
    assert r.status_code == 409 and r.json()["failed_checks"][0]["name"] == "text_complete"
    assert world.sent == []


# ----------------------------------------------------------------- preview

def test_the_attachment_can_be_previewed_before_approving(world):
    r = client.get("/emails/e1/attachment", params={"template": "modern"})
    assert r.status_code == 200 and r.content.startswith(b"%PDF")
    assert 'filename="Alex_Doe_Resume.pdf"' in r.headers["content-disposition"]
    assert r.headers["x-resume-renderer"] == "template:modern"
    assert world.sent == [] and world.emails["e1"]["status"] == "pending_review"


def test_preview_asks_for_a_template_too(world):
    r = client.get("/emails/e1/attachment")
    assert r.status_code == 409 and r.json()["action"] == "choose_template"


@pytest.mark.parametrize("world", [{"resume_draft_id": None}], indirect=True)
def test_preview_is_404_for_an_email_without_a_tailored_resume(world):
    assert client.get("/emails/e1/attachment").status_code == 404


def test_attachment_names_are_safe():
    assert main._attachment_name({"name": "Vidip Ghosh"}) == "Vidip_Ghosh_Resume.pdf"
    assert main._attachment_name({"name": "../../etc/passwd"}) == "etc_passwd_Resume.pdf"
    assert main._attachment_name({"name": "José O'Neil"}) == "Jose_O_Neil_Resume.pdf"
    assert main._attachment_name({"name": ""}) == "Resume.pdf" and main._attachment_name({}) == "Resume.pdf"
