"""The new outreach endpoints: atomic sending, retry, the no-attachment rule, independent regeneration, attachment info."""
import threading
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app import main
from app.models.company import Company
from app.models.profile import Profile
from app.models.recipient import Recipient
from app.services.outreach_agent import DraftEmail
from app.services.outreach_personal import AnswerResult, PersonalLine
from test_email_attachment import World, PROFILE_ROW

client = TestClient(main.app)


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setenv("ORIGINAL_RESUME_DIR", str(tmp_path / "resumes"))
    w = World(resume_draft_id="d1")
    w.writes = []
    real_update = w.update

    def update(table, id_, data):
        w.writes.append((table, id_, dict(data)))
        return real_update(table, id_, data)

    w.update = update
    with patch.object(main.db, "fetch", w.fetch), patch.object(main.db, "update", update), \
         patch.object(main.db, "claim", w.claim), patch.object(main.email_service, "send_email", w.send):
        yield w


# ----------------------------------------------------------------------------- atomic send

def test_two_simultaneous_approvals_send_once_and_the_second_gets_409(world):
    gate = threading.Barrier(2)
    original = world.send

    def slow_send(**kw):
        original(**kw)

    results = []

    def approve():
        gate.wait()
        results.append(client.post("/emails/e1/approve", params={"template": "classic"}).status_code)

    threads = [threading.Thread(target=approve) for _ in range(2)]
    with patch.object(main.email_service, "send_email", slow_send):
        [t.start() for t in threads]
        [t.join() for t in threads]
    assert sorted(results) == [200, 409]
    assert len(world.sent) == 1 and world.emails["e1"]["status"] == "sent"


def test_the_email_is_claimed_as_sending_before_anything_is_sent(world):
    seen = []
    world.send = lambda **kw: seen.append(world.emails["e1"]["status"])
    with patch.object(main.email_service, "send_email", world.send):
        assert client.post("/emails/e1/approve", params={"template": "classic"}).status_code == 200
    assert seen == ["sending"] and world.emails["e1"]["status"] == "sent"


@pytest.mark.parametrize("status", ["sending", "sent", "rejected"])
def test_an_email_that_is_not_pending_cannot_be_approved(world, status):
    world.emails["e1"]["status"] = status
    assert client.post("/emails/e1/approve", params={"template": "classic"}).status_code == 409
    assert world.sent == []


def test_a_failed_send_keeps_the_content_and_needs_confirmation_to_retry(world):
    def boom(**kw):
        raise RuntimeError("SMTP down")
    with patch.object(main.email_service, "send_email", boom):
        failed = client.post("/emails/e1/approve", params={"template": "classic"}).json()
    assert failed["status"] == "failed" and "SMTP down" in failed["error"]
    assert world.emails["e1"]["body"].startswith("Hi Miles") and world.emails["e1"]["subject"] == "Hello"

    asked = client.post("/emails/e1/approve", params={"template": "classic"})
    assert asked.status_code == 409 and asked.json()["action"] == "confirm_retry" and "SMTP down" in asked.json()["detail"]
    assert world.sent == []

    ok = client.post("/emails/e1/approve", params={"template": "classic", "confirm_retry": "true"})
    assert ok.status_code == 200 and ok.json()["status"] == "sent" and len(world.sent) == 1


def test_an_email_that_already_has_a_sent_time_is_never_retried(world):
    world.emails["e1"].update(status="failed", sent_at="2026-10-01T10:00:00+00:00")
    r = client.post("/emails/e1/approve", params={"template": "classic", "confirm_retry": "true"})
    assert r.status_code == 409 and world.sent == []


# ----------------------------------------------------------------------------- no attachment

def test_without_an_approved_resume_sending_needs_an_explicit_confirmation():
    w = World(resume_draft_id=None)
    with patch.object(main.db, "fetch", w.fetch), patch.object(main.db, "update", w.update), \
         patch.object(main.db, "claim", w.claim), patch.object(main.email_service, "send_email", w.send):
        asked = client.post("/emails/e1/approve")
        assert asked.status_code == 409 and asked.json()["action"] == "confirm_no_attachment"
        assert w.sent == [] and w.emails["e1"]["status"] == "pending_review"
        ok = client.post("/emails/e1/approve", params={"send_without_attachment": "true"})
    assert ok.status_code == 200 and "attachments" not in w.sent[0]       # the original upload is never attached


def test_attachment_info_gives_the_exact_filename_to_confirm(world):
    r = client.get("/emails/e1/attachment-info", params={"template": "classic"})
    assert r.status_code == 200
    body = r.json()
    assert body["has_attachment"] is True and body["filename"] == "Alex_Doe_Resume.pdf" and body["size_bytes"] > 100


def test_attachment_info_without_a_resume_says_so():
    w = World(resume_draft_id=None)
    with patch.object(main.db, "fetch", w.fetch):
        assert client.get("/emails/e1/attachment-info").json() == {"has_attachment": False, "filename": None}


def test_attachment_info_asks_for_a_template_like_the_send_does(world):
    r = client.get("/emails/e1/attachment-info")
    assert r.status_code == 409 and r.json()["action"] == "choose_template"


# ----------------------------------------------------------------------------- regeneration

class FakeAgent:
    def __init__(self):
        self.calls = []

    def run(self, *a, **kw):
        self.calls.append(kw)
        if kw.get("part") == "subject":
            return DraftEmail(subject="New subject | Alex", body="", words=0)
        return DraftEmail(subject="ignored", body="Hi Miles,\n\nNew body.\n\nThanks & Regards,\nAlex", words=3, source="template")


@pytest.fixture
def regen(world):
    world.emails["e1"]["resume_draft_id"] = None
    fake = FakeAgent()
    inputs = (Profile(id="p1", name="Alex Doe", email="alex@example.com"), Company(id="c1", name="Acme"),
              Recipient(id="r1", name="Miles", email="miles@cyberdyne.ai"), None, "jd text")
    with patch.object(main, "agent", fake), patch.object(main, "_outreach_inputs", return_value=inputs):
        yield world, fake


def test_regenerating_the_subject_returns_only_a_subject_and_writes_nothing(regen):
    world, fake = regen
    r = client.post("/emails/e1/regenerate", json={"part": "subject", "variant": 2})
    assert r.json() == {"part": "subject", "subject": "New subject | Alex"}
    assert fake.calls[0]["part"] == "subject" and fake.calls[0]["variant"] == 2
    assert world.writes == [] and world.emails["e1"]["body"].startswith("Hi Miles,\n\nText.")


def test_regenerating_the_body_returns_only_a_body_and_keeps_the_users_note_and_answers(regen):
    world, fake = regen
    r = client.post("/emails/e1/regenerate", json={"part": "body", "personal_note": "I like Acme's payments work.",
                                                    "answers": [{"question": "Why you?", "answer": "I ship."}]})
    out = r.json()
    assert out["part"] == "body" and "subject" not in out and out["body"].startswith("Hi Miles")
    assert fake.calls[0]["personal_note"] == "I like Acme's payments work."
    assert fake.calls[0]["answers"] == [{"question": "Why you?", "answer": "I ship."}]
    assert fake.calls[0]["job_description"] == "jd text"
    assert world.writes == [] and world.emails["e1"]["subject"] == "Hello"


def test_a_sent_email_cannot_be_regenerated(regen):
    world, _ = regen
    world.emails["e1"]["status"] = "sent"
    assert client.post("/emails/e1/regenerate", json={"part": "body"}).status_code == 409


def test_a_model_failure_while_regenerating_is_a_clear_error_and_changes_nothing(regen):
    world, fake = regen
    fake.run = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("model down"))
    r = client.post("/emails/e1/regenerate", json={"part": "body"})
    assert r.status_code == 502 and "model down" in r.json()["detail"] and world.writes == []


# ----------------------------------------------------------------------------- personal line / answer

def test_personal_line_and_answer_endpoints_are_stateless(world):
    inputs = (Profile(id="p1", name="Alex Doe", email="alex@example.com"), Company(id="c1", name="Acme"),
              Recipient(id="r1", name=None, email="team@acme.com"), None, None)
    line = PersonalLine(text="I read that Acme does X.", basis="job_description", source="template", evidence=["Acme does X."])
    ans = AnswerResult(answer="I ship.", evidence=["At Y: shipped"], source="template", words=2)
    ids = {"profile_id": "p1", "company_id": "c1", "recipient_id": "r1"}
    with patch.object(main, "_outreach_inputs", return_value=inputs), \
         patch.object(main.outreach_personal, "generate_personal_line", return_value=line) as g, \
         patch.object(main.outreach_personal, "answer_question", return_value=ans) as a:
        r1 = client.post("/outreach/personal-line", json={**ids, "job_description": "jd", "variant": 1})
        r2 = client.post("/outreach/answer", json={**ids, "question": "Why you?"})
    assert r1.json()["text"] == line.text and r1.json()["basis"] == "job_description"
    assert g.call_args.kwargs["variant"] == 1
    assert r2.json() == {"answer": "I ship.", "evidence": ["At Y: shipped"], "source": "template", "words": 2}
    assert a.call_args.args[3] == "Why you?" and world.writes == []


def test_a_question_that_is_too_short_is_rejected():
    r = client.post("/outreach/answer", json={"profile_id": "p", "company_id": "c", "recipient_id": "r", "question": "?"})
    assert r.status_code == 422
