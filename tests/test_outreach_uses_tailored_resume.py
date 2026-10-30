from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app import main
from app.models.company import Company
from app.models.profile import Profile
from app.models.recipient import Recipient
from app.services import outreach_agent as oa
from app.services.outreach_agent import DraftEmail, OutreachAgent, _Draft

client = TestClient(main.app)

PROFILE = Profile(
    id="p1", name="Sarah Connor", email="sarah@example.com",
    skills=["Python, Kubernetes, PostgreSQL, Redis, Docker"],
    experience=["Backend Engineer | PayLoop, Pune\nJun 2020 – Dec 2021\n"
                "• Cut query latency 45% with a Redis caching layer in front of PostgreSQL.\n"
                "• Designed REST APIs for a payments reconciliation service used by the finance team."],
    projects=["StreamLite | GitHub | Go, Python, Kafka\n• Built a real-time event processing engine with a plugin API."],
)
COMPANY = Company(id="c1", name="Cyberdyne Systems", website="https://cyberdyne.ai")
RECIPIENT = Recipient(id="r1", name="Miles Dyson", email="miles@cyberdyne.ai", role="CTO")

TAILORED = {
    "headline": "Backend Engineer | Python, Redis",
    "summary": "Backend engineer at PayLoop with Python and Redis.",
    "skills_to_highlight": ["Redis", "PostgreSQL", "Docker"],
    "experience_edits": ["Backend Engineer at PayLoop: lead with “Cut query latency 45% with a Redis caching layer in front of PostgreSQL.” (matches Redis, PostgreSQL)"],
    "projects_to_highlight": ["StreamLite (Go, Python, Kafka): matches the job on Python"],
    "recommendations": ["Put StreamLite first."],
    "claims_requiring_confirmation": ["Terraform: the job asks for it but your resume doesn't show it. Add it only if you have really used it."],
}


# --- what the agent takes from an approved tailored resume ----------------------------------------

def test_the_facts_use_highlights_but_never_the_claims_or_recommendations():
    facts = oa._tailored_facts(TAILORED)
    assert facts["skills"] == ["Redis", "PostgreSQL", "Docker"] and facts["projects"] == ["StreamLite"]
    assert facts["bullet"] == "Cut query latency 45% with a Redis caching layer in front of PostgreSQL."
    assert "Terraform" not in facts["text"] and "Put StreamLite first" not in facts["text"]
    assert oa._tailored_facts(None)["lines"] == [] and oa._tailored_facts({})["text"] == ""


def run_agent(tailored, body=None):
    llm = MagicMock()
    structured = llm.with_structured_output.return_value
    structured.invoke.return_value = _Draft(subject="Latency work and Cyberdyne Systems", body=body or "")
    with patch.object(oa, "get_retriever", return_value=None), patch.object(oa, "get_llm", return_value=llm):
        result = OutreachAgent().run(PROFILE, COMPANY, RECIPIENT, tailored=tailored)
    return result, structured.invoke.call_args[0][0]


def test_the_prompt_leads_with_the_approved_highlights_and_leaves_out_the_claims():
    _, messages = run_agent(TAILORED)
    prompt = " ".join(text for _, text in messages)
    assert "Approved tailored resume highlights" in prompt
    assert "Skills to emphasise: Redis, PostgreSQL, Docker" in prompt and "Projects to mention: StreamLite" in prompt
    assert "A result to mention: Cut query latency 45%" in prompt
    assert "Terraform" not in prompt                                              # a claim to confirm is not a strength


def test_without_a_tailored_resume_the_prompt_is_unchanged():
    _, messages = run_agent(None)
    assert "Approved tailored resume highlights" not in " ".join(text for _, text in messages)


def test_the_fallback_template_leads_with_the_tailored_skills_and_result():
    draft = oa._template_email(PROFILE, COMPANY, RECIPIENT, TAILORED)
    assert "I work mainly with Redis, PostgreSQL, Docker." in draft.body
    assert "For example: Cut query latency 45% with a Redis caching layer in front of PostgreSQL" in draft.body
    plain = oa._template_email(PROFILE, COMPANY, RECIPIENT)
    assert "I work mainly with Python, Kubernetes, PostgreSQL." in plain.body          # no draft: as before


def test_an_email_still_cannot_claim_something_the_tailored_resume_only_listed_as_a_claim_to_confirm():
    body = ("Hi Miles,\n\nAt PayLoop I cut query latency 45% with a Redis caching layer, and I use Terraform daily, "
            "so I'd love to hear how Cyberdyne Systems thinks about performance. Would you be open to a short conversation?"
            "\n\nBest regards,\nSarah Connor")
    result, _ = run_agent(TAILORED, body)
    assert result.subject == "Sarah Connor: interested in opportunities at Cyberdyne Systems"   # Terraform is not in the resume


def test_an_email_that_uses_the_tailored_emphasis_is_accepted():
    body = ("Hi Miles,\n\nAt PayLoop I cut query latency 45% with a Redis caching layer in front of PostgreSQL, and I'd love "
            "to hear how Cyberdyne Systems thinks about performance. Would you be open to a short conversation?"
            "\n\nBest regards,\nSarah Connor")
    result, _ = run_agent(TAILORED, body)
    assert "cut query latency 45%" in result.body and "interested in opportunities" not in result.subject


# --- the endpoint ----------------------------------------------------------------------------------

def outreach(draft_row, agent_result=DraftEmail(subject="s", body="b")):
    rows = {"profiles": PROFILE.model_dump(), "companies": COMPANY.model_dump(), "recipients": RECIPIENT.model_dump(), "resume_drafts": draft_row}
    inserted = []

    def insert(table, data):
        inserted.append(data)
        return {"id": "e1"}

    def update(table, id_, data):
        return {"id": id_, "profile_id": "p1", "company_id": "c1", "recipient_id": "r1", **data}

    body = {"profile_id": "p1", "company_id": "c1", "recipient_id": "r1"}
    if draft_row is not None or True:
        body["resume_draft_id"] = "d1"
    with patch.object(main.db, "fetch", side_effect=lambda table, id_: rows[table]), patch.object(main.db, "insert", side_effect=insert), \
         patch.object(main.db, "update", side_effect=update), patch.object(main.agent, "run", return_value=agent_result) as run:
        response = client.post("/outreach", json=body)
    return response, inserted, run


def draft_row(status="approved", profile_id="p1"):
    return {"id": "d1", "profile_id": profile_id, "job_description": "Backend Engineer", "status": status, "draft": TAILORED}


def test_an_approved_draft_is_passed_to_the_agent_and_recorded_on_the_email():
    response, inserted, run = outreach(draft_row("approved"))
    assert response.status_code == 201
    assert run.call_args.kwargs["tailored"] == TAILORED
    assert inserted[0]["resume_draft_id"] == "d1"


@pytest.mark.parametrize("status", ["pending_review", "rejected"])
def test_a_draft_that_is_not_approved_is_refused_before_anything_is_created(status):
    response, inserted, run = outreach(draft_row(status))
    assert response.status_code == 409 and "approved" in response.json()["detail"]
    assert inserted == [] and run.call_count == 0


def test_a_draft_from_another_profile_is_refused():
    response, inserted, _ = outreach(draft_row(profile_id="someone-else"))
    assert response.status_code == 409 and "different profile" in response.json()["detail"]
    assert inserted == []


def test_a_missing_draft_is_404():
    with patch.object(main.db, "fetch", side_effect=lambda table, id_: None if table == "resume_drafts" else PROFILE.model_dump() | {"id": id_}):
        response = client.post("/outreach", json={"profile_id": "p1", "company_id": "c1", "recipient_id": "r1", "resume_draft_id": "nope"})
    assert response.status_code == 404


def test_without_a_draft_id_outreach_behaves_exactly_as_before():
    rows = {"profiles": PROFILE.model_dump(), "companies": COMPANY.model_dump(), "recipients": RECIPIENT.model_dump()}
    inserted = []
    with patch.object(main.db, "fetch", side_effect=lambda table, id_: rows[table]), \
         patch.object(main.db, "insert", side_effect=lambda t, d: inserted.append(d) or {"id": "e1"}), \
         patch.object(main.db, "update", side_effect=lambda t, i, d: {"id": i, "profile_id": "p1", "company_id": "c1", "recipient_id": "r1", **d}), \
         patch.object(main.agent, "run", return_value=DraftEmail(subject="s", body="b")) as run:
        response = client.post("/outreach", json={"profile_id": "p1", "company_id": "c1", "recipient_id": "r1"})
    assert response.status_code == 201
    assert "resume_draft_id" not in inserted[0]                       # works before the emails.resume_draft_id column exists
    assert run.call_args.kwargs["tailored"] is None
