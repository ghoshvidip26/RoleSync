from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app import main

client = TestClient(main.app)

PROFILE = {"id": "p1", "name": "A", "email": "a@example.com"}
DRAFT = {"headline": "Backend Engineer | Python", "summary": "Built payment services.", "skills_to_highlight": ["Python"],
         "experience_edits": [], "projects_to_highlight": [], "recommendations": [], "claims_requiring_confirmation": []}
TAILORED = {"profile_id": "p1", "status": "pending_review", "draft": DRAFT, "evidence_used": [{"content": "x", "metadata": {}}]}


class FakeDB:
    """In-memory stand-in for the db helpers used by the endpoints."""

    def __init__(self, fail_insert=False):
        self.rows, self.fail_insert, self.inserted = {}, fail_insert, []

    def fetch(self, table, id_):
        return PROFILE if table == "profiles" else self.rows.get(id_)

    def insert(self, table, data):
        if self.fail_insert:
            raise RuntimeError('relation "public.resume_drafts" does not exist')
        row = {"id": f"d{len(self.rows) + 1}", "created_at": f"2026-10-0{len(self.rows) + 1}T10:00:00+00:00", "reviewed_at": None, **data}
        row["status"] = getattr(row["status"], "value", row["status"])
        self.rows[row["id"]] = row
        self.inserted.append(row)
        return row

    def update(self, table, id_, data):
        self.rows[id_].update({k: getattr(v, "value", v) for k, v in data.items()})
        return self.rows[id_]

    def fetch_all(self, table, **filters):
        return [r for r in self.rows.values() if all(r.get(k) == v for k, v in filters.items())]


@pytest.fixture
def fake_db():
    fake = FakeDB()
    with patch.object(main.db, "fetch", fake.fetch), patch.object(main.db, "insert", fake.insert), \
         patch.object(main.db, "update", fake.update), patch.object(main.db, "fetch_all", fake.fetch_all):
        yield fake


def tailor(job_description="Backend Engineer role"):
    with patch.object(main, "tailor_resume", return_value=dict(TAILORED)):
        return client.post("/profiles/p1/tailor-resume", json={"job_description": job_description})


def test_the_tailor_endpoint_keeps_its_response_and_adds_a_draft_id(fake_db):
    response = tailor()
    body = response.json()
    assert response.status_code == 200
    assert {"profile_id", "status", "draft", "evidence_used"} <= set(body)
    assert body["draft_id"] == "d1" and body["status"] == "pending_review"
    saved = fake_db.inserted[0]
    assert saved["profile_id"] == "p1" and saved["job_description"] == "Backend Engineer role" and saved["draft"] == DRAFT
    assert "evidence_used" not in saved                                               # the large evidence is not stored


def test_a_draft_that_cannot_be_saved_fails_instead_of_returning_a_draft_that_cannot_be_reviewed():
    """A missing table gives the clear 503; any other save failure is a 500. Either way the caller is told,
    instead of receiving a draft with no id that can never be approved."""
    missing = FakeDB(fail_insert=True)                                   # raises: relation "public.resume_drafts" does not exist
    with patch.object(main.db, "fetch", missing.fetch), patch.object(main.db, "insert", missing.insert):
        response = tailor()
    assert response.status_code == 503 and "supabase/schema.sql" in response.json()["detail"]

    def broken(table, data):
        raise RuntimeError("connection reset")

    with patch.object(main.db, "fetch", FakeDB().fetch), patch.object(main.db, "insert", broken):
        response = tailor()
    assert response.status_code == 500 and response.json()["detail"] == "Failed to persist resume draft."


def test_a_saved_draft_can_be_read_and_listed_newest_first(fake_db):
    tailor("first job"); tailor("second job")
    got = client.get("/resume-drafts/d1")
    assert got.status_code == 200 and got.json()["draft"] == DRAFT and got.json()["status"] == "pending_review"
    listed = client.get("/profiles/p1/resume-drafts").json()
    assert [d["job_description"] for d in listed] == ["second job", "first job"]
    assert client.get("/resume-drafts/nope").status_code == 404


def test_a_pending_draft_can_be_approved_and_the_time_is_recorded(fake_db):
    tailor()
    response = client.post("/resume-drafts/d1/approve")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "approved" and body["reviewed_at"] is not None


def test_a_pending_draft_can_be_rejected(fake_db):
    tailor()
    assert client.post("/resume-drafts/d1/reject").json()["status"] == "rejected"


@pytest.mark.parametrize("first, second", [("approve", "reject"), ("reject", "approve"), ("approve", "approve")])
def test_a_draft_can_only_be_reviewed_once(fake_db, first, second):
    tailor()
    assert client.post(f"/resume-drafts/d1/{first}").status_code == 200
    again = client.post(f"/resume-drafts/d1/{second}")
    assert again.status_code == 409 and "pending review" in again.json()["detail"]


def test_reviewing_a_missing_draft_is_404(fake_db):
    assert client.post("/resume-drafts/nope/approve").status_code == 404
    assert client.post("/resume-drafts/nope/reject").status_code == 404


def missing_table(*args, **kwargs):
    raise RuntimeError("{'message': \"Could not find the table 'public.resume_drafts' in the schema cache\", 'code': 'PGRST205'}")


@pytest.mark.parametrize("method, url", [
    ("get", "/resume-drafts/d1"), ("get", "/profiles/p1/resume-drafts"),
    ("post", "/resume-drafts/d1/approve"), ("post", "/resume-drafts/d1/reject"),
])
def test_a_missing_resume_drafts_table_gives_a_clear_503_not_a_raw_error(method, url):
    with patch.object(main.db, "fetch", missing_table), patch.object(main.db, "fetch_all", missing_table):
        response = getattr(client, method)(url)
    assert response.status_code == 503 and "supabase/schema.sql" in response.json()["detail"]


def test_other_database_errors_are_not_disguised():
    def other(*args, **kwargs):
        raise RuntimeError("connection reset")

    with patch.object(main.db, "fetch", other):
        with pytest.raises(RuntimeError, match="connection reset"):
            TestClient(main.app, raise_server_exceptions=True).get("/resume-drafts/d1")


# --- editing a draft before approving it ---------------------------------------------------------

def edit(body, draft_id="d1"):
    return client.patch(f"/resume-drafts/{draft_id}", json=body)


def test_editing_the_headline_changes_only_the_headline(fake_db):
    tailor()
    response = edit({"headline": "Platform Engineer | Python"})
    assert response.status_code == 200
    body = response.json()
    assert body["draft"]["headline"] == "Platform Engineer | Python"
    assert {k: v for k, v in body["draft"].items() if k != "headline"} == {k: v for k, v in DRAFT.items() if k != "headline"}
    assert body["status"] == "pending_review" and body["reviewed_at"] is None            # still waiting for review


def test_both_fields_can_be_edited_and_text_is_trimmed(fake_db):
    tailor()
    draft = edit({"headline": "  New headline  ", "summary": "  A better summary.  "}).json()["draft"]
    assert draft["headline"] == "New headline" and draft["summary"] == "A better summary."
    assert draft["skills_to_highlight"] == DRAFT["skills_to_highlight"]                  # lists untouched


@pytest.mark.parametrize("blank", [None, "", "   "])
def test_sending_null_or_blank_clears_the_field_but_not_the_other(fake_db, blank):
    tailor()
    draft = edit({"summary": blank}).json()["draft"]
    assert draft["summary"] is None and draft["headline"] == DRAFT["headline"]


def test_an_edit_with_no_fields_is_rejected(fake_db):
    tailor()
    assert edit({}).status_code == 422
    assert fake_db.rows["d1"]["draft"] == DRAFT


@pytest.mark.parametrize("body", [{"headline": "x" * 121}, {"summary": "y" * 601}, {"headline": 5}])
def test_over_long_or_wrongly_typed_values_are_rejected_and_nothing_is_saved(fake_db, body):
    tailor()
    assert edit(body).status_code == 422
    assert fake_db.rows["d1"]["draft"] == DRAFT


@pytest.mark.parametrize("review", ["approve", "reject"])
def test_a_reviewed_draft_can_no_longer_be_edited(fake_db, review):
    tailor()
    client.post(f"/resume-drafts/d1/{review}")
    response = edit({"headline": "Too late"})
    assert response.status_code == 409 and "pending review" in response.json()["detail"]
    assert fake_db.rows["d1"]["draft"]["headline"] == DRAFT["headline"]


def test_editing_a_missing_draft_is_404(fake_db):
    assert edit({"headline": "x"}, draft_id="nope").status_code == 404


def test_the_edited_text_is_what_gets_approved(fake_db):
    tailor()
    edit({"summary": "My own wording of the summary."})
    approved = client.post("/resume-drafts/d1/approve").json()
    assert approved["status"] == "approved" and approved["draft"]["summary"] == "My own wording of the summary."


def test_editing_with_a_missing_table_gives_the_clear_503():
    with patch.object(main.db, "fetch", missing_table):
        response = edit({"headline": "x"})
    assert response.status_code == 503 and "supabase/schema.sql" in response.json()["detail"]


def test_a_summary_of_exactly_the_limit_is_accepted(fake_db):
    tailor()
    assert edit({"summary": "y" * 600}).status_code == 200


# --- editing the rest of the draft ---------------------------------------------------------------------------

ROLE = {"company": "PayLoop", "job_title": "Backend Engineer", "duration": "2020 – 2021", "location": "Pune", "bullets": ["Cut latency 45%."]}
PROJECT = {"name": "StreamLite", "technologies": ["Go"], "description": None, "bullets": ["Built an engine."]}


def test_skills_achievements_and_education_can_be_replaced_as_lists(fake_db):
    tailor()
    body = edit({"skills": ["Python", "Go"], "achievements": ["Won a hackathon."], "education": [{"details": "B.Tech"}]}).json()["draft"]
    assert body["skills"] == ["Python", "Go"] and body["achievements"] == ["Won a hackathon."] and body["education"] == [{"details": "B.Tech"}]
    assert body["headline"] == DRAFT["headline"]                         # fields that were not sent are untouched


def test_experience_and_projects_can_be_replaced_with_structured_entries(fake_db):
    tailor()
    draft = edit({"experience": [ROLE], "projects": [PROJECT]}).json()["draft"]
    assert draft["experience"] == [ROLE] and draft["projects"] == [PROJECT]


@pytest.mark.parametrize("field", ["skills", "achievements", "education", "experience", "projects"])
def test_sending_null_for_a_list_clears_it(fake_db, field):
    tailor()
    edit({"skills": ["Python"], "experience": [ROLE]})
    assert edit({field: None}).json()["draft"][field] == []


@pytest.mark.parametrize("body", [
    {"skills": "Python"},                                   # must be a list
    {"experience": [{"company": "PayLoop"}]},               # job_title is required
    {"projects": [{"technologies": ["Go"]}]},               # name is required
    {"achievements": [1, 2]},
])
def test_malformed_list_edits_are_rejected_and_nothing_is_saved(fake_db, body):
    tailor()
    assert edit(body).status_code == 422
    assert fake_db.rows["d1"]["draft"] == DRAFT


def test_the_empty_edit_message_lists_every_editable_field(fake_db):
    tailor()
    detail = edit({}).json()["detail"]
    assert all(field in detail for field in ("headline", "summary", "skills", "experience", "projects", "achievements", "education"))
