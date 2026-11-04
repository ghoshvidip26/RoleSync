"""POST /resume-drafts/{id}/suggest and the optional job title on tailor-resume."""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app import main
from test_positioning import JD_RN, PROFILE

client = TestClient(main.app)
PROFILE_ROW = {"id": "p1", "name": "Vidip", "email": "v@example.com", **PROFILE}


class Rows:
    def __init__(self, status="pending_review", jd=JD_RN):
        self.draft = {"id": "d1", "profile_id": "p1", "job_description": jd, "draft": {"headline": "H", "summary": "S", "skills": []},
                      "status": status, "created_at": "2026-11-01T10:00:00+00:00", "reviewed_at": None}
        self.writes = []

    def fetch(self, table, id_):
        if table == "profiles":
            return PROFILE_ROW if id_ == "p1" else None
        return self.draft if id_ == "d1" else None

    def update(self, *a, **k):
        self.writes.append(a)

    def insert(self, *a, **k):
        self.writes.append(a)


def suggest(body, status="pending_review", draft_id="d1", jd=JD_RN):
    rows = Rows(status, jd)
    with patch.object(main.db, "fetch", rows.fetch), patch.object(main.db, "update", rows.update), patch.object(main.db, "insert", rows.insert):
        return client.post(f"/resume-drafts/{draft_id}/suggest", json=body), rows


def test_each_field_gets_a_suggestion_and_nothing_is_saved():
    head, rows = suggest({"field": "headline"})
    assert head.status_code == 200 and head.json()["field"] == "headline"
    assert head.json()["value"].startswith("React Native Developer | ")
    summary, _ = suggest({"field": "summary"})
    assert summary.json()["value"].startswith("Software engineer with 1 year of experience")
    skills, _ = suggest({"field": "skills"})
    assert skills.json()["value"][0] == "React Native" and isinstance(skills.json()["value"], list)
    assert rows.writes == []                                                     # a suggestion never changes the stored draft


def test_a_different_variant_gives_a_different_suggestion():
    values = [suggest({"field": "summary", "variant": v})[0].json()["value"] for v in range(3)]
    assert len(set(values)) == 3
    assert suggest({"field": "summary", "variant": 3})[0].json()["value"] == values[0]
    assert suggest({"field": "summary", "variant": 1})[0].json()["variant"] == 1


def test_the_users_job_title_shapes_the_headline_and_is_still_checked():
    ok, _ = suggest({"field": "headline", "job_title": "Senior Python Engineer"})
    assert ok.json()["value"].startswith("Python Engineer | ")
    unsupported, _ = suggest({"field": "headline", "job_title": "Kafka Streaming Engineer"})
    assert unsupported.json()["value"].startswith("Junior Mobile Application Developer | ")
    blank, _ = suggest({"field": "headline", "job_title": "   "})
    assert blank.json()["value"].startswith("React Native Developer | ")


@pytest.mark.parametrize("body", [{"field": "education"}, {}, {"field": "headline", "variant": -1}, {"field": "headline", "variant": "x"},
                                  {"field": "headline", "job_title": "x" * 101}])
def test_invalid_requests_are_422(body):
    assert suggest(body)[0].status_code == 422


@pytest.mark.parametrize("status", ["approved", "rejected"])
def test_only_drafts_pending_review_get_suggestions(status):
    r, _ = suggest({"field": "headline"}, status=status)
    assert r.status_code == 409 and "pending review" in r.json()["detail"]


def test_an_unknown_draft_is_404():
    assert suggest({"field": "headline"}, draft_id="nope")[0].status_code == 404


def test_a_draft_with_an_empty_job_description_is_a_clear_422():
    r, _ = suggest({"field": "headline"}, jd="   ")
    assert r.status_code == 422 and "must not be empty" in r.json()["detail"]


# ------------------------------------------------------------------ tailor-resume: optional job title

def tailor(body):
    rows = Rows()
    with patch.object(main.db, "fetch", rows.fetch), patch.object(main.db, "insert", lambda t, d: {"id": "d9", **d}), \
         patch.object(main, "tailor_resume", return_value={"profile_id": "p1", "status": "pending_review", "draft": {"headline": "h"}, "evidence_used": []}) as t:
        response = client.post("/profiles/p1/tailor-resume", json=body)
    return response, t


def test_the_job_title_is_passed_on_only_when_given():
    r, t = tailor({"job_description": JD_RN, "job_title": "  React Native Developer "})
    assert r.status_code == 200 and t.call_args.kwargs["job_title"] == "React Native Developer"
    r, t = tailor({"job_description": JD_RN})
    assert r.status_code == 200 and "job_title" not in t.call_args.kwargs                # the call is unchanged without it
    r, t = tailor({"job_description": JD_RN, "job_title": "   "})
    assert "job_title" not in t.call_args.kwargs
    assert tailor({"job_description": JD_RN, "job_title": "x" * 101})[0].status_code == 422
