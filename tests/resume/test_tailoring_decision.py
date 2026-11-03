from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app import main
from app.agents.tailoring_decision import detect_repeated_action_verbs, find_buried_work, needs_tailoring

client = TestClient(main.app)


def role(*bullets, header="Mobile Developer | Aforro, Bengaluru"):
    return header + "\nJan 2024 – Present\n" + "\n".join(f"• {b}" for b in bullets)


def project(name, *bullets, tech="Python"):
    return f"{name} | GitHub | {tech}\nJan 2025 – Present\n" + "\n".join(f"• {b}" for b in bullets)


BACKEND_JOB = "Backend Engineer. Python, FastAPI, PostgreSQL, REST APIs, Docker."
RN_JOB = "React Native Developer. React Native, push notifications, maps, TypeScript."

RN_ROLE = role("Built React Native screens for the rider app.", "Added push notifications and maps for riders.")
BACKEND_PROJECT = project("OrderService", "Created a REST API with FastAPI and PostgreSQL.", "Shipped it with Docker.", tech="Python, FastAPI, PostgreSQL")


# --- check 1: repeated opening words -----------------------------------------------------------

def test_a_word_opening_three_bullets_is_flagged_but_two_is_fine():
    assert detect_repeated_action_verbs(["Built a", "Built b", "Built c"]) == {"built": 3}
    assert detect_repeated_action_verbs(["Built a", "Built b", "Led c"]) == {}


def test_case_and_bullet_markers_do_not_matter():
    assert detect_repeated_action_verbs(["Built a", "built b", "  BUILT c", "• Built d"]) == {"built": 4}


def test_a_bullet_starting_with_a_number_is_not_counted_as_a_word():
    assert detect_repeated_action_verbs(["50% faster a", "50% faster b", "50% faster c"]) == {}


def test_it_counts_across_the_whole_resume_not_per_role():
    profile = {"experience": [role("Built a.", "Built b."), role("Built c.", header="Intern | Acme")],
               "projects": [project("P", "Led d.")]}
    result = needs_tailoring(profile, "")
    assert result["needed"] and "'built' starts 3 bullets" in result["reasons"][0]


def test_a_wrapped_bullet_counts_once():
    profile = {"experience": ["Dev | Acme\n• Built a payment service for the\nfinance team.\n• Built b.\n• Led c."]}
    assert needs_tailoring(profile, "")["reasons"] == []


def test_lines_that_are_not_bullets_are_ignored():
    profile = {"projects": ["Tool: Built for fun.\nBuilt this.\nBuilt that.\nBuilt more."]}
    assert needs_tailoring(profile, "")["needed"] is False


# --- check 2: relevant work buried behind the latest role ---------------------------------------

def test_react_native_resume_with_backend_projects_for_a_backend_job_is_flagged():
    profile = {"experience": [RN_ROLE], "projects": [BACKEND_PROJECT]}
    result = needs_tailoring(profile, BACKEND_JOB)
    assert result["needed"] is True
    assert "OrderService" in result["reasons"][0] and "Lead with that work" in result["reasons"][0]


def test_work_buried_in_an_earlier_role_is_flagged_too():
    earlier = role("Created a REST API with FastAPI and PostgreSQL.", header="Backend Developer | OldCo, Pune")
    profile = {"experience": [RN_ROLE, earlier]}
    assert "Backend Developer at OldCo" in needs_tailoring(profile, BACKEND_JOB)["reasons"][0]


def test_a_resume_that_already_leads_with_the_right_work_is_not_flagged():
    assert needs_tailoring({"experience": [RN_ROLE]}, RN_JOB) == {"needed": False, "reasons": []}


def test_a_resume_with_no_matching_work_anywhere_is_not_a_tailoring_problem():
    assert needs_tailoring({"experience": [RN_ROLE], "projects": [project("Notes", "Wrote notes.", tech="Go")]}, BACKEND_JOB)["needed"] is False


def test_a_job_title_noun_alone_does_not_count_as_a_match():
    profile = {"experience": [RN_ROLE], "projects": [project("SearchTool", "Built a search engine.", "Wrote an engine for notes.", tech="Go")]}
    assert find_buried_work(profile, "Backend Engineer. Engineer a backend.") is None


def test_a_job_description_with_almost_no_keywords_never_triggers_it():
    assert find_buried_work({"experience": [RN_ROLE], "projects": [BACKEND_PROJECT]}, "Python.") is None


def test_both_checks_can_fire_together():
    profile = {"experience": [role("Built a.", "Built b.", "Built c.")], "projects": [BACKEND_PROJECT]}
    assert len(needs_tailoring(profile, BACKEND_JOB)["reasons"]) == 2


# --- robustness -------------------------------------------------------------------------------------

@pytest.mark.parametrize("profile", [{}, {"experience": None, "projects": None}, {"experience": [], "projects": []},
                                     {"experience": "just a string"}, {"experience": [None, 5, ""]}])
def test_an_empty_or_odd_profile_is_simply_not_needed(profile):
    assert needs_tailoring(profile, BACKEND_JOB) == {"needed": False, "reasons": []}


def test_old_uploads_with_a_literal_slash_n_are_handled():
    legacy = "Mobile Developer | Aforro/nJan 2024 – Present/n• Built a./n• Built b./n• Built c."
    assert needs_tailoring({"experience": [legacy]}, "")["needed"] is True


def test_the_input_is_not_modified():
    profile = {"experience": [RN_ROLE], "projects": [BACKEND_PROJECT]}
    before = repr(profile)
    needs_tailoring(profile, BACKEND_JOB)
    assert repr(profile) == before


# --- the review endpoint ------------------------------------------------------------------------------

def review_with(profile, orchestrate_result=None, **patches):
    result = orchestrate_result or {"verdict": {"fit_assessment": {"verdict": "partial_match"}}, "evidence_used": []}
    with patch.object(main.db, "fetch", return_value=profile), patch.object(main, "orchestrate", return_value=result), \
         patch.object(main, "needs_tailoring", wraps=main.needs_tailoring):
        return client.post("/profiles/p1/review", json={"job_description": BACKEND_JOB})


def test_the_review_response_keeps_its_keys_and_gains_a_tailoring_field():
    response = review_with({"id": "p1", "experience": [RN_ROLE], "projects": [BACKEND_PROJECT]})
    body = response.json()
    assert response.status_code == 200
    assert set(body) == {"verdict", "evidence_used", "tailoring"}
    assert body["verdict"]["fit_assessment"]["verdict"] == "partial_match"                    # unchanged
    assert body["tailoring"]["needed"] is True and body["tailoring"]["reasons"]


def test_review_errors_are_unchanged():
    from app.agents.review import ReviewError

    with patch.object(main.db, "fetch", return_value={"id": "p1"}), patch.object(main, "orchestrate", side_effect=ReviewError("model down")):
        assert client.post("/profiles/p1/review", json={"job_description": "x"}).status_code == 502
    with patch.object(main.db, "fetch", return_value={"id": "p1"}), patch.object(main, "orchestrate", side_effect=ValueError("no index")):
        assert client.post("/profiles/p1/review", json={"job_description": "x"}).status_code == 409
    with patch.object(main.db, "fetch", return_value=None):
        assert client.post("/profiles/p1/review", json={"job_description": "x"}).status_code == 404
