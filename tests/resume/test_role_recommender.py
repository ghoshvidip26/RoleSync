from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app import main
from app.agents.role_recommender import ROLES, candidate_terms, recommend_roles, score_role

client = TestClient(main.app)

BACKEND = {"role_id": "backend", "title": "Backend Engineer",
           "required_skills": ["Python", "FastAPI", "PostgreSQL", "REST APIs"], "preferred_skills": ["Docker", "Redis"]}
MOBILE = {"role_id": "mobile", "title": "Mobile Developer",
          "required_skills": ["React Native", "TypeScript"], "preferred_skills": ["Firebase"]}


def have(**fields):
    return candidate_terms(fields)


# --- what the candidate has ---------------------------------------------------------------------------

def test_terms_come_from_skills_experience_and_projects():
    terms = have(skills=["Python, SQL"], experience=["Dev | Acme\n• Built a Kafka pipeline."], projects=["Tool | GitHub | Go"])
    assert {"python", "sql", "kafka", "go"} <= terms


@pytest.mark.parametrize("profile", [{}, {"skills": None, "experience": None}, {"skills": "Python"}, {"experience": [None, 5, "Dev"]}])
def test_missing_or_odd_fields_never_crash(profile):
    assert isinstance(candidate_terms(profile), set)


def test_old_uploads_with_a_literal_slash_n_still_match():
    assert "kafka" in have(experience=["Dev | Acme/n• Built Kafka pipelines."])


# --- scoring one role ----------------------------------------------------------------------------------

def test_a_candidate_with_every_skill_is_a_strong_match_with_no_gaps():
    result = score_role(BACKEND, have(skills=["Python, FastAPI, PostgreSQL, REST APIs, Docker, Redis"]))
    assert result["fit"] == "strong_match" and result["score"] == 1.0
    assert result["gaps"] == {"required": [], "preferred": []}
    assert result["matching"]["required"] == ["Python", "FastAPI", "PostgreSQL", "REST APIs"]


def test_matching_and_gaps_are_split_into_required_and_preferred():
    result = score_role(BACKEND, have(skills=["Python, PostgreSQL, Docker"]))
    assert result["matching"] == {"required": ["Python", "PostgreSQL"], "preferred": ["Docker"]}
    assert result["gaps"] == {"required": ["FastAPI", "REST APIs"], "preferred": ["Redis"]}
    assert result["fit"] == "weak_match"                                    # half of the required skills are missing
    assert result["score"] == round((2 * 2 + 1) / (2 * 4 + 2), 2)           # required skills count double


def test_a_candidate_with_nothing_is_a_weak_match_with_everything_missing():
    result = score_role(BACKEND, have(skills=["Cooking"]))
    assert result["fit"] == "weak_match" and result["score"] == 0.0
    assert len(result["gaps"]["required"]) == 4 and len(result["gaps"]["preferred"]) == 2


def test_word_forms_and_punctuation_still_match():
    assert score_role({"title": "x", "required_skills": ["REST APIs", "Node.js"]}, have(skills=["REST API, NodeJS"]))["gaps"]["required"] == []


def test_every_word_of_a_skill_must_be_present():
    result = score_role({"title": "x", "required_skills": ["React Native"]}, have(skills=["React"]))
    assert result["matching"]["required"] == [] and result["gaps"]["required"] == ["React Native"]


def test_evidence_in_work_counts_not_only_the_skills_list():
    assert score_role(BACKEND, have(experience=["Dev | Acme\n• Built FastAPI services on PostgreSQL in Python with REST APIs."]))["fit"] == "strong_match"


def test_a_role_without_skills_does_not_crash():
    assert score_role({"title": "Empty"}, have(skills=["Python"])) == {
        "role_id": None, "title": "Empty", "fit": "weak_match", "score": 0.0,
        "matching": {"required": [], "preferred": []}, "gaps": {"required": [], "preferred": []},
    }


def test_a_skill_made_only_of_filler_words_is_ignored_not_counted_as_a_gap():
    # "Strong knowledge" has no distinctive word to look for, so it can be neither matched nor missed.
    result = score_role({"title": "x", "required_skills": ["Python", "Strong knowledge"]}, have(skills=["Python"]))
    assert result["matching"]["required"] == ["Python"] and result["gaps"]["required"] == []
    assert result["fit"] == "strong_match" and result["score"] == 1.0


def test_a_real_skill_the_resume_lacks_is_a_gap():
    result = score_role({"title": "x", "required_skills": ["Python", "Strong communication"]}, have(skills=["Python"]))
    assert result["gaps"]["required"] == ["Strong communication"]


# --- ranking ------------------------------------------------------------------------------------------------

def test_roles_come_back_best_fit_first():
    ranked = recommend_roles({"skills": ["React Native, TypeScript, Firebase"]}, {"roles": [BACKEND, MOBILE]})
    assert [r["title"] for r in ranked] == ["Mobile Developer", "Backend Engineer"]
    assert ranked[0]["fit"] == "strong_match" and ranked[1]["fit"] == "weak_match"


def test_equal_scores_keep_the_order_of_the_role_list():
    twin = {**MOBILE, "role_id": "twin", "title": "Twin"}
    assert [r["title"] for r in recommend_roles({"skills": ["Cooking"]}, {"roles": [MOBILE, twin]})] == ["Mobile Developer", "Twin"]


def test_the_default_roles_rank_a_react_native_candidate_first_for_react_native():
    ranked = recommend_roles({"skills": ["React Native, TypeScript, JavaScript, REST APIs"]})
    assert ranked[0]["title"] == "React Native Developer" and len(ranked) == len(ROLES["roles"])


def test_the_profile_is_not_modified():
    profile = {"skills": ["Python"], "experience": ["Dev | Acme\n• Built it."]}
    before = repr(profile)
    recommend_roles(profile)
    assert repr(profile) == before


# --- the endpoint ----------------------------------------------------------------------------------------------

def test_the_endpoint_returns_the_ranked_recommendations():
    with patch.object(main.db, "fetch", return_value={"id": "p1", "skills": ["React Native, TypeScript, JavaScript, REST APIs"]}):
        response = client.post("/profiles/p1/recommend-roles")
    body = response.json()
    assert response.status_code == 200 and body["profile_id"] == "p1"
    assert body["recommendations"][0]["title"] == "React Native Developer"
    assert {"role_id", "title", "fit", "score", "matching", "gaps"} <= set(body["recommendations"][0])


def test_the_endpoint_answers_404_for_an_unknown_profile():
    with patch.object(main.db, "fetch", return_value=None):
        assert client.post("/profiles/nope/recommend-roles").status_code == 404
