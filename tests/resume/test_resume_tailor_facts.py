import inspect
import re
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from app.agents import resume_tailor as rt

PROFILE = {
    "skills": ["Python, Redis, PostgreSQL"],
    "experience": ["Backend Engineer | PayLoop, Pune\nJan 2022 – Present\n• Cut query latency 45% with a Redis caching layer in Python services."],
    "projects": [],
}
JD = "Backend Engineer. Python, Redis, PostgreSQL."


def facts_keys_read_by_the_model_text_code() -> set[str]:
    source = inspect.getsource(rt._verify_prose) + inspect.getsource(rt._llm_prose)
    return {a or b for a, b in re.findall(r'facts\["(\w+)"\]|facts\[\'(\w+)\'\]', source)}


def test_build_draft_supplies_every_fact_the_model_text_code_reads():
    """Regression: a rewrite of _build_draft returned only 4 facts, and /tailor-resume failed with KeyError: 'anchor_terms'."""
    _, facts = rt._build_draft(PROFILE, JD, "")
    assert facts_keys_read_by_the_model_text_code() <= set(facts)


def test_tailor_resume_runs_end_to_end_when_the_model_is_unavailable():
    retriever = MagicMock()
    retriever.invoke.return_value = [Document(page_content="Backend Engineer | PayLoop", metadata={})]
    with patch.object(rt, "get_retriever", return_value=retriever), \
         patch.object(rt, "tailor_agent", MagicMock(invoke=MagicMock(side_effect=RuntimeError("model not found")))):
        result = rt.tailor_resume("p1", PROFILE, JD)
    assert result["status"] == "pending_review" and result["profile_id"] == "p1"
    assert result["draft"]["headline"] and result["draft"]["summary"]


def test_a_verified_model_text_is_used_through_the_same_facts():
    retriever = MagicMock()
    retriever.invoke.return_value = [Document(page_content="Backend Engineer | PayLoop", metadata={})]
    good = ('{"headline": "Backend Engineer | Python, Redis", "summary": "Backend engineer at PayLoop who cut query latency 45% '
            'using Redis and Python."}')
    agent = MagicMock(invoke=MagicMock(return_value=MagicMock(content=good)))
    with patch.object(rt, "get_retriever", return_value=retriever), patch.object(rt, "tailor_agent", agent):
        draft = rt.tailor_resume("p1", PROFILE, JD)["draft"]
    assert draft["headline"] == "Backend Engineer | Python, Redis" and "cut query latency 45%" in draft["summary"]


FULL_PROFILE = {
    "summary": "Mobile developer.",
    "skills": ["Python, REST APIs, Docker"],
    "experience": ["Mobile Developer | Aforro, Bengaluru\nJan 2024 – Present\n• Built React Native screens over REST APIs.\n• Integrated payments APIs."],
    "projects": ["OrderService | GitHub | Python, FastAPI\n• Created a REST API with FastAPI."],
    "achievements": ["Won a hackathon in 2024."],
    "education": ["B.Tech in Computer Engineering | Jamia Millia Islamia, New Delhi\nDec 2020 – Jun 2024"],
}


def test_a_complete_profile_with_education_and_achievements_builds_a_valid_draft():
    """Regression: education was passed as plain strings while the draft model (and the edit endpoint) use
    {"details": ...} dictionaries, so tailoring failed with a validation error for any profile with education."""
    draft, _ = rt._build_draft(FULL_PROFILE, "Backend developer. APIs, services.", "")
    assert draft.education == [{"details": FULL_PROFILE["education"][0]}]
    assert draft.achievements == ["Won a hackathon in 2024."]
    assert [e.job_title for e in draft.experience] == ["Mobile Developer"] and [p.name for p in draft.projects] == ["OrderService"]


def test_tailor_resume_returns_the_complete_draft_for_a_complete_profile():
    retriever = MagicMock()
    retriever.invoke.return_value = [Document(page_content="Mobile Developer | Aforro", metadata={})]
    with patch.object(rt, "get_retriever", return_value=retriever), \
         patch.object(rt, "tailor_agent", MagicMock(invoke=MagicMock(side_effect=RuntimeError("model not found")))):
        draft = rt.tailor_resume("p1", FULL_PROFILE, "Backend developer. APIs, services.")["draft"]
    assert set(draft) == {"headline", "summary", "skills", "experience", "projects", "achievements", "education",
                          "recommendations", "claims_requiring_confirmation"}
    assert draft["education"] == [{"details": FULL_PROFILE["education"][0]}]
