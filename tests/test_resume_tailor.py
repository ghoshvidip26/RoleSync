import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.agents import resume_tailor as rt

DOCS = [Document(page_content="Work Experience: Backend Engineer | PayLoop\n• Cut query latency 45% with Redis.",
                 metadata={"source_section": "experience"})]
PROFILE = {"summary": "Backend engineer.", "skills": ["Python, Redis"], "experience": ["Backend Engineer | PayLoop, Pune"]}
JD = "Backend Engineer role using Python and Kafka."
DRAFT = {"headline": "Backend Engineer | Python, Redis", "summary": "Backend engineer who cut query latency 45% with Redis.",
         "skills_to_highlight": ["Python"], "experience_edits": ["Lead with the 45% latency result."],
         "projects_to_highlight": [], "recommendations": ["Mention Redis near the top."], "claims_requiring_confirmation": []}


class FakeLLM:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def invoke(self, messages, format=None):
        self.calls.append((messages, format))
        if isinstance(self.answer, Exception):
            raise self.answer
        return SimpleNamespace(content=self.answer if isinstance(self.answer, str) else json.dumps(self.answer))


def tailor(llm, profile=PROFILE, jd=JD, retriever="default"):
    if retriever == "default":
        retriever = MagicMock()
        retriever.invoke.return_value = DOCS
    with patch.object(rt, "get_retriever", return_value=retriever), patch.object(rt, "tailor_agent", llm):
        return rt.tailor_resume("p1", profile, jd)


def test_the_response_structure_is_preserved():
    out = tailor(FakeLLM(DRAFT))
    assert set(out) == {"profile_id", "status", "draft", "evidence_used"}
    assert out["profile_id"] == "p1" and out["status"] == "pending_review"
    assert set(out["draft"]) == set(rt.TailoredResumeDraft.model_fields)
    assert out["draft"]["headline"] == DRAFT["headline"] and out["draft"]["skills_to_highlight"] == ["Python"]
    assert out["evidence_used"] == [{"content": DOCS[0].page_content, "metadata": DOCS[0].metadata}]


def test_the_model_is_asked_for_the_drafts_schema():
    llm = FakeLLM(DRAFT)
    tailor(llm)
    assert llm.calls[0][1]["title"] == "TailoredResumeDraft"


def test_input_checks():
    with pytest.raises(ValueError, match="must not be empty"):
        rt.tailor_resume("p1", PROFILE, "   ")
    with pytest.raises(ValueError, match="not contain enough resume information"):
        rt.tailor_resume("p1", {}, JD)


def test_a_missing_or_unusable_index_gives_a_clear_error():
    with pytest.raises(rt.ResumeTailorError, match="No resume index found"):
        tailor(FakeLLM(DRAFT), retriever=None)


def test_a_failing_search_is_reported_as_a_retrieval_error():
    retriever = MagicMock(); retriever.invoke.side_effect = AssertionError()
    with pytest.raises(rt.ResumeTailorError, match="Could not retrieve resume evidence"):
        tailor(FakeLLM(DRAFT), retriever=retriever)


def test_invalid_model_output_is_reported_not_returned():
    for bad in ("not json", {"headline": 5}, {"skills_to_highlight": "Python"}):
        with pytest.raises(rt.ResumeTailorError, match="invalid resume draft"):
            tailor(FakeLLM(bad))


def test_a_model_failure_names_the_likely_cause():
    with pytest.raises(rt.ResumeTailorError, match="Ollama is running and the configured model is available"):
        tailor(FakeLLM(RuntimeError("model 'qwen:latest' not found (status code: 404)")))


def test_light_cleanup_trims_dedupes_caps_and_blanks_become_none():
    messy = {**DRAFT, "headline": "   ", "summary": "  A real summary.  ",
             "skills_to_highlight": ["Python", " python ", "", "Kafka"] + [f"skill {i}" for i in range(20)]}
    draft = tailor(FakeLLM(messy))["draft"]
    assert draft["headline"] is None and draft["summary"] == "A real summary."
    assert draft["skills_to_highlight"][:2] == ["Python", "Kafka"] and len(draft["skills_to_highlight"]) == 10
