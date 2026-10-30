import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.agents import resume_tailor as rt

PROFILE = {
    "summary": "Mobile developer.",
    "skills": ["OOP, Data Structures & Algorithms, Python, SQL, PostgreSQL, Redis, REST APIs, Docker, React Native"],
    "programming_languages": ["Python", "SQL"],
    "experience": [
        "Junior Mobile Developer | Aforro, Bengaluru/nJul 2026 – Present/n"
        "• Built end-to-end Rider app workflows in React Native over REST APIs: pickup, delivery,\nreturns and cancellations.\n"
        "• Integrated rider payments and COD collection, supporting backend APIs.",
        "SDE Intern | TechKareer, Bengaluru\nJan 2026 – Mar 2026\n• Optimized PostgreSQL schemas and queries for production features.",
    ],
    "projects": [
        "StackSearch | GitHub | Python, FAISS, PostgreSQL\nJan 2026 – Present\n• Built a semantic search engine using Python and PostgreSQL.",
        "Built a small unnamed tool with some other things that has a very long description here.",
    ],
}
JD = ("Backend Engineer. Build high-throughput payment services. Requirements: Python, FastAPI, Kafka, PostgreSQL, "
      "Redis caching, designing REST APIs, Docker. Nice to have: Kubernetes.")
DOCS = [Document(page_content="Work Experience: Junior Mobile Developer | Aforro", metadata={"source_section": "experience"})]


class FakeLLM:
    """Stands in for the model: plays back one answer (a dict/str, or an exception to raise)."""

    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def invoke(self, messages, format=None):
        self.calls.append((messages, format))
        if isinstance(self.answer, Exception):
            raise self.answer
        return SimpleNamespace(content=self.answer if isinstance(self.answer, str) else json.dumps(self.answer))


@pytest.fixture(autouse=True)
def model_unavailable():
    """By default the model fails, so tests exercise the rule-based draft and never call Ollama."""
    llm = FakeLLM(RuntimeError("model 'qwen:latest' not found (status code: 404)"))
    with patch.object(rt, "tailor_agent", llm):
        yield llm


def tailor(profile=PROFILE, jd=JD, retriever="default"):
    if retriever == "default":
        retriever = MagicMock()
        retriever.invoke.return_value = DOCS
    with patch.object(rt, "get_retriever", return_value=retriever):
        return rt.tailor_resume("p1", profile, jd)


def draft(**kw):
    return tailor(**kw)["draft"]


def test_the_response_structure_is_preserved():
    out = tailor()
    assert set(out) == {"profile_id", "status", "draft", "evidence_used"}
    assert out["profile_id"] == "p1" and out["status"] == "pending_review"
    assert set(out["draft"]) == set(rt.TailoredResumeDraft.model_fields)
    assert out["evidence_used"] == [{"content": DOCS[0].page_content, "metadata": DOCS[0].metadata}]


def test_the_draft_is_never_empty_for_a_real_resume():
    d = draft()
    assert d["headline"] and d["summary"]
    for field in ("skills_to_highlight", "experience_edits", "projects_to_highlight", "recommendations", "claims_requiring_confirmation"):
        assert d[field], field


def test_highlighted_skills_are_the_candidates_own_and_asked_for_by_the_job():
    skills = draft()["skills_to_highlight"]
    assert skills == ["Python", "PostgreSQL", "Redis", "REST APIs", "Docker"]       # in the job's order
    assert not {"Kafka", "FastAPI", "Kubernetes"} & set(skills)                        # only the job asks for these


def test_what_the_job_wants_but_the_resume_lacks_is_a_claim_to_confirm_never_a_strength():
    d = draft()
    claims = " ".join(d["claims_requiring_confirmation"])
    for missing in ("FastAPI", "Kafka", "Kubernetes"):
        assert missing in claims
    assert "Python" not in claims and "Docker" not in claims                           # the candidate has these
    everything = " ".join([d["headline"], d["summary"], *d["skills_to_highlight"], *d["experience_edits"], *d["projects_to_highlight"]])
    for missing in ("FastAPI", "Kafka", "Kubernetes"):
        assert missing not in everything


def test_headline_and_summary_use_the_real_title_employer_and_skills():
    d = draft()
    assert d["headline"] == "Junior Mobile Developer | Python, PostgreSQL, Redis"
    assert d["summary"].startswith("Junior Mobile Developer at Aforro with experience in Python, PostgreSQL, Redis, REST APIs.")


def test_experience_edits_quote_real_bullets_unchanged():
    d = draft()
    flat = " ".join(" ".join(PROFILE["experience"]).replace("/n", "\n").split())
    for edit in d["experience_edits"]:
        quoted = edit.split("“", 1)[1].split("”", 1)[0]
        assert quoted.rstrip("…").rstrip() in flat                                      # a real bullet, not a paraphrase
    assert d["experience_edits"][0].startswith("Junior Mobile Developer at Aforro: lead with")
    assert "returns and cancellations" not in " ".join(d["experience_edits"])           # the wrapped bullet is not the best match here


def test_wrapped_bullets_are_rejoined_before_matching():
    profile = {"skills": ["Python"], "experience": ["Dev | Acme\n• Built payment services in Python that handle\nhigh volume traffic with retries."]}
    edit = draft(profile=profile, jd="Python payment services.")["experience_edits"][0]
    assert "handle high volume traffic with retries." in edit


def test_only_named_real_projects_are_suggested():
    projects = draft()["projects_to_highlight"]
    assert len(projects) == 1 and projects[0].startswith("StackSearch (Python, FAISS, PostgreSQL): matches the job on")


def test_a_job_title_noun_is_not_counted_as_a_skill_match():
    profile = {"skills": ["Python"], "projects": ["SearchTool | GitHub | Go\n• Built a search engine in Go."]}
    assert draft(profile=profile, jd="Backend Engineer. Python.")["projects_to_highlight"] == []     # "engine" is not "Engineer"


def test_a_resume_with_no_overlap_gets_an_honest_explanation_not_an_empty_draft():
    profile = {"skills": ["Cooking, Baking"], "experience": ["Chef | Bistro, Paris\n• Ran the pastry section."]}
    d = draft(profile=profile, jd="Backend Engineer. Python, Kafka, PostgreSQL.")
    assert d["skills_to_highlight"] == [] and d["projects_to_highlight"] == [] and d["experience_edits"] == []
    assert d["headline"] == "Chef"
    assert any("little overlap" in r for r in d["recommendations"])
    assert "Kafka" in " ".join(d["claims_requiring_confirmation"])


def test_a_profile_without_structured_entries_still_works():
    d = draft(profile={"summary": "Backend engineer with Python experience.", "skills": ["Python, Kafka"]}, jd="Python and Kafka.")
    assert d["skills_to_highlight"] == ["Python", "Kafka"] and d["headline"] == "Python, Kafka"
    assert d["summary"] == "Backend engineer with Python experience."


def test_lists_are_capped_and_free_of_duplicates():
    profile = {"skills": [", ".join(f"Tool{i}" for i in range(30)) + ", Python, python"], "projects": []}
    d = draft(profile=profile, jd="Python " + " ".join(f"Tool{i}" for i in range(30)))
    assert len(d["skills_to_highlight"]) <= rt.MAX_ITEMS
    assert len({s.lower() for s in d["skills_to_highlight"]}) == len(d["skills_to_highlight"])


def test_input_checks_and_retrieval_errors_are_unchanged():
    with pytest.raises(ValueError, match="must not be empty"):
        rt.tailor_resume("p1", PROFILE, "   ")
    with pytest.raises(ValueError, match="not contain enough resume information"):
        rt.tailor_resume("p1", {}, JD)
    with pytest.raises(rt.ResumeTailorError, match="No resume index found"):
        tailor(retriever=None)
    failing = MagicMock(); failing.invoke.side_effect = AssertionError()
    with pytest.raises(rt.ResumeTailorError, match="Could not retrieve resume evidence"):
        tailor(retriever=failing)


# --- the model rewrites the headline and summary; its text is verified, with the rule-based text as fallback ---

GOOD = {"headline": "Mobile Developer | Python, PostgreSQL, Redis",
        "summary": "Mobile developer at Aforro who integrated rider payments and COD collection while working with Python, PostgreSQL and Redis."}


def tailor_with(answer):
    llm = FakeLLM(answer)
    with patch.object(rt, "tailor_agent", llm):
        return tailor(), llm


def test_verified_model_text_replaces_only_the_headline_and_summary():
    baseline = draft()
    out, llm = tailor_with(GOOD)
    d = out["draft"]
    assert d["headline"] == GOOD["headline"] and d["summary"] == GOOD["summary"]
    assert {k: v for k, v in d.items() if k not in ("headline", "summary")} == {k: v for k, v in baseline.items() if k not in ("headline", "summary")}
    assert len(llm.calls) == 1                                                       # one attempt, no retry loop


def test_the_model_is_only_given_verified_facts_not_the_job_descriptions_skills():
    _, llm = tailor_with(GOOD)
    prompt = " ".join(text for _, text in llm.calls[0][0])
    assert "Python, PostgreSQL, Redis" in prompt and "Junior Mobile Developer at Aforro" in prompt
    for jd_only in ("Kafka", "FastAPI", "Kubernetes"):
        assert jd_only not in prompt                                                  # nothing to copy that the candidate lacks
    assert llm.calls[0][1]["required"] == ["headline", "summary"]                    # `{}` is not an acceptable answer


@pytest.mark.parametrize("bad", [
    {**GOOD, "summary": "Mobile developer at Aforro experienced with Kubernetes, Python and Redis for payment services."},   # skill only the job asks for
    {**GOOD, "summary": "Mobile developer at Aforro who cut costs by 70% working with Python, PostgreSQL and Redis."},        # invented number
    {**GOOD, "summary": "Mobile developer at Aforro who is fluent in TensorFlow with Python and PostgreSQL daily."},          # made-up technology
    {**GOOD, "headline": "Senior Kafka Engineer | Python"},                                                                   # claims a JD-only skill
    {**GOOD, "headline": "Backend Engineer | Python, PostgreSQL, Redis, REST APIs, Docker"},                                   # the job's title, never held
    {**GOOD, "headline": "Backend Engineer at Aforro"},                                                                       # same, pinned on a real employer
    {**GOOD, "headline": "Lead Mobile Developer | Python, Redis"},                                                            # adds seniority the candidate lacks
    {**GOOD, "summary": "Integrated rider payments and COD collection, supporting backend APIs. Optimized PostgreSQL schemas and queries for production features."},   # two pasted bullets
    {**GOOD, "summary": "I am a mobile developer at Aforro working with Python, PostgreSQL and Redis every day."},            # first person
    {**GOOD, "summary": "The candidate is a mobile developer at Aforro who works with Python, PostgreSQL and Redis."},         # third person
    {**GOOD, "summary": 'Mobile developer at Aforro with Python, PostgreSQL and Redis"}'},                                    # JSON debris
    {**GOOD, "summary": "Junior Mobile Developer | Aforro, Bengaluru with Python and PostgreSQL experience today."},          # pasted header
    {**GOOD, "summary": "Python Python Python Python Python Python Python Python Python Python Python Python."},               # repetition
    {**GOOD, "summary": "Mobile developer at Aforro."},                                                                       # too short
    {"headline": "x", "summary": "short"},                                                                                    # below the schema minimums
    {},                                                                                                                       # the empty answer that used to give a null draft
    {**GOOD, "extra": 1},                                                                                                     # unknown key
    "not json at all",
    RuntimeError("token repeat limit reached"),                                                                               # model failure
])
def test_unsupported_or_broken_model_output_falls_back_to_the_rule_based_text(bad):
    baseline = draft()
    out, _ = tailor_with(bad)
    assert out["draft"] == baseline
    assert out["draft"]["headline"] and out["draft"]["summary"]                     # never null


def test_a_summary_copied_from_a_resume_bullet_is_rejected():
    copied = {**GOOD, "summary": "Integrated rider payments and COD collection, supporting backend APIs with Python and PostgreSQL."}
    profile = {**PROFILE, "experience": [PROFILE["experience"][0].replace(
        "Integrated rider payments and COD collection, supporting backend APIs.",
        "Integrated rider payments and COD collection, supporting backend APIs with Python and PostgreSQL.")]}
    llm = FakeLLM(copied)
    with patch.object(rt, "tailor_agent", llm):
        d = tailor(profile=profile)["draft"]
    assert d["summary"] != copied["summary"]


def test_the_model_is_not_called_when_there_is_nothing_verified_to_write_from():
    llm = FakeLLM(GOOD)
    with patch.object(rt, "tailor_agent", llm):
        d = draft(profile={"summary": "Chef."}, jd="Backend Engineer. Python.")
    assert llm.calls == [] and d["summary"] == "Chef."


def test_a_headline_may_drop_seniority_or_be_just_the_candidates_matching_skills():
    for headline in ("Mobile Developer | Python, PostgreSQL", "Python, PostgreSQL, Redis", "Junior Mobile Developer at Aforro"):
        out, _ = tailor_with({**GOOD, "headline": headline})
        assert out["draft"]["headline"] == headline, headline


def test_the_job_title_phrase_is_checked_as_a_phrase_not_word_by_word():
    # "backend" and "Engineer" both appear in this resume, but the candidate never held "Backend Engineer".
    profile = {**PROFILE, "experience": PROFILE["experience"] + ["Assistant System Engineer | TCS, Noida\n• Supported backend services."]}
    llm = FakeLLM({**GOOD, "headline": "Backend Engineer | Python"})
    with patch.object(rt, "tailor_agent", llm):
        assert tailor(profile=profile)["draft"]["headline"] != "Backend Engineer | Python"


def test_the_model_step_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("TAILOR_USE_LLM", "0")
    out, llm = tailor_with(GOOD)
    assert llm.calls == [] and out["draft"] == draft()
