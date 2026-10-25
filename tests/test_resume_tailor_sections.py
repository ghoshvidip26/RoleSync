import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.agents import resume_tailor as rt


class FakeLLM:
    """Plays back canned answers (a string, or an exception to raise) and records every call."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def invoke(self, messages, format):
        self.calls.append((format["title"], messages))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return SimpleNamespace(content=answer if isinstance(answer, str) else json.dumps(answer))


MESSAGES = [("system", "rules"), ("human", "job + resume")]


def generate(llm, schema=rt.HeadlineDraft, section="headline", validate=None):
    with patch.object(rt, "tailor_agent", llm):
        return rt._generate_section(section, schema, MESSAGES, validate)


def test_a_valid_section_is_returned_and_the_model_gets_only_that_sections_schema():
    llm = FakeLLM({"headline": "Backend Engineer, Payments"})
    assert generate(llm).headline == "Backend Engineer, Payments"
    assert [title for title, _ in llm.calls] == ["HeadlineDraft"]


def test_only_the_failing_section_is_retried_with_the_reason_fed_back():
    llm = FakeLLM("not json at all", {"headline": "Backend Engineer, Payments"})
    assert generate(llm).headline == "Backend Engineer, Payments"
    assert len(llm.calls) == 2
    retry_messages = llm.calls[1][1]
    assert retry_messages[:2] == MESSAGES                          # the original task is kept
    assert "headline was rejected" in retry_messages[-1][1]        # ...with the reason for the retry


def test_a_failed_model_call_is_retried():
    llm = FakeLLM(RuntimeError("prediction aborted, token repeat limit reached"), {"headline": "Backend Engineer"})
    assert generate(llm).headline == "Backend Engineer"
    assert len(llm.calls) == 2


def test_a_section_that_keeps_failing_raises_a_clear_error_naming_the_section():
    llm = FakeLLM(RuntimeError("token repeat limit reached"), RuntimeError("token repeat limit reached"))
    with pytest.raises(rt.ResumeTailorError, match=r"Could not generate the headline after 2 attempts: model call failed: token repeat limit"):
        generate(llm)


@pytest.mark.parametrize("bad, reason", [
    ({"headline": "!!! ???"}, "no meaningful text"),
    ({"headline": "Developer"}, "Headline is incomplete"),
    ({"headline": "10000000000000000000000000000000000000000000"}, "Headline contains malformed tokens"),
    ({"headline": "Engineer 100000000000000000000000000000000000000"}, "Headline contains malformed tokens"),
    ({"headline": "x " * 61}, "headline: String should have at most 120 characters"),    # over the schema's cap
    ({"headline": "Backend Engineer", "extra": 1}, "extra: Extra inputs are not permitted"),
    ('{"headline": "Backend Engi', "not valid JSON"),                                    # truncated JSON
])
def test_malformed_or_meaningless_output_is_rejected_not_accepted(bad, reason):
    llm = FakeLLM(bad, bad)
    with pytest.raises(rt.ResumeTailorError) as err:
        generate(llm)
    assert reason.lower() in str(err.value).lower()


@pytest.mark.parametrize("repeated", [
    "and and and and and and and and and",
    "Python Python Python Python Python Python Python Python",
    "scalable systems scalable systems scalable systems scalable systems",
])
def test_repetitive_text_is_rejected(repeated):
    llm = FakeLLM({"summary": repeated}, {"summary": repeated})
    with pytest.raises(rt.ResumeTailorError, match="excessive repetition"):
        generate(llm, schema=rt.SummaryDraft, section="summary")


def test_normal_text_is_not_mistaken_for_repetition():
    text = "Backend engineer who built a Kafka pipeline and reduced latency by 45% with Redis caching."
    assert generate(FakeLLM({"summary": text}), schema=rt.SummaryDraft, section="summary").summary == text


def test_list_sections_reject_empty_entries_and_oversized_lists():
    too_many = {"skills_to_highlight": [f"skill {i}" for i in range(11)]}
    empty_entry = {"skills_to_highlight": ["Python", "  "]}
    for bad in (too_many, empty_entry):
        with pytest.raises(rt.ResumeTailorError):
            generate(FakeLLM(bad, bad), schema=rt.SkillsDraft, section="skills")
    assert generate(FakeLLM({"skills_to_highlight": ["Python", "Kafka"]}), schema=rt.SkillsDraft, section="skills").skills_to_highlight == ["Python", "Kafka"]


def test_a_section_specific_check_can_reject_and_trigger_a_retry():
    def must_mention_python(out):
        if "python" not in out.headline.lower():
            raise rt.OutputQualityError("headline does not mention a real skill")

    llm = FakeLLM({"headline": "Great Engineer Wanted"}, {"headline": "Python Backend Engineer"})
    assert generate(llm, validate=must_mention_python).headline == "Python Backend Engineer"
    assert len(llm.calls) == 2


# --- the existing whole-draft check keeps its behaviour --------------------------------------

def draft(**kw):
    base = dict(headline="Backend Engineer", summary="Builds payment services.", skills_to_highlight=["Python"],
                experience_edits=[], projects_to_highlight=[], recommendations=[], claims_requiring_confirmation=[])
    return rt.TailoredResumeDraft(**{**base, **kw})


def test_the_whole_draft_check_still_accepts_good_drafts_and_rejects_the_same_things():
    rt._validate_draft_quality(draft())
    rt._validate_draft_quality(draft(headline=None, summary=""))              # nothing to suggest is fine
    for bad, message in ((draft(headline="---"), "no meaningful text"), (draft(headline="Engineer"), "incomplete"),
                         (draft(skills_to_highlight=["Python", ""]), "empty or meaningless"),
                         (draft(summary="and " * 10), "excessive repetition")):
        with pytest.raises(rt.OutputQualityError, match=message):
            rt._validate_draft_quality(bad)


# --- tailor_resume: the public behaviour must not change ----------------------------------------

DOCS = [Document(page_content="Work Experience: Backend Engineer | PayLoop\n• Cut query latency 45% with a Redis caching layer in Python services.",
                 metadata={"source_section": "experience"})]
PROFILE = {"summary": "Backend engineer.", "skills": ["Python, Redis, Kafka"],
           "experience": ["Backend Engineer | PayLoop, Pune\nJun 2020 – Dec 2021\n• Cut query latency 45% with a Redis caching layer in Python services."]}
JD = "Backend Engineer. Build high-throughput services with Python, Kafka and Kubernetes."
GOOD_HEADLINE = {"headline": "Backend Engineer | Python, Redis, Kafka"}
GOOD_SUMMARY = {"summary": "Backend engineer who cut query latency 45% with a Redis caching layer in Python services."}
REMAINING = {"skills_to_highlight": ["Python"], "experience_edits": ["Lead with the 45% latency result."], "projects_to_highlight": [],
             "recommendations": ["Mention Redis near the top."], "claims_requiring_confirmation": []}


def tailor(llm, profile=PROFILE, jd=JD):
    retriever = MagicMock(); retriever.invoke.return_value = DOCS
    with patch.object(rt, "get_retriever", return_value=retriever), patch.object(rt, "tailor_agent", llm):
        return rt.tailor_resume("p1", profile, jd)


def test_tailor_resume_keeps_its_response_structure():
    llm = FakeLLM(GOOD_HEADLINE, GOOD_SUMMARY, REMAINING)
    out = tailor(llm)
    assert set(out) == {"profile_id", "status", "draft", "evidence_used"}
    assert out["profile_id"] == "p1" and out["status"] == "pending_review"
    assert set(out["draft"]) == set(rt.TailoredResumeDraft.model_fields)
    assert out["draft"]["headline"] == GOOD_HEADLINE["headline"] and out["draft"]["summary"] == GOOD_SUMMARY["summary"]
    assert out["draft"]["skills_to_highlight"] == ["Python"]
    assert out["evidence_used"] == [{"content": DOCS[0].page_content, "metadata": DOCS[0].metadata}]


def test_each_section_is_a_separate_small_model_call():
    llm = FakeLLM(GOOD_HEADLINE, GOOD_SUMMARY, REMAINING)
    tailor(llm)
    assert [title for title, _ in llm.calls] == ["HeadlineDraft", "SummaryDraft", "_RemainingDraft"]
    assert all(len(msgs[1][1]) < 4000 + 4000 + 2500 for _, msgs in llm.calls)       # prompts stay bounded


def test_tailor_resume_input_checks_are_unchanged():
    with pytest.raises(ValueError, match="must not be empty"):
        rt.tailor_resume("p1", {"summary": "x"}, "   ")
    with pytest.raises(ValueError, match="not contain enough resume information"):
        rt.tailor_resume("p1", {}, "Backend Engineer role")


# --- grounding of the headline and summary -------------------------------------------------

@pytest.mark.parametrize("bad_summary, mentioned", [
    ("Backend engineer experienced with Kubernetes and Python.", "Kubernetes"),                 # skill only the job asks for
    ("Backend engineer who cut query latency 70% with Redis.", "70"),                          # invented number
    ("Backend engineer building high-throughput services in Python.", "high-throughput"),       # wording only the job description has
    ("Backend engineer fluent in TensorFlow and Python.", "TensorFlow"),                        # made-up technology
])
def test_a_summary_that_claims_things_the_resume_lacks_is_retried_then_accepted(bad_summary, mentioned):
    llm = FakeLLM(GOOD_HEADLINE, {"summary": bad_summary}, GOOD_SUMMARY, REMAINING)
    out = tailor(llm)
    assert out["draft"]["summary"] == GOOD_SUMMARY["summary"]
    assert [title for title, _ in llm.calls] == ["HeadlineDraft", "SummaryDraft", "SummaryDraft", "_RemainingDraft"]
    assert mentioned in llm.calls[2][1][-1][1]                                                   # the retry says what was wrong


def test_only_the_failing_section_is_regenerated_and_a_section_that_keeps_failing_names_itself():
    bad = {"summary": "Backend engineer experienced with Kubernetes."}
    llm = FakeLLM(GOOD_HEADLINE, bad, bad)
    with pytest.raises(rt.ResumeTailorError, match=r"Could not generate the summary after 2 attempts: invalid output: .*Kubernetes"):
        tailor(llm)
    assert [title for title, _ in llm.calls] == ["HeadlineDraft", "SummaryDraft", "SummaryDraft"]     # headline was not redone


@pytest.mark.parametrize("bad", [
    {"summary": "The candidate is a backend engineer who knows Python."},
    {"summary": 'Backend engineer with Python"}'},
])
def test_text_about_the_candidate_or_with_json_debris_is_rejected(bad):
    with pytest.raises(rt.ResumeTailorError):
        tailor(FakeLLM(GOOD_HEADLINE, bad, bad))


def test_a_headline_borrowed_from_the_job_description_is_rejected():
    llm = FakeLLM({"headline": "Kubernetes Platform Engineer"}, GOOD_HEADLINE, GOOD_SUMMARY, REMAINING)
    assert tailor(llm)["draft"]["headline"] == GOOD_HEADLINE["headline"]
    assert [title for title, _ in llm.calls][:2] == ["HeadlineDraft", "HeadlineDraft"]    # only the headline was retried


def test_null_headline_and_summary_are_allowed_when_nothing_truthful_can_be_written():
    out = tailor(FakeLLM({"headline": None}, {"summary": None}, REMAINING))
    assert out["draft"]["headline"] is None and out["draft"]["summary"] is None


@pytest.mark.parametrize("bad_headline", [
    "Senior Full-stack developer for a high-visibility job posting and career summary: Senior Full-stack developer for a high",
    "Backend Engineer Backend Engineer Backend Engineer Backend Engineer Backend Engineer",
])
def test_run_on_and_repeating_headlines_are_rejected(bad_headline):
    with pytest.raises(rt.ResumeTailorError, match="headline"):
        tailor(FakeLLM({"headline": bad_headline}, {"headline": bad_headline}))


def test_a_pasted_header_or_copied_sentence_is_not_accepted_as_a_summary():
    pasted_header = {"summary": "Backend Engineer  | PayLoop, Pune"}
    copied = {"summary": "Cut query latency 45% with a Redis caching layer in Python services."}
    for bad in (pasted_header, copied):
        with pytest.raises(rt.ResumeTailorError, match="summary"):
            tailor(FakeLLM(GOOD_HEADLINE, bad, bad))


def test_a_short_headline_that_matches_a_real_title_is_fine():
    assert tailor(FakeLLM({"headline": "Backend Engineer"}, GOOD_SUMMARY, REMAINING))["draft"]["headline"] == "Backend Engineer"


def test_degenerate_numeric_headline_is_rejected_and_grounded_headline_accepted():
    bad = {"headline": "10000000000000000000000000000000000000000000000000"}
    llm = FakeLLM(bad, GOOD_HEADLINE, GOOD_SUMMARY, REMAINING)
    out = tailor(llm)
    assert out["draft"]["headline"] == GOOD_HEADLINE["headline"]
    assert out["status"] == "pending_review"
    assert "draft" in out and "evidence_used" in out
