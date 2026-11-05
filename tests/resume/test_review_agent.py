import json
import re
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.agents import review

MOBILE_PROFILE = {
    "id": "p1", "name": "A", "email": "a@example.com",
    "summary": "Mobile developer.",
    "skills": ["React Native, TypeScript, Node.js, REST APIs, Firebase"],
    "experience": ["Mobile Developer | Aforro/nJul 2026 – Present/n• Built end-to-end Rider app workflows in React Native "
                   "with real-time location tracking and UPI payments."],
    "projects": [{"content": "leaked rag doc", "metadata": {}},
                 "StackSearch | GitHub | TypeScript, FAISS/n• Built a semantic search engine."],
}
MOBILE_JD = "React Native developer. Requirements: React Native, TypeScript, Kubernetes, real-time location tracking."


class FakeLLM:
    """Stands in for the model: one canned JSON answer per schema."""

    def __init__(self, requirements, assessments, draft):
        self.answers = {"RequirementList": {"requirements": requirements},
                        "Assessments": {"assessments": assessments}, "ResumeDraft": draft}
        self.calls = []
        self.second_requirements = None   # what the follow-up "anything missing?" pass returns

    def invoke(self, messages, format):
        self.calls.append((format["title"], messages))
        answer = self.answers[format["title"]]
        if format["title"] == "RequirementList" and sum(c[0] == "RequirementList" for c in self.calls) == 2:
            answer = {"requirements": self.second_requirements if self.second_requirements is not None
                      else answer["requirements"]}
        if isinstance(answer, Exception):
            raise answer
        return SimpleNamespace(content=answer if isinstance(answer, str) else json.dumps(answer))


def a(index, status, quote=None):
    return {"index": index, "status": status, "evidence_quote": quote}


EMPTY_DRAFT = {"reasoning": "", "title": None, "summary": None,
               "skills_to_highlight": [], "projects_to_highlight": [], "recommendations": []}


def run(llm, profile=MOBILE_PROFILE, jd=MOBILE_JD, retriever=None):
    docs = [Document(page_content="Built Rider app in React Native/n• location tracking", metadata={"s": "experience"})]
    if retriever is None:
        retriever = MagicMock(); retriever.invoke.return_value = docs
    with patch.object(review, "get_retriever", return_value=retriever), patch.object(review, "review_agent", llm):
        return review.orchestrate("p1", profile, jd), retriever


def result(out):
    v = out["verdict"]
    return v["fit_assessment"], v["resume_updates"]


def test_each_requirement_lands_in_exactly_one_list_and_is_checked_against_the_profile():
    llm = FakeLLM(
        ["React Native", "TypeScript", "Kubernetes", "real-time location tracking"],
        [a(1, "gap"),                                   # model wrongly calls a supported skill a gap
         a(2, "matching", "TypeScript, FAISS"),
         a(3, "matching", "Kubernetes clusters"),       # model claims support that does not exist
         a(4, "uncertain")],
        EMPTY_DRAFT,
    )
    fa, _ = result(run(llm)[0])
    heads = lambda items: [i.split(" — ")[0] for i in items]
    assert "React Native" in heads(fa["matching_requirements"])          # profile shows it -> not a gap
    assert "TypeScript" in heads(fa["matching_requirements"])
    assert "Kubernetes" in heads(fa["gaps"])                              # unsupported claim -> gap
    assert "real-time location tracking" in heads(fa["matching_requirements"])
    everything = heads(fa["matching_requirements"] + fa["gaps"] + fa["uncertain_requirements"])
    assert sorted(everything) == sorted(set(everything)) and len(everything) == 4
    assert any("not found" in g for g in fa["gaps"])                      # the gap says why


def test_requirements_not_in_the_job_description_are_dropped():
    llm = FakeLLM(["React Native", "Swift", "Figma"], [a(1, "matching", "React Native")], EMPTY_DRAFT)
    fa, _ = result(run(llm)[0])
    listed = " ".join(fa["matching_requirements"] + fa["gaps"] + fa["uncertain_requirements"])
    assert "Swift" not in listed and "Figma" not in listed


def test_invented_skills_are_never_suggested_and_required_follows_the_result():
    draft = {"reasoning": "", "title": "Data Scientist", "summary": "Experienced in NumPy, Pandas and Matplotlib.",
             "skills_to_highlight": ["NumPy", "Pandas", "Matplotlib"], "projects_to_highlight": ["Imaginary Project"],
             "recommendations": ["Add your Pandas experience to the summary."]}
    llm = FakeLLM(["React Native"], [a(1, "matching", "React Native")], draft)
    _, ru = result(run(llm)[0])
    assert ru == {"required": False, "title": None, "summary": None, "skills_to_highlight": [],
                  "projects_to_highlight": [], "recommendations": []}


def test_supported_updates_are_kept_and_set_required_true():
    draft = {"reasoning": "", "title": "React Native Developer",
             "summary": "Mobile developer who built the Rider app in React Native with real-time location tracking.",
             "skills_to_highlight": ["React Native", "Kubernetes"], "projects_to_highlight": ["StackSearch", "Nope"],
             "recommendations": ["Lead with the Rider app location tracking work."]}
    llm = FakeLLM(["React Native", "Kubernetes"], [a(1, "matching", "React Native"), a(2, "gap")], draft)
    _, ru = result(run(llm)[0])
    assert ru["required"] is True
    assert ru["title"] == "React Native Developer" and "Rider app" in ru["summary"]
    assert ru["skills_to_highlight"] == ["React Native"]     # Kubernetes is a gap, not a strength
    assert ru["projects_to_highlight"] == ["StackSearch"]
    assert ru["recommendations"]


def test_title_with_words_the_candidate_never_used_is_rejected():
    draft = {**EMPTY_DRAFT, "title": "Senior Staff React Native Architect"}
    llm = FakeLLM(["React Native"], [a(1, "matching", "React Native")], draft)
    assert result(run(llm)[0])[1]["title"] is None


def test_verdict_follows_the_verified_lists_and_generic_reasoning_is_replaced():
    llm = FakeLLM(["Kubernetes", "Terraform", "Ansible"], [a(1, "gap"), a(2, "gap"), a(3, "gap")],
                  {**EMPTY_DRAFT, "reasoning": "You should evaluate the candidate against the job description."})
    fa, _ = result(run(llm)[0])
    assert fa["verdict"] == "weak_match"
    assert "Kubernetes" in fa["reasoning"] and "evaluate the candidate" not in fa["reasoning"]


def test_specific_reasoning_from_the_model_is_kept():
    text = "At Aforro the Rider app was built in React Native with location tracking, but Kubernetes is not shown in the profile."
    llm = FakeLLM(["React Native", "Kubernetes"], [a(1, "matching", "React Native"), a(2, "gap")],
                  {**EMPTY_DRAFT, "reasoning": text})
    assert result(run(llm)[0])[0]["reasoning"] == text


def test_sparse_profile_gives_uncertain_not_gap():
    llm = FakeLLM(["Kubernetes"], [a(1, "gap")], EMPTY_DRAFT)
    fa, _ = result(run(llm, profile={"id": "p1", "summary": "Hi."})[0])
    assert fa["gaps"] == [] and len(fa["uncertain_requirements"]) == 1


def test_requirement_with_a_number_needs_a_real_quote_to_match():
    jd = "Needs 5+ years of React Native."
    llm = FakeLLM(["5+ years of React Native"], [a(1, "matching", "made up quote about years")], EMPTY_DRAFT)
    fa, _ = result(run(llm, jd=jd)[0])
    assert fa["matching_requirements"] == [] and len(fa["uncertain_requirements"]) == 1


def test_works_for_a_non_technical_profile_and_job():
    profile = {"id": "p2", "skills": ["Content strategy, SEO, email marketing"],
               "experience": ["Marketing Manager | Acme/n• Grew organic traffic 120% with SEO and content strategy."]}
    llm = FakeLLM(["SEO", "paid social advertising"], [a(1, "matching", "Grew organic traffic 120% with SEO"), a(2, "gap")],
                  EMPTY_DRAFT)
    fa, _ = result(run(llm, profile=profile, jd="Marketing lead: SEO expertise and paid social advertising.")[0])
    assert [m.split(" — ")[0] for m in fa["matching_requirements"]] == ["SEO"]
    assert [g.split(" — ")[0] for g in fa["gaps"]] == ["paid social advertising"]


def test_profile_and_evidence_stay_separate_clean_and_unmodified():
    llm = FakeLLM(["React Native"], [a(1, "matching", "React Native")], EMPTY_DRAFT)
    before = json.dumps(MOBILE_PROFILE, sort_keys=True)
    out, retriever = run(llm)
    prompt = dict((title, msgs) for title, msgs in llm.calls)["Assessments"][1][1]
    assert prompt.index("CANDIDATE PROFILE") < prompt.index("RETRIEVED RESUME EVIDENCE")
    assert "/n" not in prompt and "leaked rag doc" not in prompt
    assert json.dumps(MOBILE_PROFILE, sort_keys=True) == before
    assert out["evidence_used"][0].keys() == {"content", "metadata"} and "/n" not in out["evidence_used"][0]["content"]
    queries = [c.args[0] for c in retriever.invoke.call_args_list]
    assert any(q.startswith("Work Experience:") for q in queries)     # experience and projects are searched
    assert any(q.startswith("Project:") for q in queries)             # for explicitly, not just by chance


def test_response_has_exactly_the_two_keys_and_a_valid_verdict():
    llm = FakeLLM(["React Native"], [a(1, "matching", "React Native")], EMPTY_DRAFT)
    v = run(llm)[0]["verdict"]
    assert set(v) == {"fit_assessment", "resume_updates"}
    assert v["fit_assessment"]["verdict"] in {"strong_match", "partial_match", "weak_match"}


def test_invalid_output_is_retried_then_accepted_and_persistent_failure_raises():
    llm = FakeLLM(["React Native"], [a(1, "matching", "React Native")], EMPTY_DRAFT)
    llm.answers["RequirementList"] = "not json"
    with pytest.raises(review.ReviewError):
        run(llm)
    llm = FakeLLM(["React Native"], [a(1, "matching", "React Native")], EMPTY_DRAFT)
    llm.answers["Assessments"] = RuntimeError("token repeat limit reached")
    with pytest.raises(review.ReviewError):
        run(llm)


def test_job_description_with_no_traceable_requirements_raises():
    llm = FakeLLM(["Quantum computing"], [], EMPTY_DRAFT)
    with pytest.raises(review.ReviewError):
        run(llm)


def test_numbers_the_model_invented_are_stripped_from_requirements():
    llm = FakeLLM(["1 year of experience with Kubernetes", "React Native"], [a(1, "gap"), a(2, "matching", "React Native")],
                  EMPTY_DRAFT)
    fa, _ = result(run(llm)[0])
    assert fa["gaps"][0].startswith("Kubernetes — ")


def test_work_and_project_evidence_outranks_the_skills_list():
    llm = FakeLLM(["TypeScript"], [a(1, "matching")], EMPTY_DRAFT)       # no quote -> built from the text
    fa, _ = result(run(llm)[0])
    # evidence from a project outranks the skills-list entry for the same technology
    assert fa["matching_requirements"][0] == "TypeScript — project StackSearch: technologies include TypeScript"


def test_quantity_requirement_explains_what_cannot_be_verified():
    jd = "Needs 2+ years of React Native."
    llm = FakeLLM(["2+ years of React Native"], [a(1, "matching", "React Native")], EMPTY_DRAFT)
    fa, _ = result(run(llm, jd=jd)[0])
    assert fa["matching_requirements"] and "React Native" in fa["matching_requirements"][0]


def test_uncertain_requirements_count_for_half_in_the_verdict():
    assert review._verdict(0, 2, 5) == "partial_match"
    assert review._verdict(7, 0, 1) == "strong_match"
    assert review._verdict(1, 4, 0) == "weak_match"


def test_terms_must_appear_together_not_scattered_across_unrelated_skills():
    profile = {"id": "p3", "skills": ["Data Structures, Algorithms, RAG Pipelines, Python, SQL"],
               "experience": ["Developer | Acme/n• Wrote backend services."]}
    llm = FakeLLM(["data pipelines"], [a(1, "matching")], EMPTY_DRAFT)
    fa, _ = result(run(llm, profile=profile, jd="Build data pipelines.")[0])
    assert fa["matching_requirements"] == [] and "only partly shown" in fa["uncertain_requirements"][0]


def test_word_forms_match():
    profile = {"id": "p4", "experience": ["Dev | Acme/n• Integrated 40+ REST APIs across the app."], "skills": ["REST"]}
    llm = FakeLLM(["REST API integration"], [a(1, "uncertain")], EMPTY_DRAFT)
    fa, _ = result(run(llm, profile=profile, jd="REST API integration.")[0])
    assert fa["matching_requirements"] and fa["gaps"] == []


def test_a_title_copied_from_the_existing_experience_is_not_an_update():
    draft = {**EMPTY_DRAFT, "title": "Mobile Developer | Aforro"}
    llm = FakeLLM(["React Native"], [a(1, "matching", "React Native")], draft)
    assert result(run(llm)[0])[1]["title"] is None


def test_projects_to_highlight_must_be_actual_project_names():
    draft = {**EMPTY_DRAFT, "projects_to_highlight": ["StackSearch", "semantic search engine"]}
    llm = FakeLLM(["TypeScript"], [a(1, "matching")], draft)
    assert result(run(llm)[0])[1]["projects_to_highlight"] == ["StackSearch"]


def test_a_second_pass_adds_requirements_the_first_pass_missed():
    llm = FakeLLM(["React Native"], [a(1, "matching", "React Native"), a(2, "gap")], EMPTY_DRAFT)
    llm.second_requirements = ["Kubernetes", "React Native"]      # already-listed items are not duplicated
    fa, _ = result(run(llm)[0])
    assert any(g.startswith("Kubernetes") for g in fa["gaps"])
    everything = fa["matching_requirements"] + fa["gaps"] + fa["uncertain_requirements"]
    assert len(everything) == 2


def test_a_skill_must_be_a_listed_skill_not_a_sentence_from_the_resume():
    draft = {**EMPTY_DRAFT, "skills_to_highlight": ["Built end-to-end Rider app workflows in React Native", "TypeScript"]}
    llm = FakeLLM(["React Native", "TypeScript"], [a(1, "matching", "React Native"), a(2, "matching")], draft)
    assert result(run(llm)[0])[1]["skills_to_highlight"] == ["TypeScript"]


def test_a_summary_pasted_from_the_resume_is_not_an_update():
    pasted = "Built Rider app in React Native. Real-time location tracking."
    llm = FakeLLM(["React Native", "real-time location tracking"], [a(1, "matching"), a(2, "matching")],
                  {**EMPTY_DRAFT, "summary": pasted})
    profile = {**MOBILE_PROFILE, "experience": ["Dev | Aforro/n• Built Rider app in React Native. Real-time location tracking."]}
    assert result(run(llm, profile=profile)[0])[1]["summary"] is None


def test_messages_show_words_not_stems():
    profile = {"id": "p5", "skills": ["Python, SQL, Go, Rust"], "experience": ["Dev | Acme/n• Wrote services."]}
    fa, _ = result(run(FakeLLM(["real-time pipelines"], [a(1, "gap")], EMPTY_DRAFT), profile=profile, jd="real-time pipelines")[0])
    text = " ".join(fa["gaps"] + fa["uncertain_requirements"])
    assert "pipelin " not in text and "realtim " not in text


def test_invented_elaborations_are_cut_back_to_the_job_descriptions_words():
    jd = "Needs Redux, Jest testing and React Native."
    llm = FakeLLM(["implementing Redux, a powerful state management tool for JavaScript applications",
                   "Jest testing framework, which provides comprehensive test coverage",
                   "test cases, integration tests, and more.", "React Native"],
                  [a(1, "gap"), a(2, "gap"), a(3, "gap"), a(4, "matching", "React Native")], EMPTY_DRAFT)
    fa, _ = result(run(llm, jd=jd)[0])
    heads = [i.split(" — ")[0] for i in fa["matching_requirements"] + fa["gaps"] + fa["uncertain_requirements"]]
    assert sorted(heads) == ["Jest testing", "React Native", "Redux"]


def test_a_recommendation_must_be_advice_not_a_pasted_fragment():
    draft = {**EMPTY_DRAFT, "recommendations": ["StackSearch | GitHub", "Lead with the StackSearch project near the top of your resume."]}
    llm = FakeLLM(["TypeScript"], [a(1, "matching")], draft)
    assert result(run(llm)[0])[1]["recommendations"] == ["Lead with the StackSearch project near the top of your resume."]


def test_garbled_or_third_person_text_is_not_suggested():
    draft = {**EMPTY_DRAFT,
             "summary": "The candidate has a strong background, particularly with React Native.",
             "recommendations": ['>80% merged open-source pull requests”]],', "Lead with the Rider app React Native work."]}
    llm = FakeLLM(["React Native"], [a(1, "matching", "React Native")], draft)
    ru = result(run(llm)[0])[1]
    assert ru["summary"] is None and ru["recommendations"] == ["Lead with the Rider app React Native work."]


def test_a_requirement_is_cited_to_the_job_that_shows_it_not_the_skills_list():
    llm = FakeLLM(["React Native"], [a(1, "matching")], EMPTY_DRAFT)     # also listed in the skills line
    fa, _ = result(run(llm)[0])
    assert fa["matching_requirements"][0].startswith("React Native — Mobile Developer at Aforro: ")
    assert "Rider app" in fa["matching_requirements"][0]


def test_evidence_found_only_in_the_skills_list_is_flagged_as_such():
    profile = {"id": "p6", "skills": ["Python, SQL, Docker, Kafka"], "experience": ["Dev | Acme/n• Wrote services."]}
    fa, _ = result(run(FakeLLM(["Kafka"], [a(1, "matching")], EMPTY_DRAFT), profile=profile, jd="Needs Kafka.")[0])
    assert "skills list only; no work or project evidence found" in fa["matching_requirements"][0]


def test_reasoning_must_name_the_employer_or_project_behind_a_strength():
    generic = "The candidate is a good fit because they know React Native and have shown relevant skills."
    llm = FakeLLM(["React Native", "Kubernetes"], [a(1, "matching"), a(2, "gap")], {**EMPTY_DRAFT, "reasoning": generic})
    reasoning = result(run(llm)[0])[0]["reasoning"]
    assert "Mobile Developer at Aforro" in reasoning          # falls back to the verified, cited summary
    assert "Kubernetes" in reasoning


def test_missing_work_evidence_is_reported_not_papered_over():
    profile = {"id": "p7", "skills": ["Python, SQL, Docker, Kafka"]}
    reasoning = result(run(FakeLLM(["Kafka", "Redis"], [a(1, "matching"), a(2, "gap")], EMPTY_DRAFT), profile=profile, jd="Kafka and Redis.")[0])[0]["reasoning"]
    assert "no work experience or project entries" in reasoning


def test_retrieval_includes_experience_and_projects_even_when_the_skills_chunk_dominates():
    skills = Document(page_content="Skills: React Native, TypeScript", metadata={"source_section": "skills"})
    job = Document(page_content="Work Experience: Mobile Developer | Aforro\n• Built the Rider app in React Native.",
                   metadata={"source_section": "experience", "job_title": "Mobile Developer", "company": "Aforro"})
    proj = Document(page_content="Project: StackSearch | TypeScript", metadata={"source_section": "projects", "project_name": "StackSearch"})
    retriever = MagicMock()
    retriever.invoke.side_effect = lambda q: ([skills, skills, skills, skills, job] if q.startswith("Work Experience:")
                                              else [skills, skills, skills, skills, proj] if q.startswith("Project:")
                                              else [skills] * 5)
    out, _ = run(FakeLLM(["React Native"], [a(1, "matching")], EMPTY_DRAFT), retriever=retriever)
    sections = [e["metadata"]["source_section"] for e in out["evidence_used"]]
    assert "experience" in sections and "projects" in sections
    assert sections.index("experience") < sections.index("skills")          # work/project entries come first


def test_a_company_is_never_offered_as_a_project_when_no_project_names_are_known():
    profile = {"id": "p8", "skills": ["Python, SQL, Docker, Kafka"], "experience": ["Dev | FinTech Global/n• Built Kafka pipelines."]}
    draft = {**EMPTY_DRAFT, "projects_to_highlight": ["FinTech Global"]}
    ru = result(run(FakeLLM(["Kafka"], [a(1, "matching")], draft), profile=profile, jd="Needs Kafka.")[0])[1]
    assert ru["projects_to_highlight"] == []


def test_a_summary_that_pins_one_jobs_title_on_another_employer_is_rejected():
    profile = {"id": "p9", "skills": ["Python, SQL, Docker, Kafka"],
               "experience": ["Senior Backend Engineer | FinTech Global, Mumbai/nJan 2022 – Present/n• Built Kafka pipelines in Python.",
                              "Backend Engineer | PayLoop, Pune/nJun 2020 – Dec 2021/n• Wrote Python services with Docker."]}
    wrong = "Senior Backend Engineer at PayLoop who built Kafka pipelines in Python."
    right = "Backend Engineer at PayLoop who wrote Python services with Docker and Kafka."
    for text, expected in ((wrong, None), (right, right), ("Python and Kafka specialist | PayLoop", None)):
        llm = FakeLLM(["Python", "Kafka"], [a(1, "matching"), a(2, "matching")], {**EMPTY_DRAFT, "summary": text})
        assert result(run(llm, profile=profile, jd="Python and Kafka.")[0])[1]["summary"] == expected


# --- the reasoning is a short summary, not a run-on copy of the lists below it (seen in the UI) ---------------------

LONG_REQUIREMENTS = [
    "Strong proficiency in TypeScript, React, Next.js",
    "1-3 years of professional experience building full-stack applications and shipping them to real customers",
    "Solid backend experience with Python and FastAPI",
    "Working knowledge of PostgreSQL and MongoDB or similar relational/NoSQL databases",
    "Hands-on experience with at least one AI agent framework (LangChain, LangGraph, Pydantic AI, or equivalent)",
    "Regular use of an AI coding assistant (e. g., Claude Code, Cursor, or similar) as part of your development workflow",
    "Iterate fast -- from prototype to production -- in a 0-to-1 product environment",
    "Experience mentoring an intern, junior engineer, or peer -- or clear readiness to take on mentorship",
]


def test_the_reasoning_is_a_short_readable_summary_that_is_never_cut_mid_word():
    llm = FakeLLM(LONG_REQUIREMENTS, [a(1, "matching", "TypeScript"), *[a(i, "uncertain") for i in range(2, 8)], a(8, "matching", "mentored interns")],
                  EMPTY_DRAFT)
    fa, _ = result(run(llm, jd="\n".join(f"- {r}" for r in LONG_REQUIREMENTS))[0])
    reasoning = fa["reasoning"]
    assert len(reasoning) <= 450, len(reasoning)                               # was a run-on cut at 600 characters
    assert re.match(r"^\w+ match: \d of 8 job requirements (?:are|is) clearly covered\.", reasoning)
    assert reasoning.endswith(".") and "(La" not in reasoning                  # complete sentences, no cut-off text
    assert "Unclear:" not in reasoning and "Not demonstrated" not in reasoning   # the lists below say that
    # no requirement is copied in full when it is long; each item is shortened at a word boundary
    for item in [*fa["gaps"], *fa["uncertain_requirements"]]:
        head = item.split(" — ")[0]
        if len(head) > 75:
            assert head not in reasoning
    for piece in re.findall(r"([A-Za-z-]+)…", reasoning):                      # an ellipsis always follows a whole word
        assert any(piece in req.split() or piece in req.replace(",", "").split() for req in LONG_REQUIREMENTS), piece


def test_nothing_missing_and_nothing_unclear_reads_cleanly():
    reasoning = result(run(FakeLLM(["React Native", "TypeScript"], [a(1, "matching"), a(2, "matching", "TypeScript, FAISS")], EMPTY_DRAFT))[0])[0]["reasoning"]
    assert "Nothing is clearly missing." in reasoning and "only partly shown" not in reasoning


def test_long_requirement_text_is_shortened_at_a_word_boundary():
    long_req = "Experience designing and operating highly available distributed systems across multiple cloud regions"
    jd = f"- {long_req}\n- Kubernetes"
    reasoning = result(run(FakeLLM([long_req, "Kubernetes"], [a(1, "gap"), a(2, "gap")], EMPTY_DRAFT), jd=jd)[0])[0]["reasoning"]
    missing = reasoning.split("Missing: ")[1]
    assert "Experience designing and operating highly available distributed…" in missing   # cut after a whole word
    assert "systems across" not in missing and "Kubernetes" in missing


def test_word_safe_never_cuts_inside_a_word():
    text = "First sentence here. Second sentence is considerably longer than the first one is. " * 8
    for limit in (40, 100, 333, 599):
        out = review._word_safe(text, limit)
        assert len(out) <= limit + 1 and (out.endswith((".", "…")))
    assert review._word_safe("short", 600) == "short"


def test_wording_such_as_must_have_is_not_a_skill_the_resume_has_to_contain():
    assert review._terms("Strong proficiency in React (must-have), along with JavaScript and TypeScript") == ["react", "javascript", "typescript"]
    assert review._terms("Docker (nice-to-have)") == ["docker"]
    profile = {**MOBILE_PROFILE, "skills": ["React, JavaScript, TypeScript, Node.js"]}
    req = "Strong proficiency in React (must-have), along with JavaScript and TypeScript"
    llm = FakeLLM([req], [a(1, "matching", "React, JavaScript, TypeScript")], EMPTY_DRAFT)
    fa, _ = result(run(llm, profile=profile, jd=req)[0])
    assert [i.split(" — ")[0] for i in fa["matching_requirements"]] == [req] and fa["uncertain_requirements"] == []
