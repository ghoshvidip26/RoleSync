import pytest

from app.agents import resume_tailor as rt
from app.agents import review
from app.agents.review import _terms

RN_JD = """\
About the job
Build cross-platform mobile apps using React Native.

Apply Now

Job Description

As a React Native Developer, you will be responsible for developing mobile applications for Android and iOS platforms, focusing on performance and responsiveness.

Responsibilities

Develop mobile applications using React Native.
Integrate APIs and third-party libraries.
Debug and optimize app performance.
Deploy builds to App Store and Play Store.
Collaborate with designers and backend teams.

Requirements

1+ year of experience in mobile app development.
Strong skills in JavaScript and React Native.
Familiarity with Expo, Firebase, or native modules.
Good understanding of mobile UI/UX design principles.
"""
GENERIC_JD = ("Looking for a backend developer to design and build scalable, secure services for mobile and web applications in "
              "connected-device products. You'll own APIs, data models and service integrations, working closely with mobile, QA and DevOps colleagues.")

PROFILE = {
    "skills": ["OOP", "Data Structures & Algorithms", "Python", "JavaScript", "TypeScript", "React Native", "Firebase",
               "REST APIs", "Gemini API"],
    "experience": ["Mobile Developer | Aforro, Bengaluru\nJan 2024 – Present\n"
                   "• Built end-to-end Rider app workflows in React Native over REST APIs.\n• Integrated Firebase push notifications."],
    "projects": [],
}


def classify(req, profile=PROFILE, status="uncertain"):
    cand = review._Candidate(profile, review._profile_sections(profile), [], "", complete=True)
    answer = review.RequirementAssessment(index=1, status=status, evidence_quote=None)
    return review._classify(req, answer, cand)


# --- 1. "A, B, or C": any one option is enough --------------------------------------------------------------

@pytest.mark.parametrize("model_status", ["matching", "gap", "uncertain"])
def test_an_or_list_is_met_when_any_one_option_is_present(model_status):
    bucket, explanation, _ = classify("Familiarity with Expo, Firebase, or native modules.", status=model_status)
    assert bucket == "matching"
    assert "Firebase" in explanation and "any one of the options is enough" in explanation


def test_an_or_list_with_no_option_present_is_still_a_gap():
    bucket, explanation, _ = classify("Familiarity with Expo, Kotlin, or Swift.")
    assert bucket == "gap" and "not found" in explanation


def test_an_or_option_needs_all_of_its_own_words_in_one_place():
    # "native modules": the resume has "native" (React Native) but no "modules", and no other option is present.
    assert classify("Familiarity with Expo or native modules.")[0] != "matching"


def test_the_or_rule_does_not_apply_to_quantity_requirements():
    assert "any one of the options" not in classify("2+ years of Expo or Firebase")[1]


def test_a_requirement_without_or_still_needs_all_its_words():
    assert classify("Familiarity with Expo, Firebase and Kotlin.")[0] != "matching"


# --- 2. page boilerplate and filler are not skills ---------------------------------------------------------------

def test_boilerplate_words_are_ignored_when_matching():
    ignored = {"now", "descript", "job", "about", "ll", "play", "apply", "close", "own", "colleagu"}
    assert not ignored & set(_terms("Apply Now. Job Description. About the job. You'll own it, working closely with colleagues. Play Store"))


def test_page_boilerplate_is_not_reported_as_a_missing_skill_but_real_gaps_are():
    known = set(review._tokens(str(PROFILE)))
    claims = rt._absent_technical_terms(RN_JD, known)
    assert not {"Now", "Description", "Play", "Apply", "About"} & set(claims)
    assert {"Expo", "Android", "iOS"} <= set(claims)                  # genuinely absent from the resume


# --- 3. a skill must match fully ----------------------------------------------------------------------------------

def test_a_skill_is_highlighted_only_when_every_word_of_it_is_in_the_job():
    draft, _ = rt._build_draft(PROFILE, RN_JD, "")
    assert draft.skills == ["React Native", "JavaScript", "Firebase"]               # in the job's order
    assert "Gemini API" not in draft.skills and "Data Structures & Algorithms" not in draft.skills


def test_a_generic_job_does_not_drag_loosely_related_skills_into_the_headline():
    draft, _ = rt._build_draft(PROFILE, GENERIC_JD, "")
    assert draft.headline == "Mobile Developer"                                      # the real title, no padded skills
    assert "Gemini API" not in (draft.headline or "") and "Gemini API" not in (draft.summary or "")


def test_a_comma_joined_skills_string_is_split_into_separate_skills():
    profile = {**PROFILE, "skills": ["Python, JavaScript, React Native, Firebase, Gemini API"]}
    draft, _ = rt._build_draft(profile, RN_JD, "")
    assert draft.skills == ["React Native", "JavaScript", "Firebase"]


# --- 4. work evidence is shown ahead of a skills-list line ---------------------------------------------------------

def test_evidence_cites_the_work_when_it_covers_the_requirement():
    bucket, explanation, source = classify("Experience building apps with React Native and REST APIs.")
    assert bucket == "matching" and source == "Mobile Developer at Aforro"
    assert "Mobile Developer at Aforro:" in explanation and "listed in skills" not in explanation


def test_the_skills_list_is_cited_when_the_work_does_not_cover_the_requirement():
    bucket, explanation, source = classify("Skills in JavaScript, TypeScript and Python.")
    assert bucket == "matching" and source == "skills list" and "listed in skills" in explanation


# --- "A and B": a list of separate skills --------------------------------------------------------------------------

def test_a_list_of_separate_skills_is_met_when_each_is_present_somewhere():
    bucket, explanation, _ = classify("Strong skills in JavaScript and React Native.")
    assert bucket == "matching"
    assert "listed in skills: JavaScript" in explanation and "Mobile Developer at Aforro:" in explanation


@pytest.mark.parametrize("status", ["matching", "gap", "uncertain"])
def test_a_list_with_one_missing_skill_is_not_met(status):
    assert classify("Strong skills in JavaScript and Kotlin.", status=status)[0] != "matching"


def test_a_phrase_still_needs_its_words_together():
    profile = {"skills": ["Data Structures", "RAG Pipelines"], "experience": ["Dev | Acme\n• Wrote services."]}
    assert classify("Experience building data pipelines.", profile)[0] == "uncertain"


def test_a_retrieved_skills_line_is_split_into_items_too():
    """Regression: a whole retrieved skills line counted as one unit, so any two skills on it looked 'together'."""
    evidence = [{"content": "Skills: Data Structures, RAG Pipelines", "metadata": {"source_section": "skills"}}]
    profile = {"experience": ["Dev | Acme\n• Wrote services."]}
    cand = review._Candidate(profile, review._profile_sections(profile), evidence, evidence[0]["content"], complete=True)
    answer = review.RequirementAssessment(index=1, status="matching", evidence_quote=None)
    assert review._classify("Experience building data pipelines.", answer, cand)[0] == "uncertain"


def test_a_summary_line_is_not_cited_ahead_of_an_actual_skills_entry():
    """Regression: for "JavaScript and React Native" the proof shown was the profile's summary line
    ("Junior Mobile Developer | ... Skilled in OOP, ..."), not the skills list."""
    profile = {**PROFILE, "summary": "Mobile Developer | Aforro Skilled in OOP, Python, JavaScript, TypeScript."}
    explanation = classify("Strong skills in JavaScript and TypeScript.", profile)[1]
    assert "listed in skills: JavaScript" in explanation and "summary:" not in explanation
