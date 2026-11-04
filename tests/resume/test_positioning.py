"""Resume positioning: role-specific headline, grounded summary, skills to highlight, per-field suggestions."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.agents import positioning as pos
from app.agents import resume_tailor as rt

PROFILE = {
    "summary": "Software engineer with 1 year of experience shipping mobile, backend, and AI features in TypeScript, Python, and React Native. Strongest in payments.",
    "skills": ["OOP, Python, JavaScript, TypeScript, React Native, Firebase, LangChain, FAISS, PostgreSQL, Docker"],
    "programming_languages": ["Python", "TypeScript", "JavaScript"],
    "experience": [
        "Junior Mobile Application Developer | Aforro, Bengaluru\nJul 2026 – Present\n"
        "• Delivered 6 rider order flows (pickup, delivery, returns) via REST APIs with real-time Firebase alerts.\n"
        "• Designed a custom Razorpay SDK payment flow for UPI, cards and wallets.",
        "SDE Intern | TechKareer, Bengaluru\nJan 2026 – Mar 2026\n• Optimized PostgreSQL schemas and queries for production features.",
    ],
    "projects": [
        "StackSearch | Node.js, Python, LangChain, FAISS\nJan 2026 – Present\n"
        "• Achieved sub-100 ms retrieval over 2,500+ documentation pages with a RAG-based semantic search engine.",
    ],
}
JD_RN = ("About the job\nBuild cross-platform mobile apps using React Native.\n\nApply Now\n\nJob Description\n\n"
         "As a React Native Developer, you will be responsible for shipping apps. Requirements: React Native, JavaScript, Firebase.")
BLOB = repr(PROFILE)


# ------------------------------------------------------------------------------------------- the job's role

@pytest.mark.parametrize("jd, title, role", [
    (JD_RN, None, "React Native Developer"),
    ("We are hiring a Senior Backend Engineer (Remote) to build APIs.", None, "Senior Backend Engineer"),
    ("We're looking for an experienced Python Developer who loves data.", None, "Python Developer"),
    ("Job Title: Machine Learning Engineer - Bangalore\nRequirements: Python", None, "Machine Learning Engineer"),
    ("Backend Engineer\nAbout us: payments.", None, "Backend Engineer"),
    ("Join us as a Software Engineer II, working on search.", None, "Software Engineer"),
    ("About the job\nBuild things. Apply now. We value teamwork.", None, None),
    ("", None, None),
    ("anything", "Staff Android Developer (m/f/d)", "Staff Android Developer"),
    ("anything", "Marketing Specialist", None),                       # not a role noun we recognise
    ("anything", "   ", None),
])
def test_target_role(jd, title, role):
    assert pos.target_role(jd, title) == role


@pytest.mark.parametrize("role, supported", [
    ("React Native Developer", "React Native Developer"),                 # every distinguishing word is in the resume
    ("Senior React Native Developer", "React Native Developer"),          # seniority is never claimed
    ("Lead Python Engineer", "Python Engineer"),
    ("Founding Engineer", None),                                          # "founding" is not evidenced
    ("Kafka Streaming Engineer", None),                                   # skills the resume lacks
    ("Engineering Manager", None), ("Director of Engineering", None), ("Head of Mobile Developer", None),
    ("Software Engineer", "Software Engineer"),                           # "software" is in this resume (its own summary says so)
    ("Engineer", None), ("Developer", None),                              # nothing distinguishing: no better than the real title
    (None, None), ("", None),
])
def test_a_role_is_claimed_only_when_the_resume_evidences_it(role, supported):
    assert pos.supported_role(role, BLOB) == supported


# ------------------------------------------------------------------------------------------------ headline

def test_the_headline_names_the_role_not_the_current_title_and_does_not_repeat_the_role_skills():
    h = pos.build_headline("React Native Developer", "Junior Mobile Application Developer", ["React Native", "JavaScript", "Firebase", "Python"], None, 0)
    assert h == "React Native Developer | JavaScript, Firebase, Python"
    assert "Junior" not in h and "Mobile Application" not in h


def test_without_an_evidenced_role_the_headline_uses_the_real_title():
    assert pos.build_headline(None, "Junior Mobile Developer", ["Python", "Redis"], None, 0) == "Junior Mobile Developer | Python, Redis"
    assert pos.build_headline(None, None, ["Python", "Redis", "Go", "SQL"], None, 0) == "Python, Redis, Go"
    assert pos.build_headline(None, None, [], None, 0) is None


def test_headline_variants_differ_and_stay_short():
    skills = ["JavaScript", "Firebase", "Python"]
    variants = {pos.build_headline("React Native Developer", "Dev", skills, "1 year of experience", v) for v in range(4)}
    assert len(variants) == 4 and all(len(v) <= pos.HEADLINE_MAX for v in variants)
    assert pos.build_headline("React Native Developer", "Dev", skills, None, 7) == pos.build_headline("React Native Developer", "Dev", skills, None, 7 % 3)
    assert pos.build_headline("X" * 200 + " Developer", None, ["A"], None, 0) is not None
    assert len(pos.build_headline("A" * 150, None, ["Skill"], None, 0)) <= pos.HEADLINE_MAX


def test_years_of_experience_come_from_the_resume_summary_only():
    assert pos._years_phrase("Software engineer with 1 year of experience shipping things") == "1 year of experience"
    assert pos._years_phrase("Engineer with 5+ years of professional experience.") == "5+ years of experience"
    assert pos._years_phrase("Engineer.") is None and pos._years_phrase(None) is None


# ------------------------------------------------------------------------------------------------- summary

def _summary(variant, **over):
    args = dict(profile_summary=PROFILE["summary"], title="Junior Mobile Application Developer", company="Aforro",
                skills=["React Native", "JavaScript", "Firebase", "Razorpay"],
                accomplishments=["Delivered 6 rider order flows via REST APIs", "Optimized PostgreSQL schemas and queries for production features"],
                project_lines=["Achieved sub-100 ms retrieval over 2,500+ documentation pages (StackSearch)"], variant=variant)
    args.update(over)
    return pos.build_summary(**args)


def test_the_summary_is_the_resumes_own_statements_without_repeating_skills():
    s = _summary(0)
    assert s.startswith("Software engineer with 1 year of experience shipping mobile, backend, and AI features in TypeScript, Python, and React Native.")
    assert "Delivered 6 rider order flows via REST APIs." in s
    assert s.count("React Native") == 1                                        # already said in the first sentence: not listed again
    assert "Also hands-on with JavaScript, Firebase and Razorpay." in s


def test_summary_variants_are_different_and_all_grounded_in_the_resume():
    texts = [_summary(v) for v in range(3)]
    assert len(set(texts)) == 3
    assert "StackSearch" in texts[1] and texts[2].startswith("Junior Mobile Application Developer at Aforro.")
    for text in texts:
        assert len(text) <= pos.SUMMARY_MAX and text.endswith(".")
        clean = lambda value: {w.strip(".,()") for w in value.lower().split() if len(w.strip(".,()")) > 3}
        for sentence in [x for x in text.split(". ") if x]:
            words = clean(sentence)
            source = (PROFILE["summary"] + " Delivered rider order flows REST APIs Optimized PostgreSQL schemas queries production features "
                      "Achieved sub-100 ms retrieval over 2,500+ documentation pages StackSearch Also hands-on with JavaScript Firebase and Razorpay "
                      "Junior Mobile Application Developer at Aforro").lower()
            assert words <= clean(source), sentence


def test_the_summary_never_exceeds_its_limit_and_cuts_only_between_sentences():
    long_proof = ["Built " + "an extraordinarily detailed distributed system " * 12]
    s = _summary(0, accomplishments=long_proof)
    assert len(s) <= pos.SUMMARY_MAX and s.endswith(".")
    huge_intro = "Engineer with " + "many years of experience building systems and teams across companies " * 10 + "."
    s = _summary(0, profile_summary=huge_intro)
    assert len(s) <= pos.SUMMARY_MAX and s.endswith(".")


def test_without_a_resume_summary_the_summary_leads_with_the_real_role():
    s = _summary(0, profile_summary=None)
    assert s.startswith("Junior Mobile Application Developer at Aforro.")
    assert _summary(0, profile_summary="I build things with care and attention every day.").startswith("Junior Mobile Application Developer at Aforro.")


def test_a_sparse_profile_still_gives_a_summary_or_nothing_never_an_error():
    assert pos.build_summary(profile_summary=None, title=None, company=None, skills=[], accomplishments=[], project_lines=[], variant=0) is None
    assert pos.build_summary(profile_summary=None, title="Engineer", company=None, skills=[], accomplishments=[], project_lines=[], variant=2) == "Engineer."


# -------------------------------------------------------------------------- through the real draft builder

def _draft(variant=0, jd=JD_RN, title=None):
    d, facts = rt._build_draft(PROFILE, jd, "", variant=variant, target_title=title)
    return d, facts


def test_the_draft_headline_is_role_specific_for_a_job_the_resume_supports():
    d, facts = _draft()
    assert d.headline.startswith("React Native Developer | ")
    assert "Junior Mobile Application Developer" not in d.headline
    assert facts["role"] == "React Native Developer"


def test_the_users_own_job_title_wins_and_is_checked_the_same_way():
    d, _ = _draft(title="Senior Python Engineer")
    assert d.headline.startswith("Python Engineer | ")                            # seniority dropped
    d, _ = _draft(title="Kafka Streaming Engineer")
    assert d.headline.startswith("Junior Mobile Application Developer | ")        # not evidenced: the real title


def test_every_skill_suggestion_is_a_skill_the_profile_already_has():
    have = {s.strip().lower() for field in ("skills", "programming_languages") for e in PROFILE[field] for s in e.split(",")}
    skills = rt.suggest_field(PROFILE, JD_RN, "skills")
    assert skills and all(s.lower() in have for s in skills) and len(skills) == len({s.lower() for s in skills})
    assert skills[0] == "React Native" and "Firebase" in skills
    assert "Kubernetes" not in rt.suggest_field(PROFILE, JD_RN + " Kubernetes, Terraform, Kafka.", "skills")
    assert rt.suggest_field(PROFILE, "Cooking and baking only.", "skills") == []


def test_regenerating_gives_a_different_valid_suggestion_each_time():
    heads = [rt.suggest_field(PROFILE, JD_RN, "headline", variant=v) for v in range(3)]
    sums = [rt.suggest_field(PROFILE, JD_RN, "summary", variant=v) for v in range(3)]
    assert len(set(heads)) == 3 and len(set(sums)) == 3
    assert all(len(h) <= 120 for h in heads) and all(len(s) <= 350 for s in sums)
    assert rt.suggest_field(PROFILE, JD_RN, "summary", variant=3) == sums[0]            # cycles


def test_suggest_field_validates_its_input():
    with pytest.raises(ValueError, match="field must be"):
        rt.suggest_field(PROFILE, JD_RN, "education")
    with pytest.raises(ValueError, match="must not be empty"):
        rt.suggest_field(PROFILE, "  ", "headline")


def test_no_suggestion_contains_a_fact_that_is_not_in_the_profile():
    forbidden = ("Kubernetes", "Terraform", "Kafka", "Rust", "Senior", "Lead", "Manager")
    jd = JD_RN + " Senior lead role with Kubernetes, Terraform, Kafka and Rust."
    for field in ("headline", "summary"):
        for v in range(3):
            text = rt.suggest_field(PROFILE, jd, field, variant=v)
            assert not any(word in text for word in forbidden), (field, v, text)
