"""Skills come in the order the job wants them: a job that needs Python shows Python first."""
from app.agents import positioning as pos
from app.agents import resume_tailor as rt
from app.services.resume_pdf_renderers import template_renderer as tr

SKILLS = ["Docker", "FastAPI", "Python", "PostgreSQL", "Redis", "React", "Kubernetes"]


def rank(jd, title=None, skills=SKILLS):
    return pos.rank_skills(skills, jd, title)


def test_the_skill_the_job_repeats_comes_first_even_if_it_is_mentioned_later():
    jd = "We use Docker. You will write Python services, review Python code and mentor others in Python."
    assert rank(jd)[:2] == ["Python", "Docker"]


def test_ties_keep_the_order_the_job_lists_them_in():
    jd = "Requirements: Python, FastAPI, PostgreSQL, Redis."
    assert rank(jd)[:4] == ["Python", "FastAPI", "PostgreSQL", "Redis"]


def test_required_skills_come_before_nice_to_have_ones_even_when_mentioned_earlier():
    jd = "Nice to have: Kubernetes, Docker.\nRequirements: Python and PostgreSQL."
    out = rank(jd)
    assert out.index("Python") < out.index("Docker") and out.index("PostgreSQL") < out.index("Kubernetes")


def test_a_nice_to_have_section_heading_demotes_everything_under_it():
    jd = "Requirements\n- Python\n- PostgreSQL\n\nNice to have\n- Kubernetes\n- Docker\n\nRequirements\n- Redis"
    out = rank(jd)
    assert out.index("Redis") < out.index("Kubernetes") and out.index("Redis") < out.index("Docker")


def test_inline_nice_to_have_only_demotes_what_follows_it():
    jd = "Requirements: Python, FastAPI, Docker. Nice to have: Kubernetes."
    out = rank(jd)
    assert out[:3] == ["Python", "FastAPI", "Docker"] and out.index("Kubernetes") > out.index("Docker")


def test_is_a_plus_demotes_the_skill_in_that_sentence():
    jd = "Experience with Docker is a plus. You must know Python and PostgreSQL."
    out = rank(jd)
    assert out.index("Python") < out.index("Docker") and out.index("PostgreSQL") < out.index("Docker")


def test_a_skill_in_the_job_title_comes_first():
    out = rank("We need people for many things: Docker, Redis, and some Python.", title="Python Developer")
    assert out[0] == "Python"


def test_must_have_wording_lifts_a_skill_over_an_equal_one():
    jd = "Familiar with Redis. You must have strong PostgreSQL."
    out = rank(jd)
    assert out.index("PostgreSQL") < out.index("Redis")


def test_multi_word_skills_are_found_as_a_phrase():
    out = pos.rank_skills(["Firebase", "React Native", "JavaScript"], "Build React Native apps. React Native is our core. Some Firebase.", None)
    assert out[0] == "React Native"


def test_skills_the_job_never_mentions_keep_their_order_at_the_end_and_nothing_is_added_or_dropped():
    out = rank("Python only.")
    assert out[0] == "Python" and sorted(out) == sorted(SKILLS)
    assert out[1:] == [s for s in SKILLS if s != "Python"]


def test_the_tailored_draft_lists_the_jobs_skills_in_the_jobs_order():
    profile = {"skills": ["Docker, Redis, PostgreSQL, FastAPI, Python, Kubernetes"],
               "experience": ["Dev | Acme, Pune\nJan 2025 – Present\n• Built Python services on PostgreSQL."], "projects": []}
    jd = "Backend Engineer. You will build Python services with FastAPI and PostgreSQL. Python is essential. Nice to have: Docker, Kubernetes."
    draft, _ = rt._build_draft(profile, jd, "", target_title=None)
    assert draft.skills[:3] == ["Python", "FastAPI", "PostgreSQL"]
    assert draft.skills.index("Docker") > draft.skills.index("PostgreSQL")


def test_the_pdf_shows_the_first_skill_of_the_draft_first_in_its_group():
    profile = {"programming_languages": ["JavaScript", "Python", "SQL"], "frameworks": ["Django", "FastAPI"], "databases_tools": ["Docker", "Redis", "PostgreSQL"]}
    draft = {"skills": ["Python", "FastAPI", "PostgreSQL", "Docker"]}
    groups = dict(tr._skills(profile, draft))
    assert groups["Languages"][0] == "Python" and groups["Frameworks"][0] == "FastAPI"
    assert groups["Databases & Tools"][:2] == ["PostgreSQL", "Docker"]
