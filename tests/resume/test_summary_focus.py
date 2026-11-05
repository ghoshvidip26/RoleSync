"""The summary is written for the job: a backend job, a frontend job and a full-stack job give one candidate three different
summaries, each built only from what the resume really contains. A focus the resume does not evidence is never claimed."""
import pytest

from app.agents import positioning as pos
from app.agents import resume_tailor as rt

PROFILE = {
    "summary": "Software engineer with 1 year of experience shipping mobile, backend and AI features.",
    "skills": ["React, Next.js, TypeScript, Tailwind CSS, Node.js, PostgreSQL, Redis, REST APIs, Docker, Python"],
    "experience": [
        "Software Developer | Acme, Pune\nJan 2025 – Present\n"
        "• Built responsive React dashboards with Tailwind CSS and reusable UI components used by 3 teams.\n"
        "• Designed REST APIs in Node.js and PostgreSQL with Redis caching, cutting response time 40%.\n"
        "• Shipped authentication and background queue workers for the backend payments service.",
        "Frontend Intern | Beta Labs, Remote\nJun 2024 – Dec 2024\n• Implemented accessible Next.js pages and a Figma-based design system.",
    ],
    "projects": [
        "ShopFront | GitHub | Next.js, TypeScript, Tailwind CSS\n• Built a responsive storefront UI with accessible React components.",
        "LedgerAPI | GitHub | Node.js, PostgreSQL, Redis\n• Built a REST API with database migrations and Redis caching for a ledger service.",
    ],
}
BACKEND_JD = ("Backend Engineer. We build payment services. You will design REST APIs, microservices and PostgreSQL schemas, "
              "add Redis caching and queues, and own authentication in Node.js.")
FRONTEND_JD = ("Frontend Developer. You will build responsive UIs with React, Next.js and Tailwind CSS, create accessible "
               "components and work with designers in Figma.")
FULLSTACK_JD = ("Full-Stack Developer. Build React and Next.js interfaces and the Node.js REST APIs and PostgreSQL databases behind them. "
                "You will work on the frontend and the backend.")


def summary(jd, profile=PROFILE, title=None, variant=0):
    draft, _ = rt._build_draft(profile, jd, "", target_title=title, variant=variant)
    return draft.summary


def hits(text):
    return {family: len(pos._focus_counts(text)[family]) if isinstance(pos._focus_counts(text)[family], set) else pos._focus_counts(text)[family]
            for family in ("backend", "frontend")}


# ------------------------------------------------------------------ what kind of job it is

@pytest.mark.parametrize("jd, title, expected", [
    (BACKEND_JD, None, "backend"),
    (FRONTEND_JD, None, "frontend"),
    (FULLSTACK_JD, None, "fullstack"),
    ("Build React UIs and Node.js REST APIs on PostgreSQL. Backend and frontend work, components and database schemas.", None, "fullstack"),
    ("React Native mobile app for Android and iOS. Mobile releases.", None, "mobile"),
    ("We are a friendly team who value communication.", None, None),
    ("Something generic", "Full Stack Engineer", "fullstack"),
    ("Something generic", "Backend Engineer", "backend"),
])
def test_the_focus_comes_from_what_the_job_asks_for(jd, title, expected):
    assert pos.job_focus(jd, title) == expected


def test_a_focus_is_used_only_when_the_resume_evidences_it():
    backend_only = "Python, Django, PostgreSQL, REST APIs, Redis, microservices"
    assert pos.supported_focus("backend", backend_only) == "backend"
    assert pos.supported_focus("frontend", backend_only) is None
    assert pos.supported_focus("fullstack", backend_only) is None                       # needs both sides
    assert pos.supported_focus("fullstack", backend_only + " React, CSS, HTML") == "fullstack"
    assert pos.supported_focus(None, backend_only) is None


# ------------------------------------------------------------------ the three summaries

def test_the_same_resume_gives_a_different_summary_for_each_kind_of_job():
    back, front, full = summary(BACKEND_JD), summary(FRONTEND_JD), summary(FULLSTACK_JD)
    assert len({back, front, full}) == 3
    print("\nBACKEND :", back, "\nFRONTEND:", front, "\nFULLSTACK:", full)


def test_a_backend_summary_is_about_backend_work_only():
    text = summary(BACKEND_JD)
    assert text.startswith("Backend")
    for term in ("Node.js", "PostgreSQL", "Redis", "REST"):
        assert term in text
    for frontend_only in ("React", "Tailwind", "Next.js", "Figma", "UI", "responsive"):
        assert frontend_only not in text, frontend_only


def test_a_frontend_summary_is_about_frontend_work_only():
    text = summary(FRONTEND_JD)
    assert text.startswith("Frontend")
    assert "React" in text
    for backend_only in ("PostgreSQL", "Redis", "REST", "queue", "microservice"):
        assert backend_only not in text, backend_only


def test_a_full_stack_summary_shows_both_sides():
    text = summary(FULLSTACK_JD)
    assert text.startswith("Full-stack")
    assert "front end" in text and "back end" in text
    assert "React" in text and ("Node.js" in text or "PostgreSQL" in text)
    counts = pos._focus_counts(text)
    assert counts["backend"] >= 1 and counts["frontend"] >= 1


def test_every_variant_stays_on_the_jobs_focus():
    for jd, other in ((BACKEND_JD, "frontend"), (FRONTEND_JD, "backend")):
        texts = [summary(jd, variant=v) for v in range(3)]
        assert len(set(texts)) >= 2
        for text in texts:
            assert pos._focus_counts(text)["backend" if other == "frontend" else "frontend"] >= pos._focus_counts(text)[other]


def test_changing_the_job_description_changes_the_summary():
    assert summary(BACKEND_JD) != summary(FRONTEND_JD.replace("Frontend Developer", "Backend Engineer").replace("React", "Node.js"))


def test_everything_in_a_focused_summary_comes_from_the_resume():
    resume_words = {w.strip(".,()%").lower() for w in " ".join([PROFILE["skills"][0], *PROFILE["experience"], *PROFILE["projects"]]).split()}
    for jd in (BACKEND_JD, FRONTEND_JD, FULLSTACK_JD):
        for word in summary(jd).split():
            w = word.strip(".,()%").lower()
            assert w in resume_words or w in {"backend", "frontend", "full-stack", "developer", "engineer", "with", "hands-on", "experience", "in", "and",
                                              "on", "the", "end", "front", "back", "working", "across", "stack", "also", "works", "building", "year", "of", "1", "a"}, w


def test_a_job_the_resume_cannot_support_gets_a_general_summary_not_an_invented_focus():
    backend_only = {**PROFILE, "skills": ["Python, Django, PostgreSQL, REST APIs"],
                    "experience": ["Backend Developer | Acme, Pune\nJan 2025 – Present\n• Built Django REST APIs on PostgreSQL."], "projects": []}
    text = summary(FRONTEND_JD, profile=backend_only)
    assert not text.startswith("Frontend") and "front end" not in text


# ------------------------------------------------------------------ the model path is held to the same focus

def test_model_text_that_ignores_the_jobs_focus_is_rejected():
    with pytest.raises(ValueError):
        pos.check_focus("Developer who ships features quickly and works well with others.", "backend")
    with pytest.raises(ValueError):
        pos.check_focus("Frontend developer building React UIs, also strong in PostgreSQL, Redis and REST APIs.", "frontend")
    pos.check_focus("Backend developer building REST APIs on PostgreSQL with Redis caching.", "backend")
    with pytest.raises(ValueError):
        pos.check_focus("Backend developer building REST APIs on PostgreSQL.", "fullstack")
    pos.check_focus("Builds React UIs and Node.js REST APIs across the stack.", "fullstack")


# ------------------------------------------------------------------ the summary reads like a summary

OWNED = "Owned the Sales app frontend end-to-end across 10 feature areas."


def test_a_duty_phrase_with_a_product_name_is_not_quoted_in_the_summary():
    profile = {**PROFILE, "experience": [
        f"Software Developer | Acme, Pune\nJan 2025 – Present\n• {OWNED}\n• Built responsive React dashboards with Tailwind CSS used by 3 teams."]}
    text = summary(FRONTEND_JD, profile=profile)
    assert "Owned" not in text and "Sales app" not in text
    assert "responsive React dashboards" in text                  # a proper line is still used


def test_summary_worthy_lines():
    known = {"react", "nodej", "postgresql"}
    assert not pos.summary_worthy(OWNED, known)                                          # opens with a duty phrase, names "Sales"
    assert not pos.summary_worthy("Worked on the Checkout flow for the Rider app.", known)
    assert not pos.summary_worthy("Delivered the Sales app frontend for 10 feature areas.", known)      # an internal product name
    assert pos.summary_worthy("Delivered 6 order flows via REST APIs with COD collection.", known)      # acronyms are fine
    assert pos.summary_worthy("Cut response time 40% with Redis caching in Node.js.", known | {"redi"})
    assert pos.summary_worthy("Built a RAG pipeline over 2,500 pages (StackSearch).", known, allow_names=True)   # project lines name the project
    assert not pos.summary_worthy("Owned the whole thing (StackSearch).", known, allow_names=True)


def test_regenerate_still_gives_different_summaries_when_there_is_little_to_quote():
    bare = {**PROFILE, "experience": [f"Software Developer | Acme, Pune\nJan 2025 – Present\n• {OWNED}"], "projects": []}
    texts = {summary(FRONTEND_JD, profile=bare, variant=v) for v in range(3)}
    assert len(texts) >= 2
    assert all("Owned" not in t for t in texts)
