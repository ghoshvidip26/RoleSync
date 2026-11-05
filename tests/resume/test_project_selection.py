"""Projects are chosen for the job: an AI job shows the LLM/RAG/agent projects, a full-stack job the full-stack ones (AI work too)."""
from app.agents import resume_tailor as rt

PROFILE = {
    "skills": ["Python, TypeScript, React, Next.js, Node.js, PostgreSQL, Redis, LangChain, FAISS, LLMs, RAG, Docker"],
    "experience": ["Developer | Acme, Pune\nJan 2025 – Present\n• Built Python services."],
    "projects": [
        "StackSearch | GitHub | Python, LangChain, FAISS\n• Built a RAG pipeline with embeddings and a vector index over 2,500 documentation pages.\n"
        "• Added LLM answers with source citations.",
        "AegisSRE | GitHub | TypeScript, Mastra AI\n• Built a 5-agent workflow (Coordinator, Diagnosis, Planning, Execution, Verification) for incident remediation.",
        "ShopFront | GitHub | Next.js, React, Tailwind CSS\n• Built a responsive storefront UI with accessible React components.",
        "LedgerAPI | GitHub | Node.js, PostgreSQL, Redis\n• Built a REST API with database migrations and Redis caching for a ledger service.",
        "SnakeGame | GitHub | Pygame\n• Built a small arcade game.",
    ],
}
AI_JD = ("AI Engineer. You will build LLM applications, RAG pipelines and multi-agent orchestration. Python, LangChain and vector search "
         "with FAISS are required.")
FULLSTACK_JD = ("Full-Stack Developer. Build React and Next.js interfaces and Node.js REST APIs on PostgreSQL. You will also ship AI features "
                "such as RAG and LLM-powered search.")
BACKEND_JD = "Backend Engineer. Design REST APIs and PostgreSQL schemas in Node.js, add Redis caching, deploy with Docker."
FRONTEND_JD = "Frontend Developer. Build responsive UIs with React, Next.js and Tailwind CSS."


def names(jd, profile=PROFILE, title=None):
    draft, _ = rt._build_draft(profile, jd, "", target_title=title)
    return [p.name for p in draft.projects], draft


def test_an_ai_job_shows_the_llm_rag_and_agent_projects_only():
    shown, _ = names(AI_JD)
    assert set(shown) == {"StackSearch", "AegisSRE"}
    assert shown[0] == "StackSearch"                                  # the one with the most of what the job asks for first


def test_a_full_stack_job_shows_full_stack_projects_and_the_ai_ones_too():
    shown, _ = names(FULLSTACK_JD)
    assert {"ShopFront", "LedgerAPI", "StackSearch"} <= set(shown)
    assert "SnakeGame" not in shown


def test_a_backend_job_shows_backend_projects_not_frontend_or_unrelated_ones():
    shown, _ = names(BACKEND_JD)
    assert shown[0] == "LedgerAPI"
    assert "ShopFront" not in shown and "SnakeGame" not in shown


def test_a_frontend_job_shows_the_frontend_project_first():
    shown, _ = names(FRONTEND_JD)
    assert shown[0] == "ShopFront" and "SnakeGame" not in shown and "LedgerAPI" not in shown


def test_the_same_resume_gives_a_different_project_list_for_each_job():
    lists = {tuple(names(jd)[0]) for jd in (AI_JD, FULLSTACK_JD, BACKEND_JD, FRONTEND_JD)}
    assert len(lists) == 4


def test_what_is_left_out_is_named_and_stays_in_the_profile():
    shown, draft = names(AI_JD)
    note = next(r for r in draft.recommendations if r.startswith("Left out of this resume"))
    for left_out in ("ShopFront", "LedgerAPI", "SnakeGame"):
        assert left_out in note
    assert "still in your profile" in note and "StackSearch" not in note
    assert len(PROFILE["projects"]) == 5                                # the profile itself is untouched


def test_a_project_keeps_all_its_own_content_when_shown():
    _, draft = names(AI_JD)
    stack = next(p for p in draft.projects if p.name == "StackSearch")
    assert len(stack.bullets) == 2 and stack.technologies == ["Python", "LangChain", "FAISS"]


def test_a_job_that_matches_no_project_leaves_the_projects_as_they_were():
    shown, draft = names("Pastry chef. Bake bread, laminate dough, manage the pastry section.")
    assert sorted(shown) == sorted(["StackSearch", "AegisSRE", "ShopFront", "LedgerAPI", "SnakeGame"])
    assert not any(r.startswith("Left out of this resume") for r in draft.recommendations)
