import pytest

from app.services.resume_service import extract_skills


@pytest.mark.parametrize("text, expected", [
    ("Python, Go, PostgreSQL, Docker", ["Python", "Go", "PostgreSQL", "Docker"]),
    ("Python | Go | PostgreSQL", ["Python", "Go", "PostgreSQL"]),
    ("Python; Go; PostgreSQL", ["Python", "Go", "PostgreSQL"]),
    ("Python\nGo\nPostgreSQL", ["Python", "Go", "PostgreSQL"]),
    ("• Python\n● Go\n▪ PostgreSQL\n- Docker\n* Redis", ["Python", "Go", "PostgreSQL", "Docker", "Redis"]),
    ("  Python ,  Go  ,, \n\n  PostgreSQL  ", ["Python", "Go", "PostgreSQL"]),
    ("Python, Python, Go, Python", ["Python", "Go"]),
])
def test_a_plain_skills_list_without_category_labels_is_read_as_skills(text, expected):
    assert extract_skills(text)["skills"] == expected


def test_a_plain_list_fills_only_the_general_skills_field():
    result = extract_skills("Python, Go")
    assert result["programming_languages"] == [] and result["frameworks"] == [] and result["databases_tools"] == []


@pytest.mark.parametrize("empty", ["", "   ", "\n\n", "•", ", ; |"])
def test_an_empty_skills_section_gives_no_skills(empty):
    assert extract_skills(empty) == {"skills": [], "programming_languages": [], "frameworks": [], "databases_tools": []}


def test_a_categorised_skills_section_is_parsed_exactly_as_before():
    result = extract_skills("Languages: Python, Go\nBackend: FastAPI, Celery\nDatabases: PostgreSQL, Redis")
    assert result["programming_languages"] == ["Python", "Go"]
    assert result["frameworks"] == ["FastAPI", "Celery"]
    assert result["databases_tools"] == ["PostgreSQL", "Redis"]
    assert result["skills"] == ["Python", "Go", "FastAPI", "Celery", "PostgreSQL", "Redis"]


# --- found by uploading a real resume: "AI/ML" (no spaces) crashed the parser with a KeyError -----------

@pytest.mark.parametrize("label", ["AI/ML", "AI / ML", "AI /ML", "ai/ml", "AI/ ML"])
def test_the_ai_ml_label_is_read_whatever_its_spacing(label):
    result = extract_skills(f"Languages: Python, Go\n{label}: LLMs, FAISS, LangChain")
    assert {"LLMs", "FAISS", "LangChain"} <= set(result["databases_tools"])
    assert result["programming_languages"] == ["Python", "Go"]


@pytest.mark.parametrize("front, tools", [("Frontend&Mobile", "Tools&Platforms"), ("Frontend & Mobile", "Tools & Platforms"),
                                          ("Frontend  &  Mobile", "Tools &Platforms")])
def test_ampersand_labels_are_read_whatever_their_spacing(front, tools):
    result = extract_skills(f"{front}: React Native, Next.js\n{tools}: Git, Docker")
    assert result["frameworks"] == ["React Native", "Next.js"]
    assert result["databases_tools"] == ["Git", "Docker"]


def test_every_label_the_parser_recognises_is_mapped_so_none_can_crash():
    labels = ["Languages", "Core CS", "Frontend & Mobile", "Tools & Platforms", "Backend", "Databases", "AI / ML"]
    text = "\n".join(f"{label}: Item{n}" for n, label in enumerate(labels))
    result = extract_skills(text)
    assert all(f"Item{n}" in result["skills"] for n in range(len(labels)))          # nothing dropped, nothing raised


def test_the_users_real_resume_section_parses():
    result = extract_skills(
        "Languages : Python, JavaScript, TypeScript, SQL, C++ | Core CS : OOP, Data Structures & Algorithms, Software Engineering\n"
        "Backend : Node.js, Express.js, Flask | Databases : PostgreSQL, MongoDB, Firebase, Redis\n"
        "Frontend & Mobile : React Native, React.js, Next.js, Zustand, HTML, CSS\n"
        "AI/ML : LLMs, RAG Pipelines, Embeddings, FAISS, LangChain, Prompt Engineering, Gemini API\n"
        "Tools & Platforms : Git, GitHub Actions, Linux, Playwright, Browserbase, Swagger, Docker, Postman")
    assert "LLMs" in result["databases_tools"] and "React Native" in result["frameworks"] and "Python" in result["programming_languages"]
