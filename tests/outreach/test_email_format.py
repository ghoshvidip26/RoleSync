"""The outreach email format: greeting, intro, expertise, proof of work, (answers), closing, sign-off."""
import re
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app import main
from app.models.company import Company
from app.models.profile import Profile
from app.models.recipient import Recipient
from app.services import outreach_agent as oa
from app.services.outreach_agent import OutreachAgent, _Answer, _Draft

client = TestClient(main.app)

PROFILE = Profile(
    id="p1", name="Sarah Connor", email="sarah@example.com",
    github="https://github.com/sarahconnor", linkedin="https://linkedin.com/in/sarahconnor",
    skills=["Python, Kubernetes, PostgreSQL, Redis"],
    experience=["Backend Engineer | PayLoop, Pune\nJun 2020 – Dec 2021\n"
                "• Cut query latency 45% with a Redis caching layer in front of PostgreSQL.\n"
                "• Designed REST APIs for a payments reconciliation service used by the finance team."],
    projects=["StreamLite | GitHub | Go, Python, Kafka\n• Built a real-time event processing engine with a plugin API."],
)
COMPANY = Company(id="c1", name="Cyberdyne Systems", website="https://cyberdyne.ai")
RECIPIENT = Recipient(id="r1", name="Miles Dyson", email="miles@cyberdyne.ai", role="CTO")
NAMELESS = Recipient(id="r2", name=None, email="team@cyberdyne.ai", role=None)
TAILORED = {
    "headline": "Backend Engineer | Python, Redis", "summary": "Backend engineer with Python and Redis.",
    "skills": ["Redis", "PostgreSQL", "Docker"],
    "experience": [{"company": "PayLoop", "job_title": "Backend Engineer",
                    "bullets": ["Cut query latency 45% with a Redis caching layer in front of PostgreSQL.",
                                "Designed REST APIs for a payments reconciliation service used by the finance team."]}],
    "projects": [{"name": "StreamLite", "technologies": ["Go"], "bullets": ["Built a real-time event processing engine with a plugin API."]}],
}
MIDDLE = ("I'm Sarah, a Backend Engineer who most recently worked at PayLoop in Pune.\n\n"
          "I work with Python, PostgreSQL and Redis, and I built StreamLite, a real-time event processing engine in Go.")


def blocks(body):
    return [b for b in body.split("\n\n")]


def run(model_body=None, *, recipient=RECIPIENT, tailored=None, questions=None, note=None, answers=None, profile=PROFILE,
        subject="Backend engineer intro"):
    """Run the agent with a scripted model: `model_body` for the email, `answers` (a list) for the questions."""
    llm = MagicMock()
    answers = list(answers or [])

    def structured(schema):
        handle = MagicMock()
        if schema is _Answer:
            handle.invoke.side_effect = lambda *_: _Answer(answer=answers.pop(0)) if answers else (_ for _ in ()).throw(RuntimeError("no answer"))
        else:
            handle.invoke.return_value = _Draft(subject=subject, body=model_body or "")
        return handle

    llm.with_structured_output.side_effect = structured
    llm.invoke.side_effect = RuntimeError("model unavailable")
    with patch.object(oa, "get_retriever", return_value=None), patch.object(oa, "get_llm", return_value=llm):
        return OutreachAgent().run(profile, COMPANY, recipient, tailored=tailored, questions=questions, personal_note=note)


# ----------------------------------------------------------------------------- structure

def test_the_email_has_the_agreed_order():
    body = run("").body                                                    # model gave nothing: the grounded template
    order = ["Hi Miles,", "I'm Sarah.", "I work mainly with", oa.PROOF_HEADING, "• ", "look forward to hearing from you",
             oa.CLOSING_REGARDS, "Sarah Connor"]
    positions = [body.index(x) for x in order]
    assert positions == sorted(positions)
    assert body.count("Hi ") == 1 and "Best regards" not in body


def test_the_paragraphs_are_separate_blocks():
    parts = blocks(run("").body)
    assert parts[0] == "Hi Miles,"
    assert parts[1].startswith("I'm Sarah.")                                # 1: short intro
    assert parts[2].startswith("I work mainly with")                        # 2: expertise
    assert parts[3].startswith(oa.PROOF_HEADING)                            #    ...and proof of work
    assert "look forward to hearing from you" in parts[4]                   # 3: closing
    assert parts[5].startswith(oa.CLOSING_REGARDS)


def test_the_greeting_uses_the_first_name_or_a_team_greeting():
    assert run("").body.startswith("Hi Miles,")
    assert run("", recipient=NAMELESS).body.startswith("Hi Cyberdyne Systems Team,")


def test_the_closing_mentions_the_attachment_only_with_a_tailored_resume():
    plain = run("").body
    tailored = run("", tailored=TAILORED).body
    assert "attach" not in plain.lower()
    closing = next(b for b in blocks(tailored) if "look forward" in b)
    assert closing.startswith(oa.RESUME_NOTE)


def test_a_typical_email_is_short():
    words = len(run("", tailored=TAILORED).body.split())
    assert 60 <= words <= 170, words


# ---------------------------------------------------------------------- proof of work

def test_proof_points_are_real_resume_lines_never_invented():
    real = [b.lstrip("• ").strip().rstrip(".") for raw in [*PROFILE.experience, *PROFILE.projects]
            for b in oa.parse_entry(raw).lines]
    for tailored in (None, TAILORED):
        lines = [l[2:] for l in run("", tailored=tailored).body.splitlines() if l.startswith("• ")]
        assert 1 <= len(lines) <= 3
        for line in lines:                                  # "Project: sentence (stack)" or "At Employer: sentence"
            sentence = re.sub(r" using [^()]*$", "", line.split(": ", 1)[1]) if not line.startswith("At ") else line.split(": ", 1)[1]
            assert any(r.startswith(sentence) for r in real), line


def test_proof_comes_from_the_tailored_resume_in_its_order():
    bullets = [l[2:] for l in run("", tailored=TAILORED).body.splitlines() if l.startswith("• ")]
    assert any(b.startswith("At PayLoop: Cut query latency 45%") for b in bullets)        # the approved draft's own role and words
    assert any(b.startswith("StreamLite: Built a real-time event processing engine") and b.endswith("using Go") for b in bullets)
    assert [b.split(":")[0] for b in bullets] == ["StreamLite", "At PayLoop"]               # projects first, then the role


def test_proof_does_not_repeat_what_the_written_paragraphs_already_say():
    said = ("I cut query latency 45% with a Redis caching layer in front of PostgreSQL as a Backend Engineer at PayLoop.\n\n"
            "I work with Python, PostgreSQL and Redis, and I'm keen to learn how Cyberdyne Systems approaches performance.")
    body = run(said).body
    assert "Cut query latency 45%" not in "\n".join(l for l in body.splitlines() if l.startswith("• "))
    assert "I cut query latency 45%" in body


def test_a_long_bullet_is_cut_at_a_clause_not_mid_phrase():
    profile = PROFILE.model_copy(update={"experience": ["Engineer | Acme, Pune\n• " + ", ".join(f"built the thing number {i}" for i in range(30)) + "."]})
    bullet = next(l for l in run("", profile=profile).body.splitlines() if l.startswith("• "))
    assert len(bullet) <= oa.MAX_PROOF_CHARS + 2 and not bullet.endswith((",", ".")) and "…" not in bullet


# ------------------------------------------------------------------ model-written text

def test_a_verified_model_email_supplies_the_two_paragraphs_and_code_the_rest():
    result = run(MIDDLE, subject="Backend work with Redis at PayLoop")
    parts = blocks(result.body)
    assert parts[0] == "Hi Miles," and parts[1].startswith("I'm Sarah, a Backend Engineer who most recently")
    assert parts[2].startswith("I work with Python, PostgreSQL and Redis")
    assert result.subject == "Backend work with Redis at PayLoop"
    assert result.body.count(oa.CLOSING_REGARDS) == 1


def test_a_greeting_or_sign_off_the_model_adds_is_dropped_not_doubled():
    wrapped = "Hi Miles,\n\n" + MIDDLE + "\n\nThanks & Regards,\nSarah Connor"
    body = run(wrapped).body
    assert "I'm Sarah, a Backend Engineer who most recently" in body                  # the model's text was accepted
    assert body.count("Hi Miles,") == 1 and body.count("Sarah Connor") == 1
    assert "Thanks & Regards,\nSarah Connor" in body and body.count(oa.CLOSING_REGARDS) == 1


def test_middle_strips_greetings_and_sign_offs():
    assert oa._middle("Hi Miles,\n\nParagraph one.\n\nBest regards,\nSarah") == "Paragraph one."
    assert oa._middle("Dear Sarah Team,\nText here.\n\nThanks & Regards,\nSarah") == "Text here."
    assert oa._middle("Hello,\nText.\n\nSincerely\nSarah") == "Text."
    assert oa._middle("No greeting or sign-off here.") == "No greeting or sign-off here."
    assert oa._middle("Hi Miles, I'm Sarah and I build things.") == "I'm Sarah and I build things."


def test_an_unverifiable_model_email_falls_back_to_the_template_paragraphs():
    body = run(MIDDLE.replace("Redis", "TensorFlow")).body
    assert "TensorFlow" not in body and "I work mainly with" in body


def test_a_long_model_subject_is_replaced_by_a_short_one():
    result = run("", subject="x" * 5)                                      # template path
    assert result.subject == "Sarah Connor: Backend Engineer interested in Cyberdyne Systems"
    long = run(MIDDLE, subject="A very long subject line about backend work with Redis at PayLoop that nobody would read in an inbox")
    assert long.subject == "Sarah Connor: Backend Engineer interested in Cyberdyne Systems"


def test_links_sit_directly_under_the_signature():
    body = run(MIDDLE).body
    assert body.rstrip().endswith("Sarah Connor\nGitHub: https://github.com/sarahconnor\nLinkedIn: https://linkedin.com/in/sarahconnor")


# -------------------------------------------------------------------- personal note

def test_a_personal_note_is_used_as_written_at_the_start_of_the_intro():
    note = "I've been following Cyberdyne's work on autonomous systems."
    body = run("", note=note).body
    assert blocks(body)[1].startswith(note + " I'm Sarah.")
    assert note not in run("").body


# --------------------------------------------------------------------- questions

RICH = {**TAILORED, "experience": [
    {"company": "PayLoop", "job_title": "Backend Engineer", "bullets": ["Cut query latency 45% with a Redis caching layer in front of PostgreSQL."]},
    {"company": "Acme", "job_title": "Engineer", "bullets": ["Designed REST APIs for a payments reconciliation service used by the finance team."]},
    {"company": "Beta", "job_title": "Engineer", "bullets": ["Wrote a deployment tool in Python that removed manual release steps for four teams."]}]}


def test_a_question_gets_an_answer_and_its_own_proof():
    parts = blocks(run("", tailored=RICH, questions=["Why should you choose me?"]).body)
    i = next(n for n, b in enumerate(parts) if b.startswith("Why should you choose me?"))
    assert parts[i].splitlines()[1].startswith("I'm Sarah, a Backend Engineer with hands-on experience in")
    assert i > next(n for n, b in enumerate(parts) if b.startswith(oa.PROOF_HEADING))      # after the main proof
    assert i < next(n for n, b in enumerate(parts) if "look forward" in b)                  # before the closing
    assert parts[i + 1].startswith(oa.PROOF_HEADING) and "• " in parts[i + 1]               # its own, different bullets


def test_a_sparse_profile_gets_an_answer_without_repeating_proof():
    body = run("", questions=["Why should you choose me?"]).body
    bullets = [l for l in body.splitlines() if l.startswith("• ")]
    assert len(bullets) == len(set(bullets))
    section = body[body.index("Why should you choose me?"):body.index("look forward")]
    assert oa.PROOF_HEADING not in section and "hands-on experience in" in section


def test_each_question_is_answered_separately_in_the_order_asked():
    body = run("", questions=["Why are you a good fit for this role?", "Why should you choose me?"]).body
    assert body.index("Why are you a good fit for this role?") < body.index("Why should you choose me?") < body.index("look forward")


def test_a_verified_model_answer_is_used():
    answer = ("I build and tune backend services in Python with PostgreSQL and Redis, and at PayLoop I cut query latency "
              "45% with a Redis caching layer, so I can help Cyberdyne Systems keep its services fast.")
    body = run("", questions=["Why should you choose me?"], answers=[answer]).body
    assert answer in body


def test_an_ungrounded_answer_is_replaced():
    answer = ("I have shipped TensorFlow models at scale and led teams of forty engineers, so I can transform Cyberdyne "
              "Systems and its entire platform from day one.")
    body = run("", questions=["Why should you choose me?"], answers=[answer]).body
    assert "TensorFlow" not in body and "forty" not in body and "hands-on experience in" in body


def test_no_questions_means_no_extra_sections():
    assert "?" not in run("").body


# ---------------------------------------------------------------------------- API

def outreach(**extra):
    rows = {"profiles": PROFILE.model_dump(), "companies": COMPANY.model_dump(), "recipients": RECIPIENT.model_dump()}
    with patch.object(main.db, "fetch", side_effect=lambda table, id_: rows[table]), \
         patch.object(main.db, "insert", side_effect=lambda t, d: {"id": "e1"}), \
         patch.object(main.db, "update", side_effect=lambda t, i, d: {"id": i, "profile_id": "p1", "company_id": "c1", "recipient_id": "r1", **d}), \
         patch.object(main.agent, "run", return_value=oa.DraftEmail(subject="s", body="b")) as agent_run:
        response = client.post("/outreach", json={"profile_id": "p1", "company_id": "c1", "recipient_id": "r1", **extra})
    return response, agent_run


def test_questions_and_note_reach_the_agent_cleaned_up():
    response, agent_run = outreach(questions=["  Why   should you choose me? ", ""], personal_note="  I like   your work. ")
    assert response.status_code == 201
    assert agent_run.call_args.kwargs["questions"] == ["Why should you choose me?"]
    assert agent_run.call_args.kwargs["personal_note"] == "I like your work."


def test_without_them_the_agent_is_called_exactly_as_before():
    response, agent_run = outreach()
    assert response.status_code == 201
    assert "questions" not in agent_run.call_args.kwargs and "personal_note" not in agent_run.call_args.kwargs


@pytest.mark.parametrize("bad", [{"questions": ["ok question?"] * 5}, {"questions": ["hi"]}, {"questions": ["x" * 201]},
                                 {"personal_note": "x" * 241}])
def test_invalid_questions_or_notes_are_refused(bad):
    response, agent_run = outreach(**bad)
    assert response.status_code == 422 and agent_run.call_count == 0


# ----------------------------------------------------------------- richer skills, no repeated proof

def test_the_expertise_paragraph_names_up_to_six_real_skills_tailored_ones_first():
    profile = PROFILE.model_copy(update={
        "programming_languages": ["Python", "Go", "TypeScript", "SQL"], "frameworks": ["FastAPI", "React"],
        "databases_tools": ["PostgreSQL", "Redis", "Docker"], "skills": ["OOP, Data Structures"]})
    skills = oa._expertise_skills(profile, {"skills": ["Redis", "Go"]})
    assert skills == ["Redis", "Go", "Python", "TypeScript", "SQL", "FastAPI"] and "OOP" not in skills
    assert "I work mainly with Redis, Go, Python, TypeScript, SQL and FastAPI." in oa._template_email(profile, COMPANY, RECIPIENT, {"skills": ["Redis", "Go"]}).body


def test_concepts_are_used_only_when_there_are_too_few_tools():
    assert oa._expertise_skills(PROFILE, None) == ["Python", "Kubernetes", "PostgreSQL", "Redis"]


@pytest.mark.parametrize("items, text", [(["A"], "A"), (["A", "B"], "A and B"), (["A", "B", "C"], "A, B and C"), ([], "")])
def test_join_list(items, text):
    assert oa._join_list(items) == text


def test_an_answers_proof_does_not_repeat_the_main_proof():
    many = {**TAILORED, "experience": [
        {"company": "PayLoop", "job_title": "Backend Engineer", "bullets": ["Cut query latency 45% with a Redis caching layer in front of PostgreSQL."]},
        {"company": "Acme", "job_title": "Engineer", "bullets": ["Designed REST APIs for a payments reconciliation service used by the finance team."]},
        {"company": "Beta", "job_title": "Engineer", "bullets": ["Wrote a deployment tool in Python that removed manual release steps for four teams."]}]}
    body = run("", tailored=many, questions=["Why should you choose me?"]).body
    bullets = [l for l in body.splitlines() if l.startswith("• ")]
    assert len(bullets) == len(set(bullets)) and len(bullets) >= 3                  # no bullet is shown twice


# ------------------------------------------------------------------- editing before approval

class _Rows:
    def __init__(self, status="pending_review"):
        self.row = {"id": "e1", "profile_id": "p1", "company_id": "c1", "recipient_id": "r1", "subject": "Old subject",
                    "body": "Old body", "status": status}
        self.updates = []

    def fetch(self, table, id_):
        return self.row if id_ == "e1" else None

    def update(self, table, id_, data):
        self.updates.append(data)
        self.row.update(data)
        return self.row


def patch_email(payload, status="pending_review", email_id="e1"):
    rows = _Rows(status)
    with patch.object(main.db, "fetch", rows.fetch), patch.object(main.db, "update", rows.update):
        return client.patch(f"/emails/{email_id}", json=payload), rows


def test_a_reviewer_can_change_the_subject_and_body():
    r, rows = patch_email({"subject": "  New   subject ", "body": "Line one\r\n\r\nLine two  "})
    assert r.status_code == 200
    assert r.json()["subject"] == "New subject" and r.json()["body"] == "Line one\n\nLine two"
    assert rows.updates == [{"subject": "New subject", "body": "Line one\n\nLine two"}]


def test_only_the_fields_sent_change():
    r, rows = patch_email({"body": "Only the body"})
    assert r.status_code == 200 and rows.updates == [{"body": "Only the body"}] and r.json()["subject"] == "Old subject"


@pytest.mark.parametrize("payload", [{}, {"subject": None}, {"subject": None, "body": None}])
def test_nothing_to_change_is_a_422(payload):
    r, rows = patch_email(payload)
    assert r.status_code == 422 and rows.updates == []


@pytest.mark.parametrize("payload", [{"subject": "   "}, {"body": ""}, {"body": " \n "}, {"subject": "x" * 151}, {"body": "x" * 8001}])
def test_blank_or_oversized_fields_are_refused(payload):
    r, rows = patch_email(payload)
    assert r.status_code == 422 and rows.updates == []


@pytest.mark.parametrize("status", ["sent", "rejected", "failed", "generating", "approved"])
def test_an_email_that_is_no_longer_pending_cannot_be_edited(status):
    r, rows = patch_email({"body": "Too late"}, status=status)
    assert r.status_code == 409 and rows.updates == []


def test_editing_an_unknown_email_is_a_404():
    r, _ = patch_email({"body": "x"}, email_id="nope")
    assert r.status_code == 404


# ------------------------------------------------ found in a real sent e-mail (Hi Hi, / SWE-1/ I'm / open parenthesis)

def _recipient(name):
    return Recipient(id="rx", name=name, email="hr@gnani.ai", role=None)


@pytest.mark.parametrize("name, expected", [
    ("Miles Dyson", "Hi Miles,"), ("miles", "Hi Miles,"), ("MILES", "Hi Miles,"), ("  Ana  Maria ", "Hi Ana,"),
    ("Jean-Luc Picard", "Hi Jean-Luc,"), ("O'Brien", "Hi O'Brien,"), ("McDonald", "Hi McDonald,"),
    ("Hi Miles", "Hi Miles,"), ("Hello, Miles Dyson", "Hi Miles,"), ("Dear Miles", "Hi Miles,"),
    ("Hi", "Hi Gnani Team,"), ("hello", "Hi Gnani Team,"), ("Hi there", "Hi Gnani Team,"), ("Team", "Hi Gnani Team,"),
    ("hr@gnani.ai", "Hi Gnani Team,"), ("12345", "Hi Gnani Team,"), ("???", "Hi Gnani Team,"), ("", "Hi Gnani Team,"), (None, "Hi Gnani Team,"),
])
def test_the_greeting_never_uses_something_that_is_not_a_name(name, expected):
    assert oa._greeting(_recipient(name), Company(id="c", name="Gnani")) == expected


@pytest.mark.parametrize("raw, tidy", [
    ("I'm looking for SWE-1/", "I'm looking for SWE-1."), ("looking for a backend role", "Looking for a backend role."),
    ("I like your work!", "I like your work!"), ("Why this company?", "Why this company?"), ("  spaced   out  ,", "Spaced out."),
    ("ends with dash -", "Ends with dash."), ("", ""), ("   ", ""), ("//", ""),
])
def test_the_personal_line_is_tidied_but_otherwise_left_as_written(raw, tidy):
    assert oa._tidy_note(raw) == tidy


def test_the_personal_line_does_not_run_into_the_intro():
    body = run("", note="I'm looking for SWE-1/").body
    assert "I'm looking for SWE-1. I'm Sarah" in body and "SWE-1/" not in body


def test_a_cut_never_leaves_a_parenthesis_open():
    bullet = ("Delivered 6 rider order flows (pickup, delivery, returns, replacements, cancellations, cash collection) via REST APIs, "
              "with real-time Firebase FCM/WebSocket alerts for 4 assignment flows (delivery, return, replacement, assistant-rider)")
    for limit in range(30, len(bullet) + 1, 7):
        cut = oa._clause_cut(bullet, limit)
        assert cut.count("(") == cut.count(")"), (limit, cut)
        assert not cut.endswith((",", ";", "(", " "))
    assert oa._clause_cut(bullet, 100) == "Delivered 6 rider order flows"                  # the parenthetical is dropped, not half shown
    assert oa._clause_cut(bullet, 190).endswith("via REST APIs")                              # a comma outside the brackets is used


def test_cuts_outside_brackets_are_unchanged():
    assert oa._clause_cut("Built a thing.", 50) == "Built a thing"
    assert oa._clause_cut("Built one thing, then another thing, then a third very long thing that goes on", 45) == "Built one thing, then another thing"


def test_a_job_that_is_still_going_on_is_introduced_as_current():
    now = PROFILE.model_copy(update={"experience": ["Backend Engineer | PayLoop, Pune\nJun 2020 – Present\n• Cut latency 45%."]})
    assert "I'm Sarah, currently a Backend Engineer at PayLoop." in run("", profile=now).body
    for ongoing in ("Jun 2020 - present", "2020 – Current", "since 2020 (now)"):
        profile = PROFILE.model_copy(update={"experience": [f"Backend Engineer | PayLoop, Pune\n{ongoing}\n• Cut latency 45%."]})
        assert oa._current_role_with_dates(profile)[2] is True or "since" in ongoing        # "since 2020 (now)" is not a date line
    assert "I'm Sarah. Most recently I've worked as a Backend Engineer at PayLoop." in run("").body   # ended: as before


def test_a_recipient_named_hi_gets_a_team_greeting_in_the_whole_email():
    body = run("", recipient=_recipient("Hi")).body
    assert body.startswith("Hi Cyberdyne Systems Team,") and "Hi Hi" not in body


# ------------------------------------------------ proof chosen for the company; second paragraph mixes projects, stack, role

MIXED = Profile(
    id="p9", name="Vidip Ghosh", email="v@example.com", summary="Software engineer with 1 year of experience shipping mobile, backend, and AI features in TypeScript, Python, and React Native. Strongest in payments.",
    programming_languages=["TypeScript", "Python"], frameworks=["React Native", "Node.js"], databases_tools=["PostgreSQL", "FAISS"],
    experience=["Junior Mobile Application Developer | Aforro, Bengaluru\nJul 2026 – Present\n"
                "• Delivered 6 rider order flows (pickup, delivery, returns, replacements, cancellations, cash collection) via REST APIs.\n"
                "• Designed a custom Razorpay SDK payment flow for UPI, cards, wallets, and saved payment methods.",
                "SDE Intern | TechKareer, Bengaluru\nJan 2026 – Mar 2026\n• Connected 3 external APIs to deliver real-time market data."],
    projects=["StackSearch | Node.js, TypeScript, Python, LangChain, React.js, MongoDB, FAISS\nJan 2026 – Present\n"
              "• Achieved sub-100 ms retrieval over 2,500+ documentation pages by building a RAG-based semantic search engine with embeddings.\n",
              "AegisSRE | TypeScript, Next.js, Mastra AI, Qdrant, Enkrypt AI, Node.js\nJul 2026 – Present\n"
              "• Automated incident triage with a 5-agent workflow (Coordinator, Diagnosis, Planning, Execution, Verification).",
              "PayTrack | React Native, Razorpay, Firebase\n2025\n• Built a wallet and UPI payments tracker used by 300 people."],
)


def proof(company, role=None, note=None, tailored=None, **kw):
    return oa._proof_points(MIXED, company, Recipient(id="r", name=None, email="a@b.co", role=role), tailored, note=note, **kw)


def test_an_ai_company_gets_the_ai_projects_first_with_their_tech_stacks():
    lines = proof(Company(id="c", name="gnani.ai", website="https://gnani.ai"))
    assert [l.split(":")[0] for l in lines] == ["StackSearch", "AegisSRE", "At Aforro"][:len(lines)] or {"StackSearch", "AegisSRE"} <= {l.split(":")[0] for l in lines}
    stack = next(l for l in lines if l.startswith("StackSearch"))
    assert stack.endswith("using Node.js, TypeScript, Python, LangChain and React.js")        # the project's first five technologies
    assert "sub-100 ms retrieval over 2,500+ documentation pages" in stack
    assert not any(l.startswith("PayTrack") for l in lines)


def test_a_payments_company_gets_the_payments_work():
    lines = proof(Company(id="c", name="PayLoop", website="https://payloop.io"), role="payments engineer")
    names = [l.split(":")[0] for l in lines]
    assert "PayTrack" in names and any(n == "At Aforro" for n in names)
    assert "Razorpay" in " ".join(lines) or "wallet" in " ".join(lines)
    assert names.index("PayTrack") < names.index("At Aforro")                                  # projects first, then the role
    assert "StackSearch" not in names[:1]


def test_the_personal_line_steers_the_choice_too():
    plain = proof(Company(id="c", name="Acme"))
    steered = proof(Company(id="c", name="Acme"), note="I love building payments products with UPI.")
    assert steered[0].startswith("PayTrack") and not plain[0].startswith("PayTrack") or steered != plain


def test_projects_come_first_and_the_default_is_two_projects_and_one_role():
    lines = proof(Company(id="c", name="gnani.ai"))
    kinds = ["role" if l.startswith("At ") else "project" for l in lines]
    assert kinds == ["project", "project", "role"] and len(lines) == 3


def test_limit_two_gives_the_best_project_and_the_best_role():
    lines = proof(Company(id="c", name="gnani.ai"), limit=2)
    assert [l.startswith("At ") for l in lines] == [False, True]


def test_every_line_is_bounded_complete_and_never_leaves_a_bracket_open():
    for company in ("gnani.ai", "PayLoop", "Acme"):
        for line in proof(Company(id="c", name=company)):
            assert len(line) <= oa.MAX_PROOF_CHARS + 6 and line.count("(") == line.count(")") and not line.endswith((",", ";", ":"))
    aforro = next(l for l in proof(Company(id="c", name="PayLoop"), role="payments") if l.startswith("At Aforro"))
    assert "(pickup" not in aforro or aforro.count("(") == aforro.count(")")


def test_the_approved_tailored_resume_supplies_the_words_when_it_has_projects_and_roles():
    tailored = {"headline": "Backend Engineer | Python", "skills": ["Python"],
                "experience": [{"company": "Aforro", "job_title": "Dev", "bullets": ["Cut checkout errors by half with retries."]}],
                "projects": [{"name": "AegisSRE", "technologies": ["TypeScript", "Qdrant"], "bullets": ["Automated incident triage with agents."]}]}
    lines = proof(Company(id="c", name="gnani.ai"), tailored=tailored)
    assert lines == ["AegisSRE: Automated incident triage with agents using TypeScript and Qdrant", "At Aforro: Cut checkout errors by half with retries"]


def test_nothing_is_shown_twice_in_one_email_when_a_question_is_answered():
    body = run("", profile=MIXED, questions=["Why should you choose me?"]).body
    bullets = [l for l in body.splitlines() if l.startswith("• ")]
    assert len(bullets) == len(set(bullets)) and len(bullets) >= 3


def test_a_profile_without_projects_or_roles_has_no_proof_and_no_crash():
    bare = Profile(id="p0", name="Ann Lee", email="a@b.co", skills=["Go"])
    assert oa._proof_points(bare, Company(id="c", name="X"), RECIPIENT, None) == []


@pytest.mark.parametrize("summary, sentence", [
    ("Software engineer with 1 year of experience shipping mobile, backend, and AI features in TypeScript. Strongest in payments.",
     "I have 1 year of experience shipping mobile, backend, and AI features in TypeScript."),
    ("An experienced backend developer with five years of building payment systems at scale.", "I have five years of building payment systems at scale."),
    ("Backend developer.", ""), ("I build things with care and attention to detail every day.", ""), ("", ""), (None, ""),
    ("Engineer with a bit.", ""),
])
def test_the_resume_summary_becomes_a_first_person_sentence_only_when_it_has_that_shape(summary, sentence):
    assert oa._summary_sentence(MIXED.model_copy(update={"summary": summary})) == sentence


def test_the_intro_reuses_the_resume_summary():
    paragraphs = oa._template_paragraphs(MIXED, None)
    assert paragraphs[0] == ("I'm Vidip, currently a Junior Mobile Application Developer at Aforro. "
                             "I have 1 year of experience shipping mobile, backend, and AI features in TypeScript, Python, and React Native.")


def test_company_terms_include_the_domain_and_expand_known_fields():
    terms = oa._wanted_terms(Company(id="c", name="Gnani", website="https://www.gnani.ai"), RECIPIENT, None, None)
    assert {"gnani", "ai", "llm", "rag", "agent"} <= terms and "https" not in terms and "www" not in terms
    assert "razorpay" in oa._wanted_terms(Company(id="c", name="PayCo"), Recipient(id="r", name=None, email="a@b.co", role="fintech lead"), None, None)


def test_a_long_sentence_is_cut_before_a_clause_word_not_in_the_middle_of_a_clause():
    aegis = ("Architected an autonomous AI incident response platform using a multi-agent workflow (Coordinator, Diagnosis, Planning, "
             "Execution, Verification) to automate incident triage and remediation")
    assert oa._clause_cut(aegis, 150) == "Architected an autonomous AI incident response platform using a multi-agent workflow (Coordinator, Diagnosis, Planning, Execution, Verification)"
    cut = oa._clause_cut("Built a payments service that handles refunds and settlements for forty merchants across three countries", 70)
    assert cut == "Built a payments service that handles refunds and settlements"         # before " for ", a whole clause dropped
    assert oa._clause_cut("Short one", 50) == "Short one"
