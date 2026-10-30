from unittest.mock import MagicMock, patch

import pytest

from app.models.company import Company
from app.models.profile import Profile
from app.models.recipient import Recipient
from app.services import outreach_agent as oa
from app.services.outreach_agent import DraftEmail, OutreachAgent, _Draft

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

GOOD = ("Hi Miles,\n\nAt PayLoop I cut query latency 45% by adding a Redis caching layer in front of PostgreSQL, "
        "and I'd love to hear how Cyberdyne Systems thinks about performance. Would you be open to a short conversation?\n\n"
        "Best regards,\nSarah Connor")


def run(body, subject="Performance work and Cyberdyne Systems", profile=PROFILE, recipient=RECIPIENT, llm_error=None):
    llm = MagicMock()
    if llm_error:
        llm.with_structured_output.return_value.invoke.side_effect = llm_error
        llm.invoke.side_effect = llm_error
    else:
        llm.with_structured_output.return_value.invoke.return_value = _Draft(subject=subject, body=body)
    with patch.object(oa, "get_retriever", return_value=None), patch.object(oa, "get_llm", return_value=llm):
        return OutreachAgent().run(profile, COMPANY, recipient)


def used_template(result):
    return result.subject == "Sarah Connor: interested in opportunities at Cyberdyne Systems"


def test_a_supported_email_is_used_as_written():
    result = run(GOOD)
    assert not used_template(result) and "cut query latency 45%" in result.body
    assert "https://github.com/sarahconnor" in result.body                     # links are still added


@pytest.mark.parametrize("bad, why", [
    (GOOD.replace("At PayLoop", "At [Company]"), "placeholder"),
    (GOOD.replace("Would you be open", "I've attached my resume. Would you be open"), "attachment"),
    (GOOD.replace("how Cyberdyne Systems thinks about performance", "your recent funding round"), "company news"),
    (GOOD.replace("Redis caching layer", "TensorFlow pipeline"), "invented technology"),
    (GOOD.replace("45%", "70%"), "invented number"),
    (GOOD.replace("Hi Miles", "Hi John"), "wrong person"),
    (GOOD.replace("short conversation?", "short conversation? See https://sarahs-portfolio.example/cv"), "unknown link"),
    ("Hi Miles,\n\nI admire your team's work and would love to connect about opportunities sometime soon, "
     "since I think we would be a great fit together.\n\nBest regards,\nSarah Connor", "nothing about the candidate"),
    ("Hi Miles,\n\nI build things.\n\nBest regards,\nSarah Connor", "too short"),
    ("Hi Miles,\n\nCut query latency 45% with a Redis caching layer in front of PostgreSQL. Designed REST APIs for a "
     "payments reconciliation service used by the finance team.\n\nBest regards,\nSarah Connor", "pasted bullets"),
])
def test_an_unsupported_email_is_replaced_by_the_grounded_template(bad, why):
    result = run(bad)
    assert used_template(result), why
    assert "TensorFlow" not in result.body and "70%" not in result.body and "[" not in result.body


def test_a_recipient_without_a_name_cannot_be_greeted_by_an_invented_one():
    nameless = Recipient(id="r2", name=None, email="founders@cyberdyne.ai", role=None)
    assert used_template(run(GOOD, recipient=nameless))                            # "Hi Miles" would be made up
    generic = GOOD.replace("Hi Miles", "Hi there")
    assert not used_template(run(generic, recipient=nameless))


def test_the_template_is_built_from_the_real_background_and_passes_the_same_checks():
    draft = oa._template_email(PROFILE, COMPANY, RECIPIENT)
    assert draft.body.startswith("Hi Miles,")
    assert "Most recently I've worked as a Backend Engineer at PayLoop." in draft.body
    assert "I work mainly with Python, Kubernetes, PostgreSQL." in draft.body
    assert "Cyberdyne Systems" in draft.body and draft.body.rstrip().endswith("Sarah Connor")
    oa._verify_email(draft, PROFILE, COMPANY, RECIPIENT, evidence="")              # raises if the template made a claim


def test_the_template_picks_the_bullet_closest_to_the_recipient_and_works_for_a_sparse_profile():
    cto = Recipient(id="r3", name="Ana", email="ana@x.io", role="payments reconciliation lead")
    assert "payments reconciliation" in oa._template_email(PROFILE, COMPANY, cto).body
    sparse = Profile(id="p2", name="John Doe", email="john@example.com", summary="Backend developer.")
    body = oa._template_email(sparse, COMPANY, RECIPIENT).body
    assert "Hi Miles," in body and "John Doe" in body and "Cyberdyne Systems" in body and "None" not in body


def test_when_the_model_fails_entirely_the_template_is_used_instead_of_an_error():
    result = run("", llm_error=RuntimeError("model 'qwen:latest' not found (status code: 404)"))
    assert used_template(result) and "Sarah Connor" in result.body


def test_every_email_the_agent_returns_has_passed_verification_or_is_the_template():
    for body in (GOOD, GOOD.replace("45%", "70%"), "short", GOOD.replace("Hi Miles", "Hi John")):
        result = run(body)
        assert ("45%" in result.body and "70%" not in result.body) or used_template(result)


# --- found by reading real model output --------------------------------------------------------

LIVE_GENERIC = ("Hi Miles, I hope this email finds you well. My name is Sarah Connor and I am a Backend Engineer at PayLoop. "
                "I am excited to bring my skills and experience to your team at Cyberdyne Systems. I would be happy to schedule "
                "a call to discuss how I can contribute to your team's success. Best regards, Sarah Connor")


def test_an_email_that_names_only_the_title_and_employer_is_generic_and_replaced():
    with pytest.raises(ValueError, match="generic: it names no skill or project"):
        oa._verify_email(DraftEmail(subject="Introducing Sarah Connor", body=LIVE_GENERIC), PROFILE, COMPANY, RECIPIENT, evidence="")
    result = run(LIVE_GENERIC)
    assert used_template(result)
    assert "I work mainly with Python, Kubernetes, PostgreSQL." in result.body                  # ...and the template is specific


@pytest.mark.parametrize("phrase", ["Thank you for considering my application.", "I recently applied to the open position.", "I saw the opening on your site."])
def test_claims_about_an_application_or_an_open_role_are_rejected(phrase):
    body = GOOD.replace("Would you be open to a short conversation?", phrase)
    assert used_template(run(body)), phrase


def test_a_domain_the_model_split_with_a_space_is_repaired_and_the_link_is_not_added_twice():
    profile = PROFILE.model_copy(update={"github": "github.com/sarahconnor", "linkedin": "linkedin.com/in/sarahconnor"})
    broken = GOOD.replace("Best regards,\nSarah Connor", "Best regards,\nSarah Connor\nGitHub: github. com/sarahconnor\nLinkedIn: linkedin. com/in/sarahconnor")
    result = run(broken, profile=profile)
    assert not used_template(result)
    assert result.body.count("github.com/sarahconnor") == 1 and result.body.count("linkedin.com/in/sarahconnor") == 1
    assert "github. com" not in result.body and "linkedin. com" not in result.body


def test_a_profile_with_nothing_to_cite_always_gets_the_template():
    sparse = Profile(id="p2", name="John Doe", email="john@example.com", summary="Backend developer.")
    result = run(GOOD.replace("Sarah Connor", "John Doe"), profile=sparse)
    assert result.subject == "John Doe: interested in opportunities at Cyberdyne Systems" and "None" not in result.body


def test_the_template_prefers_languages_and_tools_over_general_concepts_and_reads_cleanly():
    profile = Profile(
        id="p9", name="Vidip Ghosh", email="v@example.com",
        skills=["OOP, Data Structures & Algorithms, Python, JavaScript"], programming_languages=["Python", "JavaScript", "TypeScript"],
        experience=["Intern | Aforro, Bengaluru\n• Built end-to-end Rider app workflows in React Native over REST APIs: pickup, delivery, returns, "
                    "replacements, cancellations, cash collection, and order management for the field team across the country."],
    )
    body = oa._template_email(profile, COMPANY, RECIPIENT).body
    assert "I work mainly with Python, JavaScript, TypeScript." in body and "OOP" not in body
    assert "worked as an Intern at Aforro" in body                                   # article matches the title
    example = next(line for line in body.splitlines() if line.startswith("For example:"))
    assert example == ("For example: Built end-to-end Rider app workflows in React Native over REST APIs: "
                       "pickup, delivery, returns, replacements, cancellations, cash collection.")      # a clean cut, no ellipsis


def test_clause_cut_keeps_short_text_and_never_ends_mid_clause():
    assert oa._clause_cut("Built a thing.", 50) == "Built a thing"
    cut = oa._clause_cut("Built one thing, then another thing, then a third very long thing that goes on", 45)
    assert cut == "Built one thing, then another thing" and "…" not in cut
