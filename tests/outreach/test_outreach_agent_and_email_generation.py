from unittest.mock import MagicMock, patch
import pytest

from app.models.company import Company
from app.models.email import Email, EmailStatus
from app.models.profile import Profile
from app.models.recipient import Recipient
from app.services import email_service
from app.services.outreach_agent import DraftEmail, OutreachAgent, _Draft
import app.main as main


@pytest.fixture
def sample_profile():
    return Profile(
        id="prof-1",
        name="Sarah Connor",
        email="sarah@example.com",
        github="https://github.com/sarahconnor",
        linkedin="https://linkedin.com/in/sarahconnor",
        portfolio="https://sarahconnor.dev",
        summary="Distributed systems engineer with 5 years experience.",
        skills=["Python", "Distributed Systems", "Kubernetes"],
    )


@pytest.fixture
def sample_company():
    return Company(
        id="comp-1",
        name="Cyberdyne Systems",
        website="https://cyberdyne.ai",
    )


@pytest.fixture
def sample_recipient():
    return Recipient(
        id="recip-1",
        name="Miles Dyson",
        email="miles@cyberdyne.ai",
        role="Director of Systems Engineering",
    )


def test_outreach_agent_draft_generation_with_links(sample_profile, sample_company, sample_recipient):
    agent = OutreachAgent()

    mock_draft = _Draft(
        subject="Distributed Systems at Cyberdyne",
        body=(
            "Hi Miles,\n\n"
            "I build distributed systems in Python and run them on Kubernetes, and I'd love to learn how the "
            "Director of Systems Engineering role approaches reliability at Cyberdyne Systems. "
            "Would you be open to a short conversation?\n\n"
            "Best regards,\nSarah Connor\n"
            "GitHub: https://github.com/sarahconnor\n"
            "Portfolio: https://sarahconnor.dev\n"
            "LinkedIn: https://linkedin.com/in/sarahconnor"
        ),
    )

    with patch("app.services.outreach_agent.get_retriever", return_value=None), \
         patch("app.services.outreach_agent.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = mock_draft
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        result = agent.run(sample_profile, sample_company, sample_recipient)

        assert isinstance(result, DraftEmail)
        assert result.subject == "Distributed Systems at Cyberdyne"
        assert "Sarah Connor" in result.body
        assert "https://github.com/sarahconnor" in result.body
        assert "https://sarahconnor.dev" in result.body
        assert "https://linkedin.com/in/sarahconnor" in result.body


def test_outreach_agent_missing_optional_links(sample_company, sample_recipient):
    profile_without_links = Profile(
        id="prof-2",
        name="John Doe",
        email="john@example.com",
        github=None,
        portfolio=None,
        linkedin=None,
        summary="Backend developer.",
        skills=["Python", "SQL"],
    )

    agent = OutreachAgent()
    mock_draft = _Draft(
        subject="Engineering at Cyberdyne",
        body=(
            "Hi Miles,\n\nI'm a backend developer who works mainly with Python and SQL, and I'd like to learn more "
            "about the systems engineering work at Cyberdyne Systems. Would you be open to a short conversation "
            "about where I could contribute?\n\nBest regards,\nJohn Doe"
        ),
    )

    with patch("app.services.outreach_agent.get_retriever", return_value=None), \
         patch("app.services.outreach_agent.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = mock_draft
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        result = agent.run(profile_without_links, sample_company, sample_recipient)

        assert result.subject == "Engineering at Cyberdyne"
        assert "John Doe" in result.body
        assert "None" not in result.body
        assert "[Link]" not in result.body
        assert "GitHub:" not in result.body


def test_outreach_agent_fallback_parsing(sample_profile, sample_company, sample_recipient):
    agent = OutreachAgent()

    with patch("app.services.outreach_agent.get_retriever", return_value=None), \
         patch("app.services.outreach_agent.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = Exception("Structured output failed")
        mock_llm.with_structured_output.return_value = mock_structured

        mock_fallback_resp = MagicMock()
        mock_fallback_resp.content = (
            "Subject: Opportunities at Cyberdyne Systems\n\n"
            "Body:\n"
            "Hi Miles,\n\n"
            "I build distributed systems in Python and run them on Kubernetes, and I'd like to talk about "
            "systems engineering at Cyberdyne Systems. Would you be open to a short chat?\n\n"
            "Best regards,\nSarah Connor"
        )
        mock_llm.invoke.return_value = mock_fallback_resp
        mock_get_llm.return_value = mock_llm

        result = agent.run(sample_profile, sample_company, sample_recipient)

        assert result.subject == "Opportunities at Cyberdyne Systems"
        assert "Hi Miles" in result.body
        assert "Sarah Connor" in result.body
        assert "https://github.com/sarahconnor" in result.body


def test_generation_never_sends_email(sample_profile, sample_company, sample_recipient):
    """Verify that generating an outreach email transitions status to PENDING_REVIEW and NEVER invokes send_email."""
    mock_draft = DraftEmail(subject="Test Subject", body="Test Body")

    with patch("app.main.agent.run", return_value=mock_draft), \
         patch("app.services.email_service.send_email") as mock_send, \
         patch("app.db.models.fetch") as mock_fetch, \
         patch("app.db.models.insert") as mock_insert, \
         patch("app.db.models.update") as mock_update:

        def mock_fetch_impl(table, id_):
            if table == "profiles":
                return sample_profile.model_dump()
            if table == "companies":
                return sample_company.model_dump()
            if table == "recipients":
                return sample_recipient.model_dump()
            return None

        mock_fetch.side_effect = mock_fetch_impl
        mock_insert.return_value = {"id": "email-gen-1"}
        mock_update.side_effect = lambda table, id_, data: {"id": id_, **data, "profile_id": sample_profile.id, "company_id": sample_company.id, "recipient_id": sample_recipient.id}

        req = main.OutreachRequest(
            profile_id=sample_profile.id,
            company_id=sample_company.id,
            recipient_id=sample_recipient.id,
        )
        result = main.create_outreach(req)

        # Status must be pending review
        assert result.status == EmailStatus.PENDING_REVIEW
        assert result.subject == "Test Subject"
        assert result.body == "Test Body"

        # Email must NOT have been sent
        mock_send.assert_not_called()


def test_approval_sends_email_via_smtp(sample_profile, sample_company, sample_recipient):
    """Verify that approving a pending email invokes email_service.send_email and transitions status to SENT."""
    email_row = {
        "id": "email-app-1",
        "profile_id": sample_profile.id,
        "company_id": sample_company.id,
        "recipient_id": sample_recipient.id,
        "subject": "Great fit for Cyberdyne",
        "body": "Hi Miles, would love to talk.",
        "status": EmailStatus.PENDING_REVIEW.value,
    }

    with patch("app.services.email_service.send_email") as mock_send, \
         patch("app.db.models.fetch") as mock_fetch, \
         patch("app.db.models.update") as mock_update:

        def mock_fetch_impl(table, id_):
            if table == "emails":
                return email_row
            if table == "profiles":
                return sample_profile.model_dump()
            if table == "recipients":
                return sample_recipient.model_dump()
            return None

        mock_fetch.side_effect = mock_fetch_impl
        mock_update.side_effect = lambda table, id_, data: {**email_row, **data}

        result = main.approve_and_send_email("email-app-1")

        assert result.status == EmailStatus.SENT
        mock_send.assert_called_once_with(
            to_email=sample_recipient.email,
            subject=email_row["subject"],
            body=email_row["body"],
            from_email=sample_profile.email,
        )
