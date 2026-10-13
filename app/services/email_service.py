import logging
import os
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

SMTP_EMAIL = os.getenv("SMTP_EMAIL")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")


def send_email(to_email: str, subject: str, body: str, from_email: str | None = None) -> None:
    """Send an outreach email via Gmail SMTP using credentials from .env."""
    sender = from_email or SMTP_EMAIL or os.getenv("SMTP_EMAIL")
    password = SMTP_PASSWORD or os.getenv("SMTP_PASSWORD")

    if not sender or not password:
        raise ValueError("SMTP credentials (SMTP_EMAIL, SMTP_PASSWORD) are missing from environment")

    message = EmailMessage()
    message["From"] = sender
    message["To"] = to_email
    message["Subject"] = subject
    message.set_content(body)

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(sender, password)
            server.send_message(message)
        logger.info("Email sent successfully to %s", to_email)
    except smtplib.SMTPException as exc:
        logger.error("SMTP error sending email to %s: %s", to_email, exc)
        raise RuntimeError(f"SMTP error sending email: {exc}") from exc
