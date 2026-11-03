import logging
import os
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

SMTP_EMAIL = os.getenv("SMTP_EMAIL")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")


# Gmail refuses messages over 25 MB in total; stay well under it (the encoded size is about a third larger).
MAX_ATTACHMENT_BYTES = 15 * 1024 * 1024


def send_email(
    to_email: str,
    subject: str,
    body: str,
    from_email: str | None = None,
    attachments: list[tuple[str, bytes, str]] | None = None,
) -> None:
    """Send an outreach email via Gmail SMTP using credentials from .env.

    ``attachments`` is a list of ``(filename, content, mime_type)``, e.g. ``("Name_Resume.pdf", pdf, "application/pdf")``.
    """
    sender = from_email or SMTP_EMAIL or os.getenv("SMTP_EMAIL")
    password = SMTP_PASSWORD or os.getenv("SMTP_PASSWORD")

    if not sender or not password:
        raise ValueError("SMTP credentials (SMTP_EMAIL, SMTP_PASSWORD) are missing from environment")

    message = EmailMessage()
    message["From"] = sender
    message["To"] = to_email
    message["Subject"] = subject
    message.set_content(body)

    total = 0
    for filename, content, mime_type in attachments or []:
        if not filename or "/" in filename or "\\" in filename or "\n" in filename:
            raise ValueError("Attachment needs a plain file name")
        if not isinstance(content, (bytes, bytearray)) or not content:
            raise ValueError(f"Attachment {filename} is empty")
        maintype, _, subtype = (mime_type or "application/octet-stream").partition("/")
        total += len(content)
        if total > MAX_ATTACHMENT_BYTES:
            raise ValueError("Attachments are too large to email")
        message.add_attachment(bytes(content), maintype=maintype, subtype=subtype or "octet-stream", filename=filename)

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
