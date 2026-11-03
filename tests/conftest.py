import io
import os
import shutil
import tempfile
import pytest
import pymupdf
from unittest.mock import MagicMock

# Ensure testing environment variables
os.environ["INDEX_DIR"] = tempfile.mkdtemp(prefix="test_indexes_")
# Defaults only: set these variables yourself to run the live test with the models you have installed.
os.environ.setdefault("OLLAMA_CHAT_MODEL", "qwen:latest")
os.environ.setdefault("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")


@pytest.fixture(autouse=True)
def no_cloud_fallback(monkeypatch):
    """The developer's .env holds a real GROQ_API_KEY; tests must never send anything to Groq."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("LLM_FALLBACK", raising=False)


@pytest.fixture(autouse=True)
def cleanup_test_indexes():
    """Clear test indexes directory and cache before each test."""
    from app.services import outreach_agent

    outreach_agent.clear_index_cache()
    test_dir = os.environ["INDEX_DIR"]
    yield
    outreach_agent.clear_index_cache()
    if os.path.exists(test_dir):
        for item in os.listdir(test_dir):
            p = os.path.join(test_dir, item)
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            elif os.path.isfile(p):
                os.unlink(p)


def create_sample_pdf(text: str = "Jane Doe\nEmail: jane@example.com\nSoftware Engineer") -> bytes:
    """Helper to generate in-memory valid PDF bytes."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def create_blank_pdf() -> bytes:
    """Helper to generate in-memory blank/scanned PDF bytes with no text."""
    doc = pymupdf.open()
    doc.new_page()
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes
