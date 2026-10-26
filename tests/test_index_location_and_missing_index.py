import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from app.agents import resume_tailor

ROOT = Path(__file__).resolve().parents[1]


def test_default_index_dir_does_not_depend_on_the_working_directory(tmp_path):
    """Regression: INDEX_DIR was the relative path data/indexes, so launching from app/ looked in
    app/data/indexes and every profile appeared to have no index."""
    code = f"import sys; sys.path.insert(0, {str(ROOT)!r}); from app.services import outreach_agent as oa; print(oa.INDEX_DIR)"
    env = {"PATH": "/usr/bin:/bin"}          # no INDEX_DIR override
    out = {
        cwd: subprocess.run([sys.executable, "-W", "ignore", "-c", code], cwd=cwd, env=env, capture_output=True, text=True).stdout.strip()
        for cwd in (str(ROOT), str(ROOT / "app"), str(tmp_path))
    }
    assert set(out.values()) == {str(ROOT / "data" / "indexes")}


def test_tailor_resume_explains_a_missing_index_instead_of_crashing():
    profile = {"summary": "Backend engineer.", "skills": ["Python"]}
    with patch.object(resume_tailor, "get_retriever", return_value=None):
        with pytest.raises(resume_tailor.ResumeTailorError, match="No resume index found"):
            resume_tailor.tailor_resume("p1", profile, "Backend Engineer role")


def test_an_index_built_with_another_embedding_model_is_reported_not_crashed_on(tmp_path, monkeypatch, caplog):
    """Regression: after the embedding model changed from a 768- to a 4096-dimensional one, searching an
    old index failed with a bare `assert d == self.d` inside FAISS."""
    from langchain_core.documents import Document
    from langchain_core.embeddings import DeterministicFakeEmbedding

    from app.services import outreach_agent as oa

    monkeypatch.setattr(oa, "INDEX_DIR", tmp_path)
    oa.clear_index_cache()
    docs = [Document(page_content="Work Experience: Backend Engineer | PayLoop", metadata={"source_section": "experience"})]

    monkeypatch.setattr(oa, "get_embeddings", lambda: DeterministicFakeEmbedding(size=16))   # the "old" model
    oa._embedding_dimension.cache_clear()
    oa.index_resume("p1", docs)
    assert oa.get_retriever("p1") is not None                                                  # same model: usable

    oa.clear_index_cache()
    monkeypatch.setattr(oa, "get_embeddings", lambda: DeterministicFakeEmbedding(size=64))   # the "new" model
    oa._embedding_dimension.cache_clear()
    with caplog.at_level("WARNING"):
        assert oa.get_retriever("p1") is None
    assert "16-dimensional" in caplog.text and "64-dimensional" in caplog.text and "re-upload" in caplog.text
    oa.clear_index_cache()
    oa._embedding_dimension.cache_clear()
