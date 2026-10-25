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
