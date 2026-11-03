"""Keep the user's uploaded resume PDF and say whether its design can be reproduced.

The preserving renderer needs the original file. It is stored on disk under
``ORIGINAL_RESUME_DIR`` (default ``data/resumes``), one file per profile id. This module
also assesses an upload: can its design be rebuilt reliably, and does it carry ATS risks?
Nothing here changes the PDF.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _directory() -> Path:
    return Path(os.getenv("ORIGINAL_RESUME_DIR", PROJECT_ROOT / "data" / "resumes"))


def _path(profile_id: str) -> Path:
    if not isinstance(profile_id, str) or not _ID.match(profile_id):
        raise ValueError("Invalid profile id")
    return _directory() / f"{profile_id}.pdf"


def save(profile_id: str, pdf: bytes) -> Path:
    if not isinstance(pdf, (bytes, bytearray)) or not pdf.startswith(b"%PDF"):
        raise ValueError("Not a PDF")
    path = _path(profile_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(bytes(pdf))
    tmp.replace(path)                       # never leave a half-written file behind
    return path


def load(profile_id: str) -> bytes | None:
    path = _path(profile_id)
    return path.read_bytes() if path.is_file() else None


def delete(profile_id: str) -> bool:
    path = _path(profile_id)
    side = path.with_suffix(".json")
    if side.is_file():
        side.unlink()
    if path.is_file():
        path.unlink()
        return True
    return False


def save_assessment(profile_id: str, assessment: dict[str, Any]) -> None:
    """Kept beside the PDF so the template gallery can answer instantly."""
    path = _path(profile_id).with_suffix(".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(assessment), encoding="utf-8")


def load_assessment(profile_id: str) -> dict[str, Any] | None:
    path = _path(profile_id).with_suffix(".json")
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    except (OSError, ValueError):
        return None


def contacts_of(pdf: bytes) -> dict[str, str]:
    """Phone, e-mail and links printed in the resume's header."""
    from .entry_grouper import group_entries
    from .layout_manifest import build_layout_manifest

    found: dict[str, str] = {}
    for c in group_entries(build_layout_manifest(pdf))["header"]["contacts"]:
        text, kind = c["text"], c["kind"]
        if kind in {"phone", "email"}:
            found.setdefault(kind, text)
        elif kind == "url":
            low = text.lower()
            found.setdefault("linkedin" if "linkedin" in low else "github" if "github" in low else "portfolio", text)
    return found


def fill_contacts(profile: dict[str, Any], assessment: dict[str, Any] | None) -> dict[str, Any]:
    """The profile with any missing contact detail taken from the stored resume (never overwrites)."""
    out = dict(profile)
    for key, value in ((assessment or {}).get("contacts") or {}).items():
        if value and not out.get(key):
            out[key] = value
    return out


def assess(pdf: bytes) -> dict[str, Any]:
    """Can this resume's own design be rebuilt, and what ATS risks does it carry?

    The design counts as supported only when an unchanged copy of the resume can be rebuilt
    and passes the same checks used on real output (nothing lost, nothing overlapping, ...).
    """
    from .layout_representation import build_layout_representation
    from .pdf_preserving_renderer import PdfPreservingRenderer
    from .render_validator import validate_rendering

    import pymupdf

    try:                                       # a file that cannot be opened is rejected, not stored
        with pymupdf.open(stream=pdf, filetype="pdf") as doc:
            if doc.is_encrypted:
                raise ValueError("Password-protected PDFs are not supported")
            if doc.page_count == 0 or not any(p.get_text().strip() for p in doc):
                raise ValueError("The PDF has no readable text (is it a scan?)")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("The PDF could not be read") from exc

    result: dict[str, Any] = {"preserve_supported": False, "reason": None, "checks": None, "ats": None,
                              "contacts": contacts_of(pdf)}
    try:
        result["ats"] = build_layout_representation(pdf)["ats"]
    except ValueError as exc:
        result["reason"] = str(exc)
        return result
    try:
        rebuilt, _ = PdfPreservingRenderer(pdf).render_with_report({})
    except ValueError as exc:
        result["reason"] = str(exc)
        return result
    verdict = validate_rendering(pdf, rebuilt, draft={})
    result["checks"] = verdict["summary"]
    if verdict["passed"]:
        result["preserve_supported"] = True
    else:
        failed = [c["name"] for c in verdict["checks"] if c["status"] == "fail"]
        result["reason"] = "A rebuilt copy did not pass these checks: " + ", ".join(failed)
    return result
