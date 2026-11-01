"""Two-column resume: a full-width header, a narrow sidebar (skills, education) and a main column
(summary, experience, projects, achievements). Content flows onto further pages and is never cut off."""
from __future__ import annotations

import html
import io
from typing import Any

import pymupdf

from app.services.resume_pdf_renderers.base import BaseResumeRenderer
from app.services.resume_pdf_renderers.resume_pdf import _clean, _items, _value

MARGIN = 42
BOTTOM = 43
SIDEBAR_WIDTH = 165
COLUMN_GAP = 18
HEADER_MAX_HEIGHT = 140
MAX_PAGES = 20            # a runaway layout (e.g. a column that can place nothing) stops here instead of looping

CSS = """
h1 { font-size: 19pt; color: #1a4066; margin: 0 0 3pt 0; }
h2 { font-size: 10pt; color: #1a4066; margin: 9pt 0 3pt 0; }
p { font-size: 9pt; margin: 0 0 3pt 0; }
li { font-size: 9pt; margin: 0 0 2pt 0; }
ul { margin: 0 0 3pt 0; padding: 0 0 0 10pt; }
.contact { font-size: 8pt; }
.headline { font-size: 10pt; font-weight: bold; }
.role { font-weight: bold; margin-bottom: 1pt; }
.meta { font-size: 8pt; margin-bottom: 2pt; }
"""


def _e(value: Any) -> str:
    return html.escape(_clean(value))


def _bullets(values: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{_e(v)}</li>" for v in values) + "</ul>" if values else ""


def _section(title: str, body: str) -> str:
    return f"<h2>{html.escape(title.upper())}</h2>{body}" if body else ""


def _education_text(item: Any) -> str:
    return _clean(item.get("details") if isinstance(item, dict) and "details" in item else item)


def _header_html(profile: dict[str, Any], draft: dict[str, Any]) -> str:
    name = _clean(profile.get("name"))
    contacts = [_clean(profile.get(k)) for k in ("email", "phone", "linkedin", "github", "portfolio")]
    parts = [f"<h1>{html.escape(name)}</h1>"]
    if any(contacts):
        parts.append(f"<p class='contact'>{html.escape('  |  '.join(c for c in contacts if c))}</p>")
    if draft.get("headline"):
        parts.append(f"<p class='headline'>{_e(draft['headline'])}</p>")
    return "".join(parts)


def _sidebar_html(draft: dict[str, Any]) -> str:
    skills = draft.get("skills")
    if isinstance(skills, dict):
        skill_html = "".join(f"<p><b>{_e(label)}:</b> {_e(value)}</p>" for label, value in skills.items() if value)
    else:
        skill_html = _bullets(_items(skills))
    education = draft.get("education") or []
    education = education if isinstance(education, list) else [education]
    return _section("Skills", skill_html) + _section(
        "Education", _bullets([t for t in (_education_text(i) for i in education) if t])
    )


def _main_html(draft: dict[str, Any]) -> str:
    experience = []
    for entry in draft.get("experience") or []:
        if isinstance(entry, str):
            experience.append(f"<p>{_e(entry)}</p>")
            continue
        title = " — ".join(filter(None, [_clean(entry.get("job_title")), _clean(entry.get("company"))]))
        meta = " | ".join(filter(None, [_clean(entry.get("duration")), _clean(entry.get("location"))]))
        experience.append(
            f"<p class='role'>{html.escape(title)}</p>"
            + (f"<p class='meta'>{html.escape(meta)}</p>" if meta else "")
            + _bullets(_items(entry.get("bullets")))
        )

    projects = []
    for project in draft.get("projects") or []:
        if isinstance(project, str):
            projects.append(f"<p>{_e(project)}</p>")
            continue
        projects.append(
            f"<p class='role'>{_e(project.get('name'))}</p>"
            + (f"<p class='meta'>Technologies: {_e(_value(project['technologies']))}</p>" if project.get("technologies") else "")
            + (f"<p>{_e(project['description'])}</p>" if project.get("description") else "")
            + _bullets(_items(project.get("bullets")))
        )

    achievements = draft.get("achievements") or []
    achievements = achievements if isinstance(achievements, list) else [achievements]
    return (
        _section("Professional Summary", f"<p>{_e(draft['summary'])}</p>" if draft.get("summary") else "")
        + _section("Professional Experience", "".join(experience))
        + _section("Projects", "".join(projects))
        + _section("Achievements", _bullets([t for t in (_clean(_value(a)) for a in achievements) if t]))
    )


class TwoColumnRenderer(BaseResumeRenderer):
    """First version: a fixed sidebar layout. It does not yet copy the exact column positions that the
    layout analyzer measured in the original resume (`layout_metadata` is accepted but unused)."""

    def render(
        self,
        profile: dict[str, Any],
        draft: dict[str, Any],
        layout_metadata: dict[str, Any],
    ) -> bytes:
        if not _clean(profile.get("name")):
            raise ValueError("Candidate name is required")
        sidebar_html, main_html = _sidebar_html(draft), _main_html(draft)
        if not (sidebar_html or main_html):
            raise ValueError("Resume draft has no renderable content")

        header = pymupdf.Story(html=_header_html(profile, draft), user_css=CSS)
        sidebar = pymupdf.Story(html=sidebar_html, user_css=CSS) if sidebar_html else None
        main = pymupdf.Story(html=main_html, user_css=CSS) if main_html else None

        buffer = io.BytesIO()
        writer = pymupdf.DocumentWriter(buffer)
        mediabox = pymupdf.paper_rect("a4")
        page_w, page_h = mediabox.width, mediabox.height
        sidebar_more, main_more, pages = sidebar is not None, main is not None, 0
        while sidebar_more or main_more or pages == 0:
            pages += 1
            if pages > MAX_PAGES:
                raise ValueError("Resume is too long to fit the two-column layout")
            device = writer.begin_page(mediabox)
            top = MARGIN
            if pages == 1:
                header_more, filled = header.place(pymupdf.Rect(MARGIN, MARGIN, page_w - MARGIN, MARGIN + HEADER_MAX_HEIGHT))
                if header_more:
                    raise ValueError("Header (name, contacts, headline) is too long for the page")
                header.draw(device)
                top = pymupdf.Rect(filled).y1 + 8      # place() returns the used area as a plain tuple
            bottom = page_h - BOTTOM
            if sidebar_more:
                sidebar_more, _ = sidebar.place(pymupdf.Rect(MARGIN, top, MARGIN + SIDEBAR_WIDTH, bottom))
                sidebar.draw(device)
            if main_more:
                main_more, _ = main.place(pymupdf.Rect(MARGIN + SIDEBAR_WIDTH + COLUMN_GAP, top, page_w - MARGIN, bottom))
                main.draw(device)
            writer.end_page()
        writer.close()

        result = buffer.getvalue()
        with pymupdf.open(stream=result, filetype="pdf") as check:
            if not check.page_count or not any(page.get_text().strip() for page in check):
                raise ValueError("Generated PDF has no extractable text")
        return result
