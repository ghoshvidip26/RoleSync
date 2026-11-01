"""ATS-friendly PDF renderer for approved structured resume drafts."""
from __future__ import annotations

import html
import re
from typing import Any

import pymupdf

PAGE_W, PAGE_H = pymupdf.paper_size('a4')
MARGIN = 42
BOTTOM = 43
FONT = 'helv'
# Font objects, not just names: drawing with insert_text turns every character outside Latin-1
# (en/em dashes, curly quotes, bullets, even the apostrophe in "I've") into a middle dot.
_FONTS = {FONT: pymupdf.Font(FONT), 'hebo': pymupdf.Font('hebo')}
INK = (0.12, 0.17, 0.23)
ACCENT = (0.10, 0.25, 0.40)


def _value(v: Any) -> str:
    if v is None:
        return ''
    if isinstance(v, dict):
        if set(v) == {'details'}:
            return _value(v['details'])
        return ', '.join(f'{k}: {_value(x)}' for k, x in v.items() if x)
    if isinstance(v, (list, tuple)):
        return ', '.join(_value(x) for x in v if x)
    return str(v).strip()


def _clean(v: Any) -> str:
    return re.sub(r'\s+', ' ', _value(v)).strip()


def _items(v: Any) -> list[str]:
    if not v:
        return []
    if isinstance(v, str):
        return [v]
    return [_clean(x) for x in v if _clean(x)]


class _Writer:
    def __init__(self):
        self.doc = pymupdf.open()
        self.page = None
        self.y = 0
        self.new_page()

    def new_page(self):
        self.page = self.doc.new_page(width=PAGE_W, height=PAGE_H)
        self.y = MARGIN

    def line(self, content: str, *, size: float = 9.2, bold: bool = False,
             color=INK, indent: float = 0, after: float = 3.5):
        content = _clean(content)
        if not content:
            return
        x = MARGIN + indent
        width = PAGE_W - MARGIN * 2 - indent
        font = 'hebo' if bold else FONT
        words = content.split()
        lines = []
        current = ''
        for word in words:
            test = (current + ' ' + word).strip()
            if _FONTS[font].text_length(test, fontsize=size) <= width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
                # Handle long unbroken URLs or words without clipping.
                while _FONTS[font].text_length(current, fontsize=size) > width:
                    cut = len(current) - 1
                    while cut > 1 and _FONTS[font].text_length(current[:cut], fontsize=size) > width:
                        cut -= 1
                    lines.append(current[:cut])
                    current = current[cut:]
        if current:
            lines.append(current)
        leading = size * 1.34
        for segment in lines:
            if self.y + leading > PAGE_H - BOTTOM:
                self.new_page()
            writer = pymupdf.TextWriter(self.page.rect)
            writer.append((x, self.y + size), segment, font=_FONTS[font], fontsize=size)
            writer.write_text(self.page, color=color)
            self.y += leading
        self.y += after

    def heading(self, title: str):
        if self.y + 34 > PAGE_H - BOTTOM:
            self.new_page()
        self.y += 7
        self.line(title.upper(), size=10, bold=True, color=ACCENT, after=3)
        self.page.draw_line((MARGIN, self.y), (PAGE_W - MARGIN, self.y),
                            color=(0.77, 0.81, 0.85), width=0.65)
        self.y += 7

    def bullet(self, content: str):
        self.line('• ' + _clean(content), indent=9, after=2)


def render_resume_pdf(profile: dict[str, Any], draft: dict[str, Any]) -> bytes:
    """Render a structured draft. Caller must verify approval and ownership."""
    w = _Writer()
    name = _clean(profile.get('name'))
    if not name:
        raise ValueError('Candidate name is required')
    w.line(name, size=19, bold=True, color=ACCENT, after=4)
    contacts = [_clean(profile.get(k)) for k in ('email', 'phone', 'linkedin', 'github', 'portfolio')]
    contacts = [c for c in contacts if c]
    if contacts:
        w.line('  |  '.join(contacts), size=8.2, after=8)
    if draft.get('headline'):
        w.line(draft['headline'], size=10, bold=True, after=6)
    if draft.get('summary'):
        w.heading('Professional Summary')
        w.line(draft['summary'])
    skills = draft.get('skills')
    if skills:
        w.heading('Technical Skills')
        if isinstance(skills, dict):
            for label, value in skills.items():
                if value:
                    w.line(f'{label}: {_value(value)}')
        else:
            w.line(_value(skills))
    experience = draft.get('experience') or []
    if experience:
        w.heading('Professional Experience')
        for entry in experience:
            if isinstance(entry, str):
                w.line(entry)
                continue
            title = ' — '.join(filter(None, [_clean(entry.get('job_title')), _clean(entry.get('company'))]))
            w.line(title, bold=True, after=1)
            w.line(' | '.join(filter(None, [_clean(entry.get('duration')), _clean(entry.get('location'))])),
                   size=8.4, after=3)
            for bullet in _items(entry.get('bullets')):
                w.bullet(bullet)
            w.y += 3
    projects = draft.get('projects') or []
    if projects:
        w.heading('Projects')
        for project in projects:
            if isinstance(project, str):
                w.line(project)
                continue
            w.line(_clean(project.get('name')), bold=True, after=1)
            if project.get('technologies'):
                w.line('Technologies: ' + _value(project['technologies']), size=8.4, after=2)
            if project.get('description'):
                w.line(project['description'], after=2)
            for bullet in _items(project.get('bullets')):
                w.bullet(bullet)
            w.y += 3
    for key, heading in (('achievements', 'Achievements'), ('education', 'Education')):
        values = draft.get(key) or []
        if values:
            w.heading(heading)
            for item in values if isinstance(values, list) else [values]:
                w.bullet(_value(item))
    if not any(draft.get(k) for k in ('summary', 'skills', 'experience', 'projects', 'achievements', 'education')):
        raise ValueError('Resume draft has no renderable content')
    result = w.doc.tobytes(garbage=4, deflate=True)
    w.doc.close()
    with pymupdf.open(stream=result, filetype='pdf') as check:
        if not check.page_count or not any(page.get_text().strip() for page in check):
            raise ValueError('Generated PDF has no extractable text')
    return result
