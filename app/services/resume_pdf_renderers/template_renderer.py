"""Draw a resume from a template in the gallery (see ``templates.py``).

Input is the user's profile (contact details, skills) and the approved tailored draft
(summary, skills to highlight, experience, projects, achievements, education). Output is
a text-only, single-column PDF. Nothing is invented: only what the profile and draft
contain is drawn, and sections with no content are left out.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

import pymupdf

from .base import BaseResumeRenderer
from .entry_grouper import has_date
from .pdf_preserving_renderer import _fix_text_maps
from dataclasses import replace

from .templates import (FONT_FAMILIES, MIN_BODY_SIZE, MIN_LEADING, SECTION_TITLES, TemplateSpec,
                        get_template, validate_spec)

MAX_PAGES = 4
# (type scale, spacing scale): gentlest first. Used to keep a resume on one page; never goes below
# the template safety limits (body size, leading), which validate_spec enforces.
FIT_STEPS = [(1.0, 1.0), (1.0, 0.92), (0.97, 0.88), (0.94, 0.85), (0.92, 0.82), (0.90, 0.8)]
_URL = re.compile(r"^(?:https?://|www\.)|(?:\.com|\.io|\.dev|\.me|\.in|\.net|\.org)(?:/|$)", re.I)


def _rgb(hex_color: str) -> tuple[float, float, float]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def _clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        if "details" in value and len(value) == 1:
            return str(value["details"]).strip()
        return ", ".join(f"{_clean(v)}" for v in value.values() if _clean(v))
    if isinstance(value, (list, tuple)):
        return ", ".join(_clean(v) for v in value if _clean(v))
    return re.sub(r"[ \t]+", " ", str(value)).strip()


def _list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [p.strip() for p in re.split(r"[,;|\n]", value) if p.strip()]
    return [t for t in (_clean(v) for v in value) if t]


# ------------------------------------------------------------------- content

def _skills(profile: dict, draft: dict) -> list[tuple[str, list[str]]]:
    """Skill groups from the profile, with the draft's job-relevant skills moved first."""
    highlight = {re.sub(r"[^a-z0-9+#]", "", s.lower()): i for i, s in enumerate(_list(draft.get("skills")))}

    def ordered(values: list[str]) -> list[str]:
        hit = sorted((v for v in values if re.sub(r"[^a-z0-9+#]", "", v.lower()) in highlight),
                     key=lambda v: highlight[re.sub(r"[^a-z0-9+#]", "", v.lower())])
        return hit + [v for v in values if v not in hit]

    groups = [("Languages", _list(profile.get("programming_languages"))),
              ("Frameworks", _list(profile.get("frameworks"))),
              ("Databases & Tools", _list(profile.get("databases_tools")))]
    other = _list(profile.get("skills"))
    used = {v.lower() for _, vals in groups for v in vals}
    groups.append(("Skills" if not any(vals for _, vals in groups) else "Other", [v for v in other if v.lower() not in used]))
    out = [(label, ordered(vals)) for label, vals in groups if vals]
    if not out and highlight:
        out = [("Skills", _list(draft.get("skills")))]
    return out


def _education(draft: dict) -> list[tuple[str, str]]:
    """(left text, right-hand dates) per education item."""
    items = draft.get("education") or []
    if isinstance(items, (str, dict)):
        items = [items]
    out = []
    for item in items:
        text = _clean(item)
        parts = [p.strip() for p in text.split("\n") if p.strip()]
        if not parts:
            continue
        if len(parts) > 1 and has_date(parts[-1]) and len(parts[-1]) <= 40:
            out.append((" | ".join(parts[:-1]), parts[-1]))
        else:
            out.append((" | ".join(parts), ""))
    return out


# ------------------------------------------------------------------- drawing

class _Pen:
    """Pagination, wrapping and drawing for one document."""

    def __init__(self, spec: TemplateSpec, doc: pymupdf.Document):
        self.spec, self.doc = spec, doc
        reg, bold, ital, bital = FONT_FAMILIES[spec.family]
        self.fonts = {"r": pymupdf.Font(reg), "b": pymupdf.Font(bold), "i": pymupdf.Font(ital), "bi": pymupdf.Font(bital)}
        self.w, self.h = pymupdf.paper_size("letter")
        self.left, self.right, self.top, self.bottom = spec.margins[0], self.w - spec.margins[1], spec.margins[2], \
            self.h - spec.margins[3]
        self.page = None
        self.y = 0.0
        self.pages = 0
        self.new_page()

    # ---- primitives
    def new_page(self):
        if self.pages >= MAX_PAGES:
            raise ValueError(f"Resume would exceed {MAX_PAGES} pages")
        self.page = self.doc.new_page(width=self.w, height=self.h)
        self.pages += 1
        self.y = self.top
        self.fresh = True

    def width(self, text: str, style: tuple) -> float:
        return self.fonts[style[0]].text_length(text, fontsize=style[1])

    def put(self, x: float, text: str, style: tuple):
        tw = pymupdf.TextWriter(self.page.rect)
        tw.append((x, self.y), text, font=self.fonts[style[0]], fontsize=style[1])
        tw.write_text(self.page, color=_rgb(style[2]))

    def pitch(self, size: float) -> float:
        return size * self.spec.leading

    def need(self, height: float):
        """Start a new page unless ``height`` more points fit (never on a fresh page)."""
        if not self.fresh and self.y + height > self.bottom:
            self.new_page()

    def advance(self, size: float):
        self.y += self.pitch(size)
        self.fresh = False

    def body(self, bold=False, italic=False, size=None, color="#222222") -> tuple:
        key = "bi" if bold and italic else "b" if bold else "i" if italic else "r"
        return (key, size or self.spec.body_size, color)

    # ---- wrapping
    def wrap(self, segs: list[tuple[str, tuple]], width: float) -> list[list[tuple[str, tuple]]]:
        lines: list[list[tuple[str, tuple]]] = [[]]
        used = 0.0
        for text, style in segs:
            for word in text.split():
                ww = self.width(word, style)
                sp = self.width(" ", style) if lines[-1] else 0.0
                if lines[-1] and used + sp + ww > width:
                    lines.append([])
                    used, sp = 0.0, 0.0
                while ww > width and not lines[-1]:                    # one word wider than the line
                    cut = len(word) - 1
                    while cut > 1 and self.width(word[:cut], style) > width:
                        cut -= 1
                    lines[-1].append((word[:cut], style))
                    lines.append([])
                    word, ww, used = word[cut:], self.width(word[cut:], style), 0.0
                lines[-1].append((word, style))
                used += sp + ww
        return [l for l in lines if l]

    def _merged(self, line):
        out: list[tuple[str, tuple]] = []
        for word, style in line:
            if out and out[-1][1] == style:
                out[-1] = (out[-1][0] + " " + word, style)
            else:
                out.append((word, style))
        return out

    def _draw_runs(self, x: float, runs: list[tuple[str, tuple]]) -> float:
        for i, (text, style) in enumerate(runs):
            if i:
                x += self.width(" ", runs[i - 1][1])
            self.put(x, text + (" " if i < len(runs) - 1 else ""), style)
            x += self.width(text, style)
        return x

    # ---- elements
    def paragraph(self, segs, x: float | None = None, width: float | None = None, align: str = "left", gap: float = 0.0):
        x = self.left if x is None else x
        width = (self.right - x) if width is None else width
        size = max(st[1] for _, st in segs)
        for line in self.wrap(segs, width):
            self.need(self.pitch(size))
            runs = self._merged(line)
            total = sum(self.width(t, s) for t, s in runs) + sum(self.width(" ", s) for _, s in runs[:-1])
            start = x if align == "left" else x + (width - total) / 2
            self._draw_runs(start, runs)
            self.advance(size)
        self.y += gap

    def pair(self, left_segs, right_text: str, right_style: tuple):
        """Left text with a right-aligned item on the same baseline."""
        size = max(st[1] for _, st in left_segs)
        rw = self.width(right_text, right_style) if right_text else 0.0
        avail = (self.right - self.left) - (rw + 12 if rw else 0)
        lines = self.wrap(left_segs, avail)
        for k, line in enumerate(lines):
            self.need(self.pitch(size))
            self._draw_runs(self.left, self._merged(line))
            if k == 0 and right_text:
                self.put(self.right - rw, right_text, right_style)
            self.advance(size)

    def bullet(self, text: str):
        sp = self.spec
        st = self.body()
        size = st[1]
        x_text = self.left + sp.bullet_indent
        lines = self.wrap([(text, st)], self.right - x_text)
        self.need(self.pitch(size) * min(len(lines), 2))
        for k, line in enumerate(lines):
            self.need(self.pitch(size))
            if k == 0:
                self.put(self.left + 2, sp.bullet + " ", st)
            self._draw_runs(x_text, self._merged(line))
            self.advance(size)

    def heading(self, title: str):
        sp = self.spec
        text = title.upper() if sp.heading_case == "upper" else title
        st = ("b", sp.heading_size, sp.accent)
        self.need(self.pitch(sp.heading_size) * 3.2 + self.pitch(sp.body_size) * 2)
        if not self.fresh:
            self.y += sp.section_gap
        self.put(self.left, text, st)
        rule_y = self.y + 3.2
        self.page.draw_line((self.left, rule_y), (self.right, rule_y), color=_rgb(sp.accent if sp.accent != "#000000" else "#444444"),
                            width=0.6)
        self.y += self.pitch(sp.heading_size) + 1.5
        self.fresh = False


# ------------------------------------------------------------------ rendering

class TemplateRenderer(BaseResumeRenderer):
    """Renders the profile and approved draft with one of the gallery templates."""

    def __init__(self, template_id: str = "classic", target_pages: int = 1):
        self.spec = get_template(template_id)
        self.target_pages = max(1, int(target_pages))

    def render(self, profile: dict[str, Any], draft: dict[str, Any], layout_metadata: dict[str, Any] | None = None) -> bytes:
        return self.render_with_report(profile, draft)[0]

    def _scaled(self, type_scale: float, space_scale: float) -> TemplateSpec | None:
        sp = self.spec
        body = round(sp.body_size * type_scale, 2)
        leading = max(MIN_LEADING, round(sp.leading * (1 - (1 - space_scale) * 0.5), 3))
        if body < MIN_BODY_SIZE and type_scale != 1.0:
            return None
        spec = replace(sp, body_size=body, name_size=round(sp.name_size * type_scale, 2),
                       heading_size=round(sp.heading_size * type_scale, 2),
                       contact_size=round(sp.contact_size * type_scale, 2), leading=leading,
                       section_gap=round(sp.section_gap * space_scale, 2), entry_gap=round(sp.entry_gap * space_scale, 2))
        validate_spec(spec)
        return spec

    def render_with_report(self, profile: dict[str, Any], draft: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
        """Render, tightening type and spacing a little (within the safety limits) until the resume
        fits ``target_pages``; if it still does not fit, the template's own sizes are used."""
        if not isinstance(profile, dict) or not isinstance(draft, dict):
            raise ValueError("Profile and draft must be dictionaries")
        first = None
        for type_scale, space_scale in FIT_STEPS:
            spec = self._scaled(type_scale, space_scale)
            if spec is None:
                continue
            try:
                pdf, report = self._render_once(spec, profile, draft)
            except ValueError as exc:
                if "exceed" in str(exc):          # too many pages even here; a tighter step may fit
                    continue
                raise
            report["fit"] = {"type_scale": type_scale, "spacing_scale": space_scale,
                             "fits_target": report["pages"] <= self.target_pages}
            first = first or (pdf, report)
            if report["pages"] <= self.target_pages:
                return pdf, report
        if first is None:
            raise ValueError("The resume is too long for this template")
        return first

    def _render_once(self, spec: TemplateSpec, profile: dict[str, Any], draft: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
        name = _clean(profile.get("name"))
        if not name:
            raise ValueError("Candidate name is required")
        self._active = spec
        sp = spec
        doc = pymupdf.open()
        pen = _Pen(sp, doc)
        drawn = self._header(pen, profile, draft, name)

        builders = {
            "summary": self._summary, "skills": self._skills, "experience": self._experience,
            "projects": self._projects, "achievements": self._achievements, "education": self._education,
        }
        sections = []
        for key in sp.section_order:
            if builders[key](pen, profile, draft):
                sections.append(SECTION_TITLES[key])
        if not sections:
            raise ValueError("Resume draft has no renderable content")

        try:
            doc.subset_fonts()
        except Exception:
            pass
        _fix_text_maps(doc)
        out = doc.tobytes(garbage=3, deflate=True)
        pages = doc.page_count
        doc.close()
        with pymupdf.open(stream=out, filetype="pdf") as check:
            if not any(p.get_text().strip() for p in check):
                raise ValueError("Generated PDF has no extractable text")
        return out, {"template": sp.id, "pages": pages, "sections": sections, "contacts": drawn}

    # ---- header
    def _header(self, pen: _Pen, profile: dict, draft: dict, name: str) -> list[str]:
        sp = self._active
        center = sp.header_align == "center"
        pen.y = pen.top + sp.name_size * 0.8
        pen.fresh = False
        pen.paragraph([(name, ("b", sp.name_size, sp.accent))], align="center" if center else "left", gap=1.5)
        headline = _clean(draft.get("headline"))
        if headline and sp.show_headline:
            pen.paragraph([(headline, ("b", sp.contact_size + 0.5, "#333333"))], align="center" if center else "left", gap=1.0)
        pieces = []
        for key in ("phone", "email", "linkedin", "github", "portfolio"):
            v = _clean(profile.get(key))
            if v:
                pieces.append((key, v))
        if pieces:
            st = ("r", sp.contact_size, "#222222")
            text = " | ".join(v for _, v in pieces)
            pen.paragraph([(text, st)], align="center" if center else "left", gap=1.0)
            self._links(pen, pieces, st, center)
        return [v for _, v in pieces]

    def _links(self, pen: _Pen, pieces, style, center: bool):
        """Make e-mail and web addresses clickable on the (single-line) contact row."""
        sep = " | "
        full = sep.join(v for _, v in pieces)
        if pen.width(full, style) > pen.right - pen.left:      # wrapped onto two lines: skip, text stays readable
            return
        x = pen.left if not center else pen.left + ((pen.right - pen.left) - pen.width(full, style)) / 2
        y = pen.y - 1.0 - pen.pitch(style[1])                     # baseline of the contact row just drawn
        for key, value in pieces:
            w = pen.width(value, style)
            uri = None
            if key == "email" or "@" in value and " " not in value:
                uri = "mailto:" + value
            elif key in {"linkedin", "github", "portfolio"} or _URL.search(value):
                uri = value if value.lower().startswith("http") else "https://" + value
            if uri:
                pen.page.insert_link({"kind": pymupdf.LINK_URI, "uri": uri,
                                      "from": pymupdf.Rect(x, y - style[1] * 0.85, x + w, y + style[1] * 0.3)})
            x += w + pen.width(sep, style)

    # ---- sections (each returns True when it drew something)
    def _summary(self, pen, profile, draft) -> bool:
        text = _clean(draft.get("summary"))
        if not text:
            return False
        pen.heading(SECTION_TITLES["summary"])
        pen.paragraph([(text, pen.body())])
        return True

    def _skills(self, pen, profile, draft) -> bool:
        groups = _skills(profile, draft)
        if not groups:
            return False
        pen.heading(SECTION_TITLES["skills"])
        for label, values in groups:
            pen.paragraph([(label + ":", pen.body(bold=True)), (", ".join(values), pen.body())])
        return True

    def _experience(self, pen, profile, draft) -> bool:
        entries = [e for e in (draft.get("experience") or []) if isinstance(e, dict)]
        if not entries:
            return False
        sp = self._active
        pen.heading(SECTION_TITLES["experience"])
        for i, e in enumerate(entries):
            company, title = _clean(e.get("company")), _clean(e.get("job_title"))
            loc, dates = _clean(e.get("location")), _clean(e.get("duration"))
            pen.need(pen.pitch(sp.body_size) * 4)
            if i:
                pen.y += sp.entry_gap
            small = sp.body_size - 0.3
            if sp.entry_layout == "org_first":
                pen.pair([(company or title, pen.body(bold=True, size=sp.body_size + 0.5, color="#000000"))], loc, pen.body(size=small))
                if company and title or dates:
                    pen.pair([(title, pen.body(italic=True, size=small))], dates, pen.body(italic=True, size=small))
            else:
                pen.pair([(title or company, pen.body(bold=True, size=sp.body_size + 0.5, color="#000000"))], dates, pen.body(size=small))
                line2 = ", ".join(x for x in (company if title else "", loc) if x)
                if line2:
                    pen.pair([(line2, pen.body(italic=True, size=small))], "", pen.body())
            for b in e.get("bullets") or []:
                if _clean(b):
                    pen.bullet(_clean(b))
        return True

    def _projects(self, pen, profile, draft) -> bool:
        entries = [p for p in (draft.get("projects") or []) if isinstance(p, dict)]
        if not entries:
            return False
        sp = self._active
        pen.heading(SECTION_TITLES["projects"])
        for i, p in enumerate(entries):
            pen.need(pen.pitch(sp.body_size) * 3)
            if i:
                pen.y += sp.entry_gap
            tech = _clean(p.get("technologies"))
            segs = [(_clean(p.get("name")), pen.body(bold=True, size=sp.body_size + 0.5, color="#000000"))]
            if tech:
                segs += [("|", pen.body()), (tech, pen.body(italic=True))]
            pen.pair(segs, _clean(p.get("dates") or p.get("duration")), pen.body(italic=True, size=sp.body_size - 0.3))
            if _clean(p.get("description")):
                pen.paragraph([(_clean(p.get("description")), pen.body())])
            for b in p.get("bullets") or []:
                if _clean(b):
                    pen.bullet(_clean(b))
        return True

    def _achievements(self, pen, profile, draft) -> bool:
        items = [t for t in (_clean(a) for a in (draft.get("achievements") or [])) if t]
        if not items:
            return False
        pen.heading(SECTION_TITLES["achievements"])
        for t in items:
            pen.bullet(t)
        return True

    def _education(self, pen, profile, draft) -> bool:
        items = _education(draft)
        if not items:
            return False
        sp = self._active
        pen.heading(SECTION_TITLES["education"])
        for i, (left, right) in enumerate(items):
            if i:
                pen.y += sp.entry_gap
            pen.pair([(left, pen.body(bold=True, color="#000000"))], right, pen.body(italic=True, size=sp.body_size - 0.3))
        return True


# -------------------------------------------------------------------- previews

def sample_profile() -> dict:
    return {"name": "Alex Morgan", "phone": "+1 555 010 1234", "email": "alex.morgan@example.com",
            "linkedin": "linkedin.com/in/alexmorgan", "github": "github.com/alexmorgan",
            "programming_languages": ["Python", "Go", "TypeScript", "SQL"],
            "frameworks": ["FastAPI", "React", "Node.js"],
            "databases_tools": ["PostgreSQL", "Redis", "Docker", "AWS", "Git"]}


def sample_draft() -> dict:
    return {
        "headline": "Backend Engineer | Python, Go, AWS",
        "summary": "Backend engineer with four years of experience building payment and data services in "
                   "Python and Go, focused on reliability and clear APIs.",
        "skills": ["Python", "AWS"],
        "experience": [
            {"company": "Northwind Labs", "job_title": "Software Engineer", "duration": "Jan 2023 – Present",
             "location": "Austin, TX", "bullets": [
                 "Designed and shipped a ledger service handling 2M transactions per day with 99.95% uptime.",
                 "Cut p95 API latency from 480 ms to 190 ms by redesigning query paths and caching in Redis.",
                 "Led the migration of two services to AWS ECS, reducing monthly infrastructure cost by 22%.",
                 "Mentored three engineers and introduced design reviews that cut production incidents by a third."]},
            {"company": "Bluepeak Systems", "job_title": "Junior Developer", "duration": "Jun 2021 – Dec 2022",
             "location": "Remote", "bullets": [
                 "Built REST endpoints and background jobs for an order-management platform used by 40 retailers.",
                 "Wrote integration tests that raised coverage from 52% to 84% and caught three release blockers.",
                 "Automated nightly data exports in Python, saving the support team about six hours a week."]},
            {"company": "Lakeside Analytics", "job_title": "Software Engineering Intern", "duration": "May 2020 – Aug 2020",
             "location": "Chicago, IL", "bullets": [
                 "Prototyped a dashboard API in Flask that replaced a manual weekly reporting process.",
                 "Added unit tests and CI checks to a data-cleaning library used by five analysts."]}],
        "projects": [
            {"name": "StreamKit", "technologies": "Go, Kafka, PostgreSQL", "dates": "2024",
             "description": "", "bullets": ["Open-source event pipeline library with 600+ GitHub stars.",
                                            "Implemented exactly-once delivery with idempotent consumers and a replay tool."]},
            {"name": "TrailNotes", "technologies": "TypeScript, React, FastAPI", "dates": "2023",
             "description": "", "bullets": ["Offline-first hiking journal with sync, used by about 1,200 people.",
                                            "Cut cold-start time by 40% with code splitting and cached map tiles."]}],
        "achievements": ["Winner, 2023 City Hackathon (48-hour build)."],
        "education": [{"details": "B.S. in Computer Science | State University, Austin, TX\nAug 2017 – May 2021"}],
    }


@lru_cache(maxsize=16)
def preview_png(template_id: str, dpi: int = 70) -> bytes:
    """PNG preview of a template filled with fictional sample data (for the gallery)."""
    pdf = TemplateRenderer(template_id).render(sample_profile(), sample_draft())
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        return doc[0].get_pixmap(dpi=dpi).tobytes("png")
