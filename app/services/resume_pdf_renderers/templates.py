"""The resume template gallery: ATS-safe designs a user can pick from.

Every template follows the same safety rules, so no choice can hurt the ATS score:
single column, standard section names, standard fonts (PDF base-14, so they render the
same everywhere), real text only (no tables, text boxes, icons, images or columns),
one plain bullet glyph, ordinary dates, margins of at least 0.4 in and leading of at
least 1.07 (the r/EngineeringResumes guidance). Templates differ in typeface, header,
density and section order so that different kinds of engineer have a good fit.

Adding a template means adding a ``TemplateSpec`` here; ``template_renderer`` draws any spec.
``validate_spec`` enforces the safety rules, so an unsafe spec cannot be registered.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SECTION_TITLES = {
    "summary": "Summary",
    "skills": "Technical Skills",
    "experience": "Experience",
    "projects": "Projects",
    "achievements": "Achievements",
    "education": "Education",
}

MIN_MARGIN = 28.8          # 0.4 in
MIN_LEADING = 1.07
MIN_BODY_SIZE = 8.5
MAX_BODY_SIZE = 12.0
SAFE_BULLETS = {"•", "-", "–"}
FONT_FAMILIES = {          # PDF base-14 families: regular, bold, italic, bold italic
    "sans": ("helv", "hebo", "heit", "hebi"),
    "serif": ("tiro", "tibo", "tiit", "tibi"),
}


@dataclass(frozen=True)
class TemplateSpec:
    id: str
    name: str
    description: str
    best_for: str
    family: str                      # "sans" | "serif"
    body_size: float
    name_size: float
    heading_size: float
    contact_size: float
    leading: float                   # line pitch as a multiple of the font size
    margins: tuple[float, float, float, float]   # left, right, top, bottom (pt)
    header_align: str                # "center" | "left"
    heading_case: str                # "upper" | "title"
    accent: str                      # heading / name colour
    section_order: tuple[str, ...]
    entry_layout: str                # "org_first" | "title_first"
    section_gap: float
    entry_gap: float
    bullet: str = "•"
    bullet_indent: float = 11.0
    show_headline: bool = True
    ats_notes: tuple[str, ...] = field(default_factory=tuple)


TEMPLATES: dict[str, TemplateSpec] = {}


def validate_spec(spec: TemplateSpec) -> None:
    """Raise ValueError if a template could hurt ATS parsing or readability."""
    problems = []
    if spec.family not in FONT_FAMILIES:
        problems.append(f"font family must be one of {sorted(FONT_FAMILIES)}")
    if min(spec.margins) < MIN_MARGIN:
        problems.append("margins must be at least 0.4 in")
    if spec.leading < MIN_LEADING:
        problems.append(f"leading must be at least {MIN_LEADING}")
    if not MIN_BODY_SIZE <= spec.body_size <= MAX_BODY_SIZE:
        problems.append(f"body size must be {MIN_BODY_SIZE}-{MAX_BODY_SIZE} pt")
    if spec.bullet not in SAFE_BULLETS:
        problems.append(f"bullet must be one of {sorted(SAFE_BULLETS)}")
    if spec.header_align not in {"center", "left"}:
        problems.append("header_align must be center or left")
    if spec.entry_layout not in {"org_first", "title_first"}:
        problems.append("entry_layout must be org_first or title_first")
    unknown = [s for s in spec.section_order if s not in SECTION_TITLES]
    if unknown:
        problems.append(f"unknown sections: {unknown}")
    if len(set(spec.section_order)) != len(spec.section_order):
        problems.append("a section appears twice")
    if problems:
        raise ValueError(f"Template '{spec.id}' is not ATS-safe: " + "; ".join(problems))


def _register(spec: TemplateSpec) -> TemplateSpec:
    validate_spec(spec)
    TEMPLATES[spec.id] = spec
    return spec


_SAFE_NOTES = (
    "Single column, no tables, text boxes, icons or images",
    "Standard section names and plain dates",
    "Standard fonts that render the same everywhere",
)

_register(TemplateSpec(
    id="classic",
    name="Classic",
    description="The familiar engineering layout used on r/EngineeringResumes: centered name, ruled "
                "section headings, company and role on two lines with location and dates on the right.",
    best_for="Software engineers with a few years of experience, and anyone who wants the safest default.",
    family="serif", body_size=10.0, name_size=21.0, heading_size=11.0, contact_size=9.5, leading=1.22,
    margins=(43.2, 43.2, 36.0, 36.0), header_align="center", heading_case="upper", accent="#000000",
    section_order=("summary", "education", "experience", "projects", "skills", "achievements"),
    entry_layout="org_first", section_gap=7.0, entry_gap=4.0,
    ats_notes=_SAFE_NOTES + ("Education first, as most tech recruiters expect for early-career engineers",),
))

_register(TemplateSpec(
    id="modern",
    name="Modern Sans",
    description="Clean sans-serif with a left-aligned header, a muted blue accent on headings, and the "
                "job title leading each entry.",
    best_for="Mid-level and senior engineers; a more contemporary look that is still plain text.",
    family="sans", body_size=9.5, name_size=22.0, heading_size=11.0, contact_size=9.0, leading=1.25,
    margins=(43.2, 43.2, 40.0, 36.0), header_align="left", heading_case="title", accent="#1F3A5F",
    section_order=("summary", "skills", "experience", "projects", "achievements", "education"),
    entry_layout="title_first", section_gap=8.0, entry_gap=5.0,
    ats_notes=_SAFE_NOTES + ("Colour is used only for headings; text stays dark for readability",),
))

_register(TemplateSpec(
    id="harvard",
    name="Harvard Traditional",
    description="Traditional serif with generous margins and spacing, modelled on the plain text-only "
                "Harvard resume format: education first, restrained styling.",
    best_for="Students, new graduates and career changers; also academic or research-leaning roles.",
    family="serif", body_size=10.5, name_size=20.0, heading_size=11.5, contact_size=10.0, leading=1.28,
    margins=(54.0, 54.0, 46.0, 42.0), header_align="center", heading_case="upper", accent="#000000",
    section_order=("summary", "education", "experience", "projects", "skills", "achievements"),
    entry_layout="org_first", section_gap=9.0, entry_gap=5.0, show_headline=False,
    ats_notes=_SAFE_NOTES + ("Largest margins and spacing of the four; reads well on screen and in print",),
))

_register(TemplateSpec(
    id="compact",
    name="Compact Technical",
    description="Dense, skills-and-projects-first layout that fits a lot on one page without crowding. "
                "Smaller type, tight but readable spacing.",
    best_for="New grads and career changers with strong projects, or long histories that must stay on one page.",
    family="sans", body_size=8.8, name_size=19.0, heading_size=9.8, contact_size=8.6, leading=1.2,
    margins=(36.0, 36.0, 30.0, 28.8), header_align="left", heading_case="upper", accent="#000000",
    section_order=("summary", "skills", "projects", "experience", "education", "achievements"),
    entry_layout="org_first", section_gap=5.5, entry_gap=3.0, bullet_indent=9.0,
    ats_notes=_SAFE_NOTES + ("Skills and projects lead, which suits keyword scanning for early-career roles",),
))


def list_templates() -> list[dict]:
    """Gallery listing for the UI."""
    return [{"id": t.id, "name": t.name, "description": t.description, "best_for": t.best_for,
             "ats_notes": list(t.ats_notes), "sections": [SECTION_TITLES[s] for s in t.section_order],
             "font": "Times" if t.family == "serif" else "Helvetica"} for t in TEMPLATES.values()]


def get_template(template_id: str) -> TemplateSpec:
    spec = TEMPLATES.get((template_id or "").strip().lower())
    if spec is None:
        raise ValueError(f"Unknown template '{template_id}'. Choose one of: {', '.join(TEMPLATES)}")
    return spec
