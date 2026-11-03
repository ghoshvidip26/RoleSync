
from __future__ import annotations

import re
from typing import Any

import pymupdf

from app.services.resume_pdf_renderers.base import BaseResumeRenderer


DEFAULTS = {
    "font": "helv",
    "bold_font": "hebo",
    "italic_font": "heit",
    "bold_italic_font": "hebi",
    "body_size": 9.0,
    "heading_size": 11.0,
    "name_size": 22.0,
    "body_color": "#000000",
    "heading_color": "#0046A0",
    "margin": 36.0,
}


_FONT_CACHE: dict[str, pymupdf.Font] = {}


def _draw_text(page, point, text, *, fontname, fontsize, color):
    """Draw with a Font object: insert_text turns en/em dashes, curly quotes and
    apostrophes (anything outside Latin-1) into a middle dot."""
    font = _FONT_CACHE.setdefault(fontname, pymupdf.Font(fontname))
    writer = pymupdf.TextWriter(page.rect)
    writer.append(point, text, font=font, fontsize=fontsize)
    writer.write_text(page, color=color)


def _clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()
    return str(value).strip()


def _as_list(value: Any) -> list:
    if not value:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [{"name": key, "value": item} for key, item in value.items()]
    return [value]


def _color(value: Any, fallback: str) -> tuple[float, float, float]:
    """Convert #RRGGBB to the RGB tuple expected by PyMuPDF."""
    if not isinstance(value, str):
        value = fallback

    value = value.strip().lstrip("#")

    if not re.fullmatch(r"[0-9a-fA-F]{6}", value):
        value = fallback.lstrip("#")

    return tuple(
        int(value[index:index + 2], 16) / 255
        for index in (0, 2, 4)
    )


def _font_name(fonts: list[str], style: str = "regular") -> str:
    """
    Map common source font names to PyMuPDF built-in fonts.

    Exact custom font reproduction requires extracting and embedding
    the original font file, which is not part of the current metadata.
    """
    source = " ".join(fonts).lower()

    if style == "bold_italic":
        return "hebi"
    if style == "bold":
        return "hebo"
    if style == "italic":
        return "heit"

    return "helv"


class _SingleColumnWriter:
    def __init__(
        self,
        page_width: float,
        page_height: float,
        margins: dict,
        styles: dict,
    ):
        self.doc = pymupdf.open()
        self.page_width = page_width
        self.page_height = page_height
        self.margins = margins
        self.styles = styles

        self.left = margins["left"]
        self.right = page_width - margins["right"]
        self.top = margins["top"]
        self.bottom = page_height - margins["bottom"]
        self.content_width = self.right - self.left

        if self.content_width <= 0 or self.bottom <= self.top:
            raise ValueError("Invalid resume page margins")

        self.page = None
        self.y = self.top
        self.new_page()

    def new_page(self):
        self.page = self.doc.new_page(
            width=self.page_width,
            height=self.page_height,
        )
        self.y = self.top

    def ensure_space(self, height: float):
        if self.y + height > self.bottom:
            self.new_page()

    def _wrap(
        self,
        text: str,
        font: str,
        size: float,
        max_width: float,
    ) -> list[str]:
        words = _clean(text).split()
        if not words:
            return []

        lines = []
        current = ""

        for word in words:
            candidate = f"{current} {word}".strip()

            if pymupdf.get_text_length(
                candidate,
                fontname=font,
                fontsize=size,
            ) <= max_width:
                current = candidate
                continue

            if current:
                lines.append(current)

            # Split a single word if it is wider than the available line.
            if pymupdf.get_text_length(
                word,
                fontname=font,
                fontsize=size,
            ) > max_width:
                part = ""
                for char in word:
                    if (
                        part
                        and pymupdf.get_text_length(
                            part + char,
                            fontname=font,
                            fontsize=size,
                        ) > max_width
                    ):
                        lines.append(part)
                        part = char
                    else:
                        part += char
                current = part
            else:
                current = word

        if current:
            lines.append(current)

        return lines

    def write_text(
        self,
        text: str,
        *,
        size: float | None = None,
        font: str | None = None,
        color: tuple | None = None,
        indent: float = 0,
        gap: float = 2,
        align: int = 0,
    ):
        text = _clean(text)
        if not text:
            return

        size = size or self.styles["body_size"]
        font = font or self.styles["body_font"]
        color = color or self.styles["body_color"]

        x = self.left + indent
        width = self.content_width - indent

        lines = self._wrap(text, font, size, width)
        leading = size * 1.3

        for line in lines:
            self.ensure_space(leading + gap)

            if align == 1:
                text_width = pymupdf.get_text_length(
                    line, fontname=font, fontsize=size
                )
                position_x = self.left + (self.content_width - text_width) / 2
            elif align == 2:
                text_width = pymupdf.get_text_length(
                    line, fontname=font, fontsize=size
                )
                position_x = self.right - text_width
            else:
                position_x = x

            _draw_text(self.page, 
                (position_x, self.y + size),
                line,
                fontname=font,
                fontsize=size,
                color=color,
            )
            self.y += leading

        self.y += gap

    def write_bullet(self, text: str):
        text = _clean(text)
        if not text:
            return

        size = self.styles["body_size"]
        font = self.styles["body_font"]
        color = self.styles["body_color"]
        indent = max(12.0, size * 1.5)

        lines = self._wrap(
            text,
            font,
            size,
            self.content_width - indent,
        )

        leading = size * 1.3

        for index, line in enumerate(lines):
            self.ensure_space(leading + 2)

            if index == 0:
                self.page.draw_circle(
                    (self.left + 3, self.y + size * 0.55),
                    1.2,
                    color=color,
                    fill=color,
                )

            _draw_text(self.page, 
                (self.left + indent, self.y + size),
                line,
                fontname=font,
                fontsize=size,
                color=color,
            )
            self.y += leading

        self.y += 2

    def write_split_line(
        self,
        left_text: str,
        right_text: str = "",
        *,
        size: float | None = None,
        font: str | None = None,
        color: tuple | None = None,
        gap: float = 1,
    ):
        left_text = _clean(left_text)
        right_text = _clean(right_text)

        if not left_text and not right_text:
            return

        size = size or self.styles["body_size"]
        font = font or self.styles["body_font"]
        color = color or self.styles["body_color"]

        leading = size * 1.3
        column_gap = 8.0

        right_width = (
            pymupdf.get_text_length(
                right_text,
                fontname=font,
                fontsize=size,
            )
            if right_text else 0
        )

        if right_width >= self.content_width:
            if left_text:
                self.write_text(
                    left_text,
                    size=size,
                    font=font,
                    color=color,
                    gap=0,
                )
            self.write_text(
                right_text,
                size=size,
                font=font,
                color=color,
                align=2,
                gap=gap,
            )
            return

        left_width = (
            self.content_width - right_width - column_gap
            if right_text else self.content_width
        )

        lines = self._wrap(left_text, font, size, left_width)

        if not lines:
            lines = [""]

        self.ensure_space(leading * len(lines) + gap)

        for index, line in enumerate(lines):
            _draw_text(self.page, 
                (self.left, self.y + size),
                line,
                fontname=font,
                fontsize=size,
                color=color,
            )

            if index == 0 and right_text:
                _draw_text(self.page, 
                    (
                        self.right - right_width,
                        self.y + size,
                    ),
                    right_text,
                    fontname=font,
                    fontsize=size,
                    color=color,
                )

            self.y += leading

        self.y += gap
    def write_section(self, title: str):
        self.ensure_space(28)

        self.y += self.styles["section_spacing"]

        self.write_text(
            title.upper(),
            size=self.styles["heading_size"],
            font=self.styles["heading_font"],
            color=self.styles["heading_color"],
            gap=2,
        )

        self.page.draw_line(
            (self.left, self.y),
            (self.right, self.y),
            color=self.styles["heading_color"],
            width=0.55,
        )
        self.y += 6


class SingleColumnRenderer(BaseResumeRenderer):
    """
    Configurable single-column resume renderer.

    Uses page geometry and typography metadata from
    resume_layout_analyzer.py.
    """

    def _get_page_metadata(self, metadata: dict) -> dict:
        pages = metadata.get("pages") or []

        if not pages:
            raise ValueError("Page layout metadata is missing")

        return pages[0]

    def _get_styles(self, page_metadata: dict) -> dict:
        fonts = page_metadata.get("dominant_fonts", [])
        sizes = page_metadata.get("dominant_font_sizes", [])
        colors = page_metadata.get("dominant_text_colors", [])

        body_size = DEFAULTS["body_size"]

        # Use the most common small text size as the body-size estimate.
        valid_sizes = [
            float(size) for size in sizes
            if isinstance(size, (int, float)) and 7 <= float(size) <= 12
        ]
        if valid_sizes:
            body_size = min(valid_sizes, key=lambda size: abs(size - 9))

        margins = page_metadata.get("margins_points", {})

        heading_color = DEFAULTS["heading_color"]
        if colors:
            non_black = [
                color for color in colors
                if color.upper() not in ("#000000", "#FFFFFF")
            ]
            if non_black:
                heading_color = non_black[0]

        return {
            "body_font": _font_name(fonts, "regular"),
            "heading_font": _font_name(fonts, "bold"),
            "body_size": body_size,
            "heading_size": max(10.0, body_size + 2),
            "name_size": max(18.0, body_size + 12),
            "body_color": _color(
                "#000000", DEFAULTS["body_color"]
            ),
            "heading_color": _color(
                heading_color, DEFAULTS["heading_color"]
            ),
            "section_spacing": 5.0,
            "margins": {
                "left": float(margins.get("left", DEFAULTS["margin"])),
                "right": float(margins.get("right", DEFAULTS["margin"])),
                "top": float(margins.get("top", DEFAULTS["margin"])),
                "bottom": float(margins.get("bottom", DEFAULTS["margin"])),
            },
        }

    @staticmethod
    def _entry_label(entry: Any, keys: tuple[str, ...]) -> str:
        if not isinstance(entry, dict):
            return _clean(entry)

        return " | ".join(
            _clean(entry.get(key))
            for key in keys
            if _clean(entry.get(key))
        )

    def render(
        self,
        profile: dict[str, Any],
        draft: dict[str, Any],
        layout_metadata: dict[str, Any],
    ) -> bytes:
        if layout_metadata.get("layout_type") != "single_column":
            raise ValueError("SingleColumnRenderer requires single_column layout")

        page_metadata = self._get_page_metadata(layout_metadata)

        page_width = float(page_metadata["width"])
        page_height = float(page_metadata["height"])

        styles = self._get_styles(page_metadata)

        writer = _SingleColumnWriter(
            page_width,
            page_height,
            styles.pop("margins"),
            styles,
        )

        # Candidate name and contact information.
        name = _clean(profile.get("name")) or "Candidate"

        writer.write_text(
            name,
            size=styles["name_size"],
            font=styles["heading_font"],
            color=styles["body_color"],
            align=1,
            gap=2,
        )

        contacts = [
            _clean(profile.get(key))
            for key in ("phone", "email", "linkedin", "github", "portfolio")
            if _clean(profile.get(key))
        ]

        if contacts:
            writer.write_text(
                " | ".join(contacts),
                size=styles["body_size"],
                color=styles["body_color"],
                align=1,
                gap=5,
            )

        if draft.get("headline"):
            writer.write_text(
                draft["headline"],
                size=styles["body_size"],
                font=styles["heading_font"],
                color=styles["heading_color"],
                align=1,
                gap=4,
            )

        if draft.get("summary"):
            writer.write_section("Summary")
            writer.write_text(draft["summary"], gap=2)

        # Professional experience.
        experience = _as_list(draft.get("experience"))

        if experience:
            writer.write_section("Experience")

            for item in experience:
                if isinstance(item, dict):
                    company = _clean(item.get("company"))
                    job_title = _clean(item.get("job_title"))
                    location = _clean(item.get("location"))
                    duration = _clean(item.get("duration"))

                    writer.write_split_line(
                        company,
                        location,
                        size=styles["body_size"],
                        font=styles["heading_font"],
                        color=styles["body_color"],
                        gap=1,
                    )

                    # Job title on the left, duration on the right.
                    writer.write_split_line(
                        job_title,
                        duration,
                        size=styles["body_size"],
                        font="heit",
                        color=styles["body_color"],
                        gap=2,
                    )

                    for bullet in _as_list(item.get("bullets")):
                        writer.write_bullet(_clean(bullet))
                else:
                    writer.write_bullet(_clean(item))

        # Achievements.
        achievements = _as_list(draft.get("achievements"))

        if achievements:
            writer.write_section("Achievements")

            for item in achievements:
                label = (
                    self._entry_label(item, ("name", "value"))
                    if isinstance(item, dict)
                    else _clean(item)
                )
                writer.write_bullet(label)

        # Projects.
        projects = _as_list(draft.get("projects"))

        if projects:
            writer.write_section("Projects")

            for item in projects:
                if isinstance(item, dict):
                    writer.write_text(
                        _clean(item.get("name")),
                        font=styles["heading_font"],
                        gap=1,
                    )

                    technologies = item.get("technologies")
                    if technologies:
                        tech_text = (
                            ", ".join(map(_clean, technologies))
                            if isinstance(technologies, list)
                            else _clean(technologies)
                        )
                        writer.write_text(
                            tech_text,
                            size=max(7.5, styles["body_size"] - 0.5),
                            gap=2,
                        )

                    if item.get("description"):
                        writer.write_text(item["description"], gap=2)

                    for bullet in _as_list(item.get("bullets")):
                        writer.write_bullet(_clean(bullet))
                else:
                    writer.write_bullet(_clean(item))

        # Technical skills.
        skills = draft.get("skills")

        if skills:
            writer.write_section("Technical Skills")

            if isinstance(skills, dict):
                for category, values in skills.items():
                    value_text = (
                        ", ".join(map(_clean, values))
                        if isinstance(values, list)
                        else _clean(values)
                    )

                    if value_text:
                        writer.write_text(
                            f"{category.replace('_', ' ').title()}: {value_text}"
                        )
            else:
                skill_text = (
                    ", ".join(map(_clean, skills))
                    if isinstance(skills, list)
                    else _clean(skills)
                )
                writer.write_text(skill_text)

        # Education.
        education = _as_list(draft.get("education"))

        if education:
            writer.write_section("Education")

            for item in education:
                if isinstance(item, dict):
                    parts = [
                        _clean(value)
                        for value in item.values()
                        if value and not isinstance(value, (list, dict))
                    ]
                    writer.write_text(" | ".join(parts))
                else:
                    writer.write_text(_clean(item))

        # Generate and validate the PDF.
        pdf_bytes = writer.doc.tobytes(
            garbage=4,
            deflate=True,
        )
        writer.doc.close()

        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as check:
            if check.page_count == 0:
                raise ValueError("Generated resume PDF has no pages")

            if not any(page.get_text().strip() for page in check):
                raise ValueError("Generated resume PDF contains no text")

            if check.page_count > 3:
                raise ValueError(
                    "Generated resume exceeds 3 pages. "
                    "Review the resume content before generating again."
                )

        return pdf_bytes
