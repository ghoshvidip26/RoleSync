
"""Extract a detailed, JSON-serializable layout manifest from a resume PDF.

This module is independent of the existing PDF renderer.
It does not modify the source PDF or any existing project files.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import pymupdf

from .resume_layout_analyzer import analyze_resume_layout


SECTION_NAMES = {
    "summary": "Summary",
    "professional summary": "Summary",
    "profile": "Summary",
    "objective": "Summary",
    "technical skills": "Technical Skills",
    "skills": "Technical Skills",
    "experience": "Experience",
    "work experience": "Experience",
    "professional experience": "Experience",
    "projects": "Projects",
    "personal projects": "Projects",
    "achievements": "Achievements",
    "accomplishments": "Achievements",
    "education": "Education",
    "certifications": "Certifications",
    "publications": "Publications",
}

BULLET_PATTERN = re.compile(r"^[•●▪◦‣⁃*-]\s*")


def _round(value: float, digits: int = 2) -> float:
    return round(float(value), digits)


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _detect_section_heading(line: dict[str, Any]) -> str | None:
    """Identify common resume section headings conservatively."""

    text = _normalize_text(line["text"]).lower().rstrip(":")
    return SECTION_NAMES.get(text)


def _font_style(flags: int, font_name: str) -> dict[str, bool]:
    """Infer bold and italic styles from PDF font metadata."""

    name = font_name.lower()

    return {
        "bold": bool(flags & 16) or "bold" in name,
        "italic": bool(flags & 2) or "italic" in name,
    }


def _extract_page(page: pymupdf.Page) -> dict[str, Any]:
    """Extract spans, lines and text blocks without flattening formatting."""

    page_rect = page.rect
    page_data = page.get_text("dict", sort=True)

    spans_out: list[dict[str, Any]] = []
    lines_out: list[dict[str, Any]] = []
    blocks_out: list[dict[str, Any]] = []

    for block_index, block in enumerate(page_data.get("blocks", [])):
        if block.get("type") != 0:
            continue

        block_lines = []

        for line_index, line in enumerate(block.get("lines", [])):
            extracted_spans = []

            for span_index, span in enumerate(line.get("spans", [])):
                text = span.get("text", "")

                if not text.strip():
                    continue

                font_name = span.get("font", "unknown")
                flags = int(span.get("flags", 0))
                bbox = [_round(v) for v in span["bbox"]]
                color_value = span.get("color", 0)

                span_item = {
                    "id": f"p{page.number + 1}_b{block_index}_l{line_index}_s{span_index}",
                    "text": text,
                    "bbox": bbox,
                    "font": font_name,
                    "font_size": _round(span.get("size", 0)),
                    "flags": flags,
                    "style": _font_style(flags, font_name),
                    "color": f"#{color_value:06x}",
                    "origin": [
                        _round(span.get("origin", (0, 0))[0]),
                        _round(span.get("origin", (0, 0))[1]),
                    ],
                }

                spans_out.append(span_item)
                extracted_spans.append(span_item)

            if not extracted_spans:
                continue

            line_bbox = [
                _round(min(s["bbox"][0] for s in extracted_spans)),
                _round(min(s["bbox"][1] for s in extracted_spans)),
                _round(max(s["bbox"][2] for s in extracted_spans)),
                _round(max(s["bbox"][3] for s in extracted_spans)),
            ]

            # line_text = _normalize_text(
            #     "".join(s["text"] for s in extracted_spans)
            # )
            line_text = _normalize_text(
                " ".join(s["text"] for s in extracted_spans)
            )

            line_item = {
                "id": f"p{page.number + 1}_b{block_index}_l{line_index}",
                "text": line_text,
                "bbox": line_bbox,
                "spans": [s["id"] for s in extracted_spans],
                "font_families": list(
                    dict.fromkeys(s["font"] for s in extracted_spans)
                ),
                "font_sizes": sorted(
                    set(s["font_size"] for s in extracted_spans)
                ),
                "colors": list(
                    dict.fromkeys(s["color"] for s in extracted_spans)
                ),
                "is_bullet": bool(BULLET_PATTERN.match(line_text)),
                "alignment": "unknown",
            }

            lines_out.append(line_item)
            block_lines.append(line_item)

        if not block_lines:
            continue

        block_bbox = [
            _round(min(line["bbox"][0] for line in block_lines)),
            _round(min(line["bbox"][1] for line in block_lines)),
            _round(max(line["bbox"][2] for line in block_lines)),
            _round(max(line["bbox"][3] for line in block_lines)),
        ]

        blocks_out.append({
            "id": f"p{page.number + 1}_b{block_index}",
            "bbox": block_bbox,
            "text": " ".join(line["text"] for line in block_lines),
            "line_ids": [line["id"] for line in block_lines],
        })

   
    # Estimate alignment using the document's actual text boundaries.
    # Left/right anchors are inferred from the extracted content.
    all_bboxes = [line["bbox"] for line in lines_out]

    left_anchor = min(
        (bbox[0] for bbox in all_bboxes),
        default=0.0,
    )
    right_anchor = max(
        (bbox[2] for bbox in all_bboxes),
        default=page_rect.width,
    )

    for line in lines_out:
        x0, _, x1, _ = line["bbox"]
        center_x = (x0 + x1) / 2
        line_width = x1 - x0

        if abs(x0 - left_anchor) <= 8:
            line["alignment"] = "left"

        # Right-aligned items (dates, locations) are short and start in the
        # right half; a wrapped bullet that merely reaches the right margin
        # starts in the left half and is not right-aligned.
        elif abs(x1 - right_anchor) <= 8 and x0 >= page_rect.width / 2:
            line["alignment"] = "right"

        # Centered text is narrow; a wide wrapped line can only coincidentally
        # have its middle near the page center.
        elif (
            line_width < page_rect.width * 0.70
            and abs(center_x - page_rect.width / 2) <= 12
        ):
            line["alignment"] = "center"

        else:
            line["alignment"] = "indented_left"

    # for line in lines_out:
    #     x0, _, x1, _ = line["bbox"]
    #     center_x = (x0 + x1) / 2

    #     if abs(center_x - page_rect.width / 2) < 12:
    #         line["alignment"] = "center"
    #     elif x1 >= page_rect.width - 35 and x0 > page_rect.width / 2:
    #         line["alignment"] = "right"
    #     else:
    #         line["alignment"] = "left"

    return {
        "page_number": page.number + 1,
        "dimensions": {
            "width": _round(page_rect.width),
            "height": _round(page_rect.height),
            "rotation": page.rotation,
        },
        "spans": spans_out,
        "lines": lines_out,
        "blocks": blocks_out,
    }


def _build_sections(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group lines under detected section headings.

    Lines before the first recognized heading belong to the header.
    """

    sections: list[dict[str, Any]] = []
    current_section: dict[str, Any] = {
        "name": "Header",
        "heading_line_id": None,
        "lines": [],
    }

    for page in pages:
        for line in page["lines"]:
            heading = _detect_section_heading(line)

            if heading:
                if current_section["lines"] or current_section["heading_line_id"]:
                    sections.append(current_section)

                current_section = {
                    "name": heading,
                    "heading_line_id": line["id"],
                    "lines": [],
                }
            else:
                current_section["lines"].append(line["id"])

        # Page boundaries are retained on the line records.
        # Sections can continue across pages.

    if current_section["lines"] or current_section["heading_line_id"]:
        sections.append(current_section)

    return sections


def _build_style_summary(pages: list[dict[str, Any]]) -> dict[str, Any]:
    fonts = Counter()
    font_sizes = Counter()
    colors = Counter()

    for page in pages:
        for span in page["spans"]:
            text_length = len(span["text"].strip())

            fonts[span["font"]] += text_length
            font_sizes[span["font_size"]] += text_length
            colors[span["color"]] += text_length

    return {
        "dominant_fonts": [
            {"font": name, "character_count": count}
            for name, count in fonts.most_common()
        ],
        "dominant_font_sizes": [
            {"size": size, "character_count": count}
            for size, count in font_sizes.most_common()
        ],
        "dominant_colors": [
            {"color": color, "character_count": count}
            for color, count in colors.most_common()
        ],
    }


def build_layout_manifest(
    pdf_bytes: bytes,
    layout_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a detailed manifest from the original PDF bytes."""

    if not isinstance(pdf_bytes, (bytes, bytearray)):
        raise ValueError("PDF input must be bytes")

    if not pdf_bytes.startswith(b"%PDF"):
        raise ValueError("Input is not a valid PDF byte stream")

    if layout_metadata is None:
        layout_metadata = analyze_resume_layout(pdf_bytes)

    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
            if doc.is_encrypted:
                raise ValueError(
                    "Password-protected PDFs are not supported yet"
                )

            if doc.page_count == 0:
                raise ValueError("PDF contains no pages")

            pages = [_extract_page(page) for page in doc]

            line_lookup = {
                line["id"]: line
                for page in pages
                for line in page["lines"]
            }

            sections = _build_sections(pages)

            # Attach page and vertical-position information to sections.
            for section in sections:
                heading_id = section["heading_line_id"]
                heading = line_lookup.get(heading_id) if heading_id else None

                section["heading_text"] = (
                    heading["text"] if heading else None
                )
                section["start_page"] = (
                    int(heading_id.split("_")[0][1:])
                    if heading_id else 1
                )

            return {
                "manifest_version": "1.0",
                "document": {
                    "page_count": doc.page_count,
                    "layout_type": layout_metadata.get("layout_type"),
                    "page_size": layout_metadata.get("page_size"),
                    "confidence": layout_metadata.get("confidence"),
                },
                "style_summary": _build_style_summary(pages),
                "sections": sections,
                "pages": pages,
                "limitations": [
                    "Section detection is heuristic.",
                    "Alignment is estimated from text coordinates.",
                    "PDF font names do not guarantee that font programs "
                    "can be extracted or embedded.",
                    "The manifest records geometry but does not itself "
                    "perform text replacement or reflow.",
                ],
            }

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            f"Could not build resume layout manifest: {exc}"
        ) from exc
