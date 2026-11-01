"""Heuristic PDF resume layout analysis using PyMuPDF.

This module classifies uploaded resumes as single-column, two-column, or
complex/unknown. It also extracts design metadata useful to a later renderer.
It does not modify the source PDF or guarantee semantic section detection.
"""
from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Any

import pymupdf

SUPPORTED_LAYOUTS = {"single_column", "two_column", "complex"}


def _safe_round(value: float, digits: int = 2) -> float:
    return round(float(value), digits)


def _page_layout(page: pymupdf.Page) -> dict[str, Any]:
    page_rect = page.rect
    data = page.get_text("dict", sort=True)
    blocks: list[dict[str, Any]] = []
    font_counter: Counter[str] = Counter()
    size_counter: Counter[float] = Counter()
    color_counter: Counter[str] = Counter()
    text_blocks = 0

    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        lines_out: list[str] = []
        block_fonts: list[str] = []
        block_sizes: list[float] = []
        block_colors: list[str] = []
        for line in block.get("lines", []):
            line_text = "".join(span.get("text", "") for span in line.get("spans", []))
            if line_text.strip():
                lines_out.append(line_text.strip())
            for span in line.get("spans", []):
                if not span.get("text", "").strip():
                    continue
                font = span.get("font")
                size = span.get("size")
                color = span.get("color")
                if font:
                    block_fonts.append(font)
                    font_counter[font] += len(span.get("text", ""))
                if isinstance(size, (int, float)):
                    block_sizes.append(float(size))
                    size_counter[_safe_round(size, 1)] += len(span.get("text", ""))
                if isinstance(color, int):
                    hex_color = f"#{color:06x}"
                    block_colors.append(hex_color)
                    color_counter[hex_color] += len(span.get("text", ""))
        text = " ".join(lines_out).strip()
        if not text:
            continue
        bbox = block.get("bbox", (0, 0, 0, 0))
        x0, y0, x1, y1 = map(float, bbox)
        blocks.append({
            "text": text,
            "bbox": [_safe_round(x0), _safe_round(y0), _safe_round(x1), _safe_round(y1)],
            "center_x": _safe_round((x0 + x1) / 2),
            "width": _safe_round(max(0, x1 - x0)),
            "height": _safe_round(max(0, y1 - y0)),
            "fonts": sorted(set(block_fonts)),
            "font_sizes": sorted(set(_safe_round(s, 1) for s in block_sizes)),
            "colors": sorted(set(block_colors)),
        })
        text_blocks += 1

    # Detect columns from text block centers, ignoring narrow blocks near the page edge.
    # A two-column page normally has substantial text on both sides and a clear gutter.
    page_mid = page_rect.width / 2
    left = [b for b in blocks if b["center_x"] < page_mid * 0.88 and b["width"] > page_rect.width * 0.12]
    right = [b for b in blocks if b["center_x"] > page_mid * 1.12 and b["width"] > page_rect.width * 0.12]
    left_chars = sum(len(b["text"]) for b in left)
    right_chars = sum(len(b["text"]) for b in right)
    total_chars = max(1, left_chars + right_chars)
    balance = min(left_chars, right_chars) / max(left_chars, right_chars, 1)

    # Require multiple blocks in each region and meaningful content on both sides.
    two_column = (
        len(left) >= 2
        and len(right) >= 2
        and left_chars >= 80
        and right_chars >= 80
        and balance >= 0.12
    )

    # A page with text/images that cannot be confidently represented by these
    # simple column heuristics is marked complex for explicit fallback handling.
    image_count = len(page.get_images(full=True))
    page_area = max(1.0, page_rect.width * page_rect.height)
    text_area = sum(b["width"] * b["height"] for b in blocks)
    text_coverage = min(1.0, text_area / page_area)
    if two_column:
        layout = "two_column"
        confidence = min(0.98, 0.65 + 0.25 * balance + 0.08 * min(1, len(left + right) / 12))
    elif len(blocks) >= 2:
        # Detect obvious nonstandard layouts: substantial imagery, many narrow
        # floating blocks, or a page split with too little evidence for columns.
        narrow_ratio = sum(1 for b in blocks if b["width"] < page_rect.width * 0.22) / len(blocks)
        if image_count > 2 or narrow_ratio > 0.45:
            layout, confidence = "complex", 0.55
        else:
            layout, confidence = "single_column", 0.78
    else:
        layout, confidence = "complex", 0.35

    margins = {
        "left": _safe_round(min((b["bbox"][0] for b in blocks), default=0)),
        "right": _safe_round(page_rect.width - max((b["bbox"][2] for b in blocks), default=page_rect.width)),
        "top": _safe_round(min((b["bbox"][1] for b in blocks), default=0)),
        "bottom": _safe_round(page_rect.height - max((b["bbox"][3] for b in blocks), default=page_rect.height)),
    }
    return {
        "page_number": page.number + 1,
        "width": _safe_round(page_rect.width),
        "height": _safe_round(page_rect.height),
        "rotation": page.rotation,
        "layout_type": layout,
        "confidence": _safe_round(confidence, 3),
        "text_block_count": text_blocks,
        "image_count": image_count,
        "text_coverage_estimate": _safe_round(text_coverage, 3),
        "margins_points": margins,
        "column_evidence": {
            "left_block_count": len(left),
            "right_block_count": len(right),
            "left_character_count": left_chars,
            "right_character_count": right_chars,
            "balance": _safe_round(balance, 3),
        },
        "dominant_fonts": [name for name, _ in font_counter.most_common(5)],
        "dominant_font_sizes": [size for size, _ in size_counter.most_common(6)],
        "dominant_text_colors": [color for color, _ in color_counter.most_common(5)],
        "blocks": blocks,
    }

def analyze_resume_layout(pdf_bytes: bytes) -> dict[str, Any]:
    """Analyse PDF geometry and typography; return JSON-serializable metadata."""
    if not isinstance(pdf_bytes, (bytes, bytearray)) or not pdf_bytes.startswith(b"%PDF"):
        raise ValueError("Input is not a valid PDF byte stream")
    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
            if doc.is_encrypted:
                raise ValueError("Password-protected PDFs are not supported yet")
            if doc.page_count == 0:
                raise ValueError("PDF contains no pages")
            pages = [_page_layout(page) for page in doc]
            page_types = [p["layout_type"] for p in pages]
            if "complex" in page_types:
                overall = "complex"
            elif "two_column" in page_types:
                overall = "two_column"
            else:
                overall = "single_column"
            confidence = min((p["confidence"] for p in pages), default=0.0)
            return {
                "layout_type": overall,
                "confidence": confidence,
                "page_count": doc.page_count,
                "page_size": "A4" if all(abs(p["width"] - 595.28) < 3 and abs(p["height"] - 841.89) < 3 for p in pages) else "custom",
                "pages": pages,
                "recommended_action": (
                    "preserve_layout" if overall in {"single_column", "two_column"} and confidence >= 0.65
                    else "request_template_choice"
                ),
                "limitations": [
                    "Column detection is heuristic and should be confirmed with a rendered preview.",
                    "Text extraction does not reliably identify semantic section roles or reproduce arbitrary graphics.",
                    "Scanned PDFs may require OCR before layout analysis."
                ],
            }
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not analyse resume PDF: {exc}") from exc
