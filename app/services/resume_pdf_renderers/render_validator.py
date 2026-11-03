"""Phase 7: validate a generated resume PDF against the original and the approved draft.

``validate_rendering`` never trusts that "a PDF came out". It re-reads the generated
file the way an ATS or a person would and reports each check as ``pass``, ``warn`` or
``fail`` (``fail`` means do not send it):

content   every approved text is present; nothing unreadable; sections in order;
          contact details and hyperlinks survived
geometry  no overlapping text; nothing outside the page or the original margins;
          lines not cramped
design    alignment, bullet glyph, section rules and margins still match the
          original; fonts and text size reported when they drifted
visual    an ink-layout similarity score, plus ``comparison_image`` for a person to look at

It reads PDFs only and changes no file.
"""

from __future__ import annotations

import re
from typing import Any

import pymupdf

from .content_mapper import map_content
from .entry_grouper import _geometry_text, group_entries
from .layout_manifest import build_layout_manifest
from .layout_representation import build_layout_representation

MIN_TEXT_SCALE = 0.90           # text smaller than this share of the original size is a warning
MIN_LEADING = 1.0               # line pitch below this multiple of the font size is cramped
POSITION_TOLERANCE = 4.0        # pt a kind of line may move before it counts as design drift
EDGE_TOLERANCE = 3.0            # pt beyond the original margins before text counts as outside
OVERLAP_SHARE = 0.20            # share of the smaller word box that must be covered to count
SIMILARITY_WARN = 0.50      # tailored content changes the ink pattern; this only catches a different layout

_BAD_CHARS = {"\ufffd": "replacement character", "\u00ad": "soft hyphen", "\u2010": "unicode hyphen",
              "\u2011": "non-breaking hyphen", "\u00a0": "non-breaking space",
              "\u202f": "narrow no-break space", "\x00": "null"}
# only used to compare approved text with extracted text; the extraction_clean check reports these characters
_DASH_FIX = str.maketrans({"\u2010": "-", "\u2011": "-", "\u00ad": "-", "\u00a0": " ", "\u202f": " "})


def _n(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").translate(_DASH_FIX)).strip()


def _check(name: str, status: str, detail: str, **evidence) -> dict:
    return {"name": name, "status": status, "detail": detail, "evidence": evidence}


# ------------------------------------------------------------------ expectations

def expected_texts(plan: dict, lines: dict[str, dict], spans: dict[str, dict] | None = None) -> list[str]:
    """Every string the generated PDF must contain, taken from the render plan."""
    out: list[str] = []

    def add(text):
        text = _n(text)
        if len(text) >= 2:
            out.append(text)

    def copied(ids):
        for i in ids:
            if i in lines:                           # text from span geometry, not the manifest's blanket-space join
                add(_geometry_text(lines[i], spans) if spans else lines[i]["text"])

    h = plan["header"]
    add((h.get("name") or {}).get("text"))
    for c in h.get("contacts", []) + h.get("other", []):
        add(c["text"])
    if h.get("headline"):
        add(h["headline"]["text"])
    for sec in plan["sections"]:
        add(sec["heading"]["text"])
        c = sec["content"]
        if "paragraph" in c:
            add(c["paragraph"]["text"])
        for g in c.get("groups", []):
            add(g["label"])
            add(g["values"]["text"])
        for co in c.get("companies", []):
            add(co["company"]["text"])
            add(co["location"]["text"])
            for r in co["roles"]:
                add(r["title"]["text"])
                add(r["duration"]["text"])
                copied(i for d in r.get("details", []) for i in d["line_ids"])
                for b in r["bullets"]:
                    add(b["text"])
        for p in c.get("projects", []):
            add(p["name"]["text"])
            add(p["technologies"]["text"])
            add(p["dates"]["text"])
            for d in p["description"]:
                add(d["text"])
            for b in p["bullets"]:
                add(b["text"])
        for b in c.get("bullets", []) or []:
            add(b["text"])
        for e in c.get("entries", []) or []:
            if "institution" in e:
                copied(e["line_ids"])
            else:
                copied((e.get("head") or {}).get("line_ids", []))
                copied(i for it in e.get("items", []) for i in it["line_ids"])
    return out


# ---------------------------------------------------------------------- geometry

def _words(doc: pymupdf.Document) -> list[list[tuple]]:
    return [[tuple(w) for w in page.get_text("words")] for page in doc]


def _overlaps(words: list[tuple]) -> list[tuple[str, str]]:
    found = []
    # a lone "|" or bullet glyph has a tall font box but little ink; it cannot collide with text
    ordered = sorted((w for w in words if any(c.isalnum() for c in w[4])), key=lambda w: w[1])
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            if b[1] >= a[3]:
                break
            if (a[5], a[6]) == (b[5], b[6]):          # same text line
                continue
            ix = min(a[2], b[2]) - max(a[0], b[0])
            iy = min(a[3], b[3]) - max(a[1], b[1])
            if ix <= 0 or iy <= 0:
                continue
            smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
            if smaller > 0 and ix * iy / smaller > OVERLAP_SHARE:
                found.append((a[4], b[4]))
    return found


def _baselines(page: pymupdf.Page) -> list[tuple[float, float]]:
    """(baseline, size) of every line."""
    out = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = [s for s in line["spans"] if s["text"].strip()]
            if spans:
                out.append((round(spans[0]["origin"][1], 1), max(s["size"] for s in spans)))
    return out


def _role_positions(manifest: dict, rep: dict) -> dict[str, tuple[str, float]]:
    """role -> (alignment, median x of the edge that defines it)."""
    from statistics import median
    lines = {l["id"]: l for p in manifest["pages"] for l in p["lines"]}
    by_role: dict[str, list] = {}
    for line_id, role in rep["role_of_line"].items():
        if line_id in lines:
            by_role.setdefault(role, []).append(lines[line_id]["bbox"])
    out = {}
    for role, boxes in by_role.items():
        # text extraction may split one row into several lines; compare whole rows
        merged: list[list[float]] = []
        for b in sorted(boxes, key=lambda b: b[1]):
            if merged and min(merged[-1][3], b[3]) - max(merged[-1][1], b[1]) > 0.5 * (b[3] - b[1]):
                merged[-1] = [min(merged[-1][0], b[0]), min(merged[-1][1], b[1]),
                              max(merged[-1][2], b[2]), max(merged[-1][3], b[3])]
            else:
                merged.append(list(b))
        boxes = merged
        align = (rep["typography"].get(role) or {}).get("alignment", "left")
        if role.endswith("_right"):
            align = "right"
        if align == "right":
            out[role] = (align, median(b[2] for b in boxes))
        elif align == "center":
            out[role] = (align, median((b[0] + b[2]) / 2 for b in boxes))
        else:
            out[role] = (align, median(b[0] for b in boxes))
    return out


# ------------------------------------------------------------------------ visual

def _ink_grid(page: pymupdf.Page, cols: int = 24, rows: int = 32) -> list[float]:
    pix = page.get_pixmap(dpi=40, colorspace=pymupdf.csGRAY)
    w, h, data = pix.width, pix.height, pix.samples
    grid = [0.0] * (cols * rows)
    count = [0] * (cols * rows)
    for y in range(h):
        gy = min(rows - 1, y * rows // h)
        base = y * w
        for x in range(w):
            gx = min(cols - 1, x * cols // w)
            grid[gy * cols + gx] += 255 - data[base + x]
            count[gy * cols + gx] += 1
    return [g / max(1, c) for g, c in zip(grid, count)]


def _correlation(a: list[float], b: list[float]) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va, vb = sum((x - ma) ** 2 for x in a), sum((y - mb) ** 2 for y in b)
    return cov / ((va * vb) ** 0.5) if va > 0 and vb > 0 else 0.0


def visual_similarity(original: pymupdf.Document, generated: pymupdf.Document) -> float:
    """Average correlation of the ink layout of the pages both files have (1.0 = same layout)."""
    pairs = min(original.page_count, generated.page_count)
    if not pairs:
        return 0.0
    scores = [_correlation(_ink_grid(original[i]), _ink_grid(generated[i])) for i in range(pairs)]
    return sum(scores) / len(scores)


def comparison_image(original_pdf: bytes, generated_pdf: bytes, dpi: int = 80) -> bytes:
    """PNG with the original pages on the left and the generated pages on the right."""
    with pymupdf.open(stream=original_pdf, filetype="pdf") as a, pymupdf.open(stream=generated_pdf, filetype="pdf") as b:
        pages = max(a.page_count, b.page_count)
        w, h = a[0].rect.width, a[0].rect.height
        out = pymupdf.open()
        gap = 12
        sheet = out.new_page(width=w * 2 + gap, height=h * pages + gap * (pages - 1))
        for i in range(pages):
            for col, src in ((0, a), (1, b)):
                if i < src.page_count:
                    top = i * (h + gap)
                    sheet.show_pdf_page(pymupdf.Rect(col * (w + gap), top, col * (w + gap) + w, top + h), src, i)
                    sheet.draw_rect(pymupdf.Rect(col * (w + gap), top, col * (w + gap) + w, top + h),
                                    color=(0.7, 0.7, 0.7), width=0.5)
        png = sheet.get_pixmap(dpi=dpi).tobytes("png")
        out.close()
        return png


# -------------------------------------------------------------------------- main

def validate_rendering(
    original_pdf: bytes,
    generated_pdf: bytes,
    draft: dict[str, Any] | None = None,
    plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check ``generated_pdf`` (made from ``original_pdf`` and the approved ``draft``/``plan``)."""
    for name, blob in (("original_pdf", original_pdf), ("generated_pdf", generated_pdf)):
        if not isinstance(blob, (bytes, bytearray)) or not blob.startswith(b"%PDF"):
            raise ValueError(f"{name} must be PDF bytes")
    if plan is None and draft is None:
        raise ValueError("Provide the approved draft or its render plan")

    manifest = build_layout_manifest(bytes(original_pdf))
    grouped = group_entries(manifest)
    if plan is None:
        plan = map_content(grouped, draft)
    lines = {l["id"]: l for p in manifest["pages"] for l in p["lines"]}
    spans = {s["id"]: s for p in manifest["pages"] for s in p["spans"]}
    orig_rep = build_layout_representation(bytes(original_pdf), manifest, grouped)

    checks: list[dict] = []
    metrics: dict[str, Any] = {}
    with pymupdf.open(stream=bytes(original_pdf), filetype="pdf") as orig, \
            pymupdf.open(stream=bytes(generated_pdf), filetype="pdf") as gen:

        # ---- pages
        if gen.page_count == 0:
            return {"passed": False, "checks": [_check("pages", "fail", "The PDF has no pages")], "metrics": {}}
        extra = gen.page_count - orig.page_count
        checks.append(_check("page_count", "pass" if extra <= 0 else "warn",
                             f"{gen.page_count} page(s); the original has {orig.page_count}",
                             original=orig.page_count, generated=gen.page_count))
        same_size = all(abs(g.rect.width - orig[0].rect.width) < 1 and abs(g.rect.height - orig[0].rect.height) < 1
                        for g in gen)
        checks.append(_check("page_size", "pass" if same_size else "fail",
                             "Page size matches the original" if same_size else "Page size differs from the original"))

        full_text = "\n".join(p.get_text() for p in gen)
        flat = _n(full_text)

        # ---- content
        missing = [t for t in expected_texts(plan, lines, spans) if t not in flat]
        checks.append(_check("text_complete", "pass" if not missing else "fail",
                             "Every approved text is in the PDF" if not missing
                             else f"{len(missing)} approved text(s) are missing", missing=missing[:8]))

        bad = {name: full_text.count(ch) for ch, name in _BAD_CHARS.items() if ch in full_text}
        pua = sum(1 for c in full_text if "\ue000" <= c <= "\uf8ff")
        if pua:
            bad["private-use symbol"] = pua
        checks.append(_check("extraction_clean", "pass" if not bad else "fail",
                             "Text extracts as plain characters" if not bad
                             else "Some characters extract as symbols an ATS cannot read", found=bad))

        pos, out_of_order = 0, []
        for sec in plan["sections"]:
            i = flat.find(_n(sec["heading"]["text"]), pos)
            if i < 0:
                out_of_order.append(sec["heading"]["text"])
            else:
                pos = i + 1
        checks.append(_check("section_order", "pass" if not out_of_order else "fail",
                             "Sections appear in the original order" if not out_of_order
                             else "Section headings are missing or out of order", problems=out_of_order))

        # A link is required only while the words it sat on are still in the resume: removing a project on purpose
        # removes its link too, and that is not a loss.
        orig_links = []
        for p in orig:
            for l in p.get_links():
                if l.get("uri"):
                    anchor = _n(p.get_textbox(pymupdf.Rect(l["from"])))
                    if not anchor or anchor in flat:
                        orig_links.append(l["uri"])
        gen_links = [l["uri"] for p in gen for l in p.get_links() if l.get("uri")]
        lost = [u for u in orig_links if u not in gen_links]
        checks.append(_check("links_preserved", "pass" if not lost else "fail",
                             f"{len(gen_links)} hyperlink(s) kept" if not lost else "Hyperlinks were lost", lost=lost))

        # ---- geometry
        words = _words(gen)
        overlaps = [o for page in words for o in _overlaps(page)]
        checks.append(_check("no_overlap", "pass" if not overlaps else "fail",
                             "No text overlaps" if not overlaps else f"{len(overlaps)} overlapping word pair(s)",
                             samples=overlaps[:5]))

        left = orig_rep["pages"][0]["content_box"][0] - EDGE_TOLERANCE
        right = (orig_rep["pairs"]["right_edge_x"] or orig_rep["pages"][0]["content_box"][2]) + EDGE_TOLERANCE
        top_margin = orig_rep["pages"][0]["margins"]["top"]
        outside = []
        for n, page_words in enumerate(words, 1):
            h = gen[n - 1].rect.height
            for w in page_words:
                if w[0] < left or w[2] > right or w[3] > h - top_margin + EDGE_TOLERANCE or w[1] < 0:
                    outside.append((n, w[4], round(w[0], 1), round(w[1], 1)))
        checks.append(_check("inside_margins", "pass" if not outside else "fail",
                             "All text is inside the original margins" if not outside
                             else f"{len(outside)} word(s) outside the margins", samples=outside[:5]))

        cramped = []
        for n, page in enumerate(gen, 1):
            ls = sorted(_baselines(page))
            for (b1, s1), (b2, s2) in zip(ls, ls[1:]):
                if 0.5 < b2 - b1 < MIN_LEADING * s2 - 0.01 and b2 - b1 > 1.0:
                    cramped.append((n, b1, b2))
        checks.append(_check("line_spacing", "pass" if not cramped else "warn",
                             "Lines are not cramped" if not cramped else f"{len(cramped)} line(s) are very tight",
                             samples=cramped[:5]))

        # ---- design
        gen_manifest = build_layout_manifest(bytes(generated_pdf))
        gen_grouped = group_entries(gen_manifest)
        gen_rep = build_layout_representation(bytes(generated_pdf), gen_manifest, gen_grouped)
        drift = []
        if gen_rep["section_headings"]["order"] != [s["name"] for s in plan["sections"]]:
            drift.append("section headings differ from the plan")
        if bool(orig_rep["rules"]) and len([r for r in gen_rep["rules"] if r["kind"] == "separator"]) < \
                len([r for r in orig_rep["rules"] if r["kind"] == "separator"]) * 0.5:
            drift.append("section rules are missing")
        if (orig_rep["bullets"] or {}).get("marker") != (gen_rep["bullets"] or {}).get("marker"):
            drift.append("bullet glyph changed")
        # where each kind of line sits: left edge for left text, right edge for right-hand items, centre for centred
        a_pos, g_pos = _role_positions(manifest, orig_rep), _role_positions(gen_manifest, gen_rep)
        for role, (align, pos) in a_pos.items():
            if role in g_pos and abs(g_pos[role][1] - pos) > POSITION_TOLERANCE:
                drift.append(f"{role} ({align}) moved from {pos:.0f} to {g_pos[role][1]:.0f}")
        if abs(gen_rep["pages"][0]["margins"]["left"] - orig_rep["pages"][0]["margins"]["left"]) > 2:
            drift.append("left margin moved")
        checks.append(_check("design_match", "pass" if not drift else "warn",
                             "Alignment, bullets, rules and margins match the original" if not drift
                             else "The design differs from the original", differences=drift))

        ratios = {}
        for role, st in orig_rep["typography"].items():
            g = gen_rep["typography"].get(role)
            if g and st["size"]:
                ratios[role] = round(g["size"] / st["size"], 3)
        smallest = min(ratios.values(), default=1.0)
        metrics["size_ratio_by_role"] = ratios
        checks.append(_check("text_size", "pass" if smallest >= MIN_TEXT_SCALE else "warn",
                             "Text size is close to the original" if smallest >= MIN_TEXT_SCALE
                             else f"Text was reduced to {smallest:.0%} of the original size", smallest=smallest))

        def family(name):                     # "Lato-Bold" and "Lato Bold" are one family
            return re.split(r"[-\s]", name)[0].lower()

        symbol = ("cm", "computer", "pdftex", "noto")      # symbol/space fonts that are not part of the design
        orig_fonts = {family(f["name"]) for f in orig_rep["fonts"] if not family(f["name"]).startswith(symbol)}
        gen_fonts = {family(f["name"]) for f in gen_rep["fonts"] if not family(f["name"]).startswith(symbol)}
        substituted = sorted(orig_fonts - gen_fonts)
        checks.append(_check("fonts", "pass" if not substituted else "warn",
                             "The original font families are used" if not substituted
                             else f"Font(s) replaced: {', '.join(substituted)}",
                             original=sorted(orig_fonts), generated=sorted(gen_fonts)))

        # ---- visual
        similarity = round(visual_similarity(orig, gen), 3)
        metrics["visual_similarity"] = similarity
        checks.append(_check("visual_similarity", "pass" if similarity >= SIMILARITY_WARN else "warn",
                             f"Ink layout similarity {similarity:.2f} (1.0 = identical layout)", score=similarity))

    metrics["ats_risk"] = gen_rep["ats"]["risk"]
    return {"passed": not any(c["status"] == "fail" for c in checks),
            "checks": checks, "metrics": metrics,
            "summary": {s: sum(c["status"] == s for c in checks) for s in ("pass", "warn", "fail")}}
