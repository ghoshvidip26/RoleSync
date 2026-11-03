"""Phase 3: describe the original resume's visual design, independent of its words.

``build_layout_representation`` combines three inputs about the same PDF:
the manifest (lines, spans, geometry), the grouped entries (what each line is)
and the PDF's own vector drawings. Output is JSON-serializable and answers:

* page size and content boundaries
* the style of every role (name, contacts, headings, company, role title, ...)
* section-heading rules/underlines
* bullet glyph, indent and hanging indent
* vertical spacing between each kind of element
* left/right pair columns (location / dates)
* fonts and whether they are embedded
* ATS risk flags (columns, images, tables, scans, odd headings, broken text)

Nothing is specific to one resume: roles come from the grouped structure, and
measurements are medians of what the PDF actually contains. This module does
not change any file.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from statistics import median
from typing import Any

import pymupdf

from .entry_grouper import BULLET_PATTERN, group_entries, _line_style
from .layout_manifest import build_layout_manifest

MARGIN_ZONE = 18.0           # text this close to a page edge is in the header/footer zone
MIN_TEXT_CHARS = 150         # less than this on a resume page suggests a scan or image PDF
RULE_MAX_THICKNESS = 2.5
RULE_MIN_WIDTH_RATIO = 0.30  # of content width


def _r(value: float, digits: int = 2) -> float:
    return round(float(value), digits)


def _hex(color: Any) -> str | None:
    if not color:
        return None
    return "#" + "".join(f"{round(c * 255):02x}" for c in color[:3])


def _med(values: list[float]) -> float | None:
    return _r(median(values)) if values else None


# --------------------------------------------------------------------- styles

def _style(line: dict, spans: dict[str, dict]) -> dict:
    parts = [spans[s] for s in line["spans"] if s in spans and spans[s]["text"].strip()]
    if not parts:
        return {}
    by_chars = Counter()
    for p in parts:
        by_chars[(p["font"], p["font_size"], p["color"])] += len(p["text"].strip())
    (font, size, color), _ = by_chars.most_common(1)[0]
    text = line["text"]
    return {
        "font": font,
        "size": _r(size),
        "bold": all(p["style"]["bold"] for p in parts),
        "italic": all(p["style"]["italic"] for p in parts),
        "color": color,
        "caps": bool([c for c in text if c.isalpha()]) and text.upper() == text,
    }


def _runs(line: dict, spans: dict[str, dict]) -> list[dict]:
    """Consecutive spans with the same font/size/color, in reading order."""
    runs: list[dict] = []
    for sid in line["spans"]:
        sp = spans.get(sid)
        if not sp or not sp["text"].strip():
            continue
        key = (sp["font"], _r(sp["font_size"]), sp["color"])
        if runs and runs[-1]["key"] == key:
            runs[-1]["sample"] += " " + sp["text"].strip()
        else:
            runs.append({"key": key, "font": sp["font"], "size": _r(sp["font_size"]),
                         "bold": sp["style"]["bold"], "italic": sp["style"]["italic"],
                         "color": sp["color"], "sample": sp["text"].strip(), "x0": _r(sp["bbox"][0])})
    return runs


def _run_pattern(lines: list[dict]) -> dict | None:
    """Most common style sequence among a role's lines, e.g. bold name,
    separator, italic technologies. Needed to redraw mixed-style lines."""
    if not lines:
        return None
    sigs = Counter(tuple(r["key"] for r in runs) for runs in lines if runs)
    if not sigs:
        return None
    sig, count = sigs.most_common(1)[0]
    runs = next(r for r in lines if tuple(x["key"] for x in r) == sig)
    return {
        "runs": [{k: v for k, v in r.items() if k != "key"} for r in runs],
        "share": _r(count / len(lines)),
    }


def _common_style(styles: list[dict]) -> dict | None:
    styles = [s for s in styles if s]
    if not styles:
        return None
    key = lambda s: (s["font"], s["size"], s["bold"], s["italic"], s["color"], s["caps"])
    best, count = Counter(key(s) for s in styles).most_common(1)[0]
    chosen = next(s for s in styles if key(s) == best)
    return dict(chosen, samples=len(styles), consistent=count == len(styles))


# ---------------------------------------------------------------- role mapping

def _role_map(grouped: dict) -> dict[str, str]:
    """line id -> role name, from the semantic grouping."""
    roles: dict[str, str] = {}

    def put(ids, role):
        for i in ids:
            roles.setdefault(i, role)

    h = grouped["header"]
    if h.get("name"):
        put(h["name"]["line_ids"], "name")
    for c in h.get("contacts", []) + h.get("other", []):
        put(c["line_ids"], "contact")

    for s in grouped["sections"]:
        roles[s["heading"]["line_id"]] = "section_heading"
        kind, c = s["kind"], s["content"]
        if kind == "summary":
            put(c["paragraph"]["line_ids"], "summary_text")
        elif kind == "skills":
            for g in c["groups"]:
                put(g["line_ids"], "skills_row")
        elif kind == "experience":
            for co in c["companies"]:
                ids = co["line_ids"]
                if ids:
                    roles.setdefault(ids[0], "company")
                    put(ids[1:], "company_right")
                for r in co["roles"]:
                    taken = {i for b in r["bullets"] for i in b["line_ids"]}
                    taken |= {i for d in r["details"] for i in d["line_ids"]}
                    head = [i for i in r["line_ids"] if i not in taken]
                    if head:
                        roles.setdefault(head[0], "role_title")
                        put(head[1:], "role_right")
                    for d in r["details"]:
                        put(d["line_ids"], "role_detail")
                    for b in r["bullets"]:
                        put(b["line_ids"][:1], "bullet")
                        put(b["line_ids"][1:], "bullet_cont")
        elif kind == "projects":
            for p in c["projects"]:
                taken = {i for b in p["bullets"] for i in b["line_ids"]}
                taken |= {i for d in p["description"] for i in d["line_ids"]}
                head = [i for i in p["line_ids"] if i not in taken]
                if head:
                    roles.setdefault(head[0], "project_title")
                    put(head[1:], "project_right")
                for d in p["description"]:
                    put(d["line_ids"], "project_description")
                for b in p["bullets"]:
                    put(b["line_ids"][:1], "bullet")
                    put(b["line_ids"][1:], "bullet_cont")
        elif kind == "education":
            for e in c["entries"]:
                taken = {i for d in e["details"] for i in d["line_ids"]}
                head = [i for i in e["line_ids"] if i not in taken]
                if head:
                    roles.setdefault(head[0], "education_title")
                    put(head[1:], "education_right")
                for d in e["details"]:
                    put(d["line_ids"], "education_detail")
        else:
            for e in c["entries"]:
                if e.get("head"):
                    ids = e["head"]["line_ids"]
                    roles.setdefault(ids[0], "entry_title")
                    put(ids[1:], "entry_right")
                for it in e["items"]:
                    if it["type"] == "bullet":
                        put(it["line_ids"][:1], "bullet")
                        put(it["line_ids"][1:], "bullet_cont")
                    else:
                        put(it["line_ids"], "entry_text")
    return roles


# ----------------------------------------------------------------------- rules

def _rules(page: pymupdf.Page, page_no: int, content_w: float, lines: list[dict],
           headings: list[dict]) -> list[dict]:
    out = []
    for d in page.get_drawings():
        x0, y0, x1, y1 = d["rect"]
        width, height = x1 - x0, y1 - y0
        if height > RULE_MAX_THICKNESS or width < 4:
            continue
        thickness = _r(max(d.get("width") or 0, height), 3)
        y = (y0 + y1) / 2
        color = _hex(d.get("color") or d.get("fill"))
        entry = {"page": page_no, "y": _r(y), "x0": _r(x0), "x1": _r(x1),
                 "thickness": thickness, "color": color}
        # a short rule that sits directly under a text line is a link underline
        under = [l for l in lines if l["bbox"][0] - 2 <= x0 and x1 <= l["bbox"][2] + 2
                 and 0 <= y - l["bbox"][3] <= 4]
        if under and width < content_w * RULE_MIN_WIDTH_RATIO:
            entry["kind"] = "link_underline"
        elif width >= content_w * RULE_MIN_WIDTH_RATIO:
            entry["kind"] = "separator"
        else:
            entry["kind"] = "other"
        if entry["kind"] == "separator":
            above = [h for h in headings if h["page"] == page_no and h["bbox"][1] <= y]
            if above:
                h = max(above, key=lambda h: h["bbox"][1])
                gap = y - h["bbox"][3]
                entry["section"] = h["text"]
                entry["gap_below_heading"] = _r(gap)
                entry["placement"] = "under_heading" if gap <= 12 else "between_content"
            else:
                entry["placement"] = "above_first_heading"
        out.append(entry)
    return sorted(out, key=lambda r: (r["page"], r["y"], r["x0"]))


# --------------------------------------------------------------------- bullets

def _bullet_style(grouped: dict, lines: dict[str, dict], spans: dict[str, dict]) -> dict | None:
    firsts, conts, glyphs = [], [], []
    for s in grouped["sections"]:
        c = s["content"]
        items = []
        if s["kind"] == "experience":
            items = [b for co in c["companies"] for r in co["roles"] for b in r["bullets"]]
        elif s["kind"] == "projects":
            items = [b for p in c["projects"] for b in p["bullets"]]
        elif "entries" in c:
            items = [i for e in c["entries"] for i in e.get("items", []) if i["type"] == "bullet"]
        for b in items:
            ids = b["line_ids"]
            firsts.append(lines[ids[0]])
            conts += [lines[i] for i in ids[1:]]
            if b.get("marker"):
                glyphs.append(b["marker"])
    if not firsts:
        return None
    first_spans = [[spans[i] for i in l["spans"] if i in spans] for l in firsts]
    marker_spans = [sp[0] for sp in first_spans if sp and BULLET_PATTERN.match(sp[0]["text"].strip() + " ")]
    result: dict[str, Any] = {
        "count": len(firsts),
        "marker": Counter(glyphs).most_common(1)[0][0] if glyphs else None,
        "line_x0": _med([l["bbox"][0] for l in firsts]),
        "wrapped_line_x0": _med([l["bbox"][0] for l in conts]),
    }
    if result["line_x0"] is not None and result["wrapped_line_x0"] is not None:
        result["hanging_indent"] = _r(result["wrapped_line_x0"] - result["line_x0"])
    if marker_spans:
        m = Counter((s["font"], s["font_size"], s["color"]) for s in marker_spans).most_common(1)[0][0]
        result["marker_font"], result["marker_size"], result["marker_color"] = m[0], _r(m[1]), m[2]
    else:
        result["marker_kind"] = "none"
    return result


# --------------------------------------------------------------------- spacing

def _spacing(rows_by_page: dict[int, list[dict]], roles: dict[str, str]) -> dict:
    gaps, pitches = defaultdict(list), defaultdict(list)
    for rows in rows_by_page.values():
        for a, b in zip(rows, rows[1:]):
            ra = roles.get(a["ids"][0], "other")
            rb = roles.get(b["ids"][0], "other")
            key = f"{ra}->{rb}"
            gaps[key].append(b["y0"] - a["y1"])
            pitches[key].append(b["y0"] - a["y0"])
    return {k: {"count": len(v), "gap_median": _med(v), "pitch_median": _med(pitches[k])}
            for k, v in sorted(gaps.items())}


def _rows(lines: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for l in sorted(lines, key=lambda l: (l["bbox"][1], l["bbox"][0])):
        x0, y0, x1, y1 = l["bbox"]
        for r in rows:
            top, bottom = max(r["y0"], y0), min(r["y1"], y1)
            if bottom - top > 0.5 * min(r["y1"] - r["y0"], y1 - y0):
                r["ids"].append(l["id"]); r["y0"] = min(r["y0"], y0); r["y1"] = max(r["y1"], y1)
                break
        else:
            rows.append({"ids": [l["id"]], "y0": y0, "y1": y1})
    return rows


# ------------------------------------------------------------------------ ATS

STANDARD_KINDS = {"summary", "skills", "experience", "projects", "achievements",
                  "education", "certifications", "publications"}


def _flag(code: str, severity: str, message: str, **evidence) -> dict:
    return {"code": code, "severity": severity, "message": message, "evidence": evidence}


def _ats_flags(doc: pymupdf.Document, manifest: dict, grouped: dict,
               lines: list[dict], fonts: list[dict]) -> list[dict]:
    flags = []
    layout = manifest["document"]
    if layout.get("layout_type") == "two_column":
        flags.append(_flag("multi_column", "warning",
                           "Two-column layouts can be read out of order by some ATS parsers.",
                           confidence=layout.get("confidence")))
    elif layout.get("layout_type") == "complex":
        # The analyzer's "complex" is a heuristic (many narrow lines also trigger it),
        # so this is informational; concrete risks are flagged separately below.
        flags.append(_flag("complex_layout", "info",
                           "The layout analyzer could not classify this as a plain single column."))

    for page in doc:
        n = page.number + 1
        if page.get_images():
            flags.append(_flag("images", "warning",
                               "Images or icons can be skipped or turned into junk by an ATS.",
                               page=n, count=len(page.get_images())))
        try:
            tables = [t for t in page.find_tables(strategy="lines").tables
                      if t.row_count >= 2 and t.col_count >= 2]
        except Exception:
            tables = []
        if tables:
            flags.append(_flag("tables", "warning",
                               "Tables are often flattened or skipped by an ATS.",
                               page=n, count=len(tables)))
        text = page.get_text()
        if len(text.strip()) < MIN_TEXT_CHARS:
            flags.append(_flag("little_text", "warning",
                               "Almost no extractable text; the PDF may be a scan or image.",
                               page=n, characters=len(text.strip())))
        bad = [c for c in text if c == "\ufffd" or "\ue000" <= c <= "\uf8ff" or c == "\x00"]
        if bad:
            flags.append(_flag("unreadable_characters", "warning",
                               "Some characters extract as symbols an ATS cannot read.",
                               page=n, count=len(bad), samples=sorted({hex(ord(c)) for c in bad})[:5]))
        h = page.rect.height
        contact_ids = {c["line_ids"][0] for c in grouped["header"].get("contacts", [])}
        edge = [l for l in lines if l["id"].startswith(f"p{n}_") and l["id"] in contact_ids
                and (l["bbox"][3] < MARGIN_ZONE or l["bbox"][1] > h - MARGIN_ZONE)]
        if edge:
            flags.append(_flag("contact_in_margin", "warning",
                               "Contact details sit in the page-edge zone; parsers often skip headers and footers.",
                               page=n))

    custom = [s["name"] for s in grouped["sections"] if s["kind"] == "custom"]
    if custom:
        flags.append(_flag("nonstandard_headings", "info",
                           "These headings are not standard ATS labels; they are kept as-is.",
                           headings=custom))
    if not any(s["kind"] == "experience" for s in grouped["sections"]):
        flags.append(_flag("no_experience_section", "info", "No Experience section was recognised."))
    if not grouped["header"].get("contacts"):
        flags.append(_flag("no_contact_details", "warning", "No email, phone or link was recognised in the header."))
    if manifest["document"]["page_count"] > 2:
        flags.append(_flag("long_resume", "info", "Resumes longer than two pages are uncommon for engineers.",
                           pages=manifest["document"]["page_count"]))
    not_embedded = [f["name"] for f in fonts if not f["embedded"] and not f["base14"]]
    if not_embedded:
        flags.append(_flag("fonts_not_embedded", "info",
                           "Some fonts are not embedded; other viewers may substitute them.", fonts=not_embedded))
    return flags


def _risk(flags: list[dict]) -> str:
    warnings = sum(1 for f in flags if f["severity"] == "warning")
    return "low" if warnings == 0 else "medium" if warnings == 1 else "high"


# ------------------------------------------------------------------------ fonts

BASE14 = ("helvetica", "arial", "times", "courier", "symbol", "zapf")


def _fonts(doc: pymupdf.Document) -> list[dict]:
    seen: dict[str, dict] = {}
    for page in doc:
        for f in page.get_fonts(full=True):
            xref, name = f[0], f[3]
            clean = name.split("+", 1)[-1]
            if clean in seen:
                continue
            try:
                _, _, _, buf = doc.extract_font(xref)
            except Exception:
                buf = b""
            seen[clean] = {"name": clean, "embedded": bool(buf), "subset": "+" in name,
                           "base14": clean.lower().startswith(BASE14)}
    return list(seen.values())


# ------------------------------------------------------------------------- main

def build_layout_representation(
    pdf_bytes: bytes,
    manifest: dict[str, Any] | None = None,
    grouped: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe the visual structure of a resume PDF (see module docstring)."""
    manifest = manifest or build_layout_manifest(pdf_bytes)
    grouped = grouped or group_entries(manifest)

    lines = {l["id"]: l for p in manifest["pages"] for l in p["lines"]}
    spans = {s["id"]: s for p in manifest["pages"] for s in p["spans"]}
    roles = _role_map(grouped)
    line_styles = {i: _style(l, spans) for i, l in lines.items()}

    by_role: dict[str, list[dict]] = defaultdict(list)
    for i, role in roles.items():
        if line_styles.get(i):
            by_role[role].append(line_styles[i])
    typography = {role: _common_style(styles) for role, styles in sorted(by_role.items())}
    for role, st in typography.items():
        st["run_pattern"] = _run_pattern(
            [_runs(lines[i], spans) for i, r in roles.items() if r == role and i in lines])

    # alignment per role
    alignment = {}
    for role in by_role:
        counts = Counter(lines[i]["alignment"] for i, r in roles.items() if r == role)
        alignment[role] = counts.most_common(1)[0][0]
    for role, st in typography.items():
        st["alignment"] = alignment[role]

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        pages = []
        all_rules = []
        rows_by_page = {}
        headings = []
        for sec in grouped["sections"]:
            l = lines[sec["heading"]["line_id"]]
            headings.append({"text": sec["name"], "page": int(l["id"].split("_")[0][1:]), "bbox": l["bbox"]})
        for page in doc:
            n = page.number + 1
            page_lines = [l for l in lines.values() if l["id"].startswith(f"p{n}_")]
            if page_lines:
                x0 = min(l["bbox"][0] for l in page_lines)
                x1 = max(l["bbox"][2] for l in page_lines)
                y0 = min(l["bbox"][1] for l in page_lines)
                y1 = max(l["bbox"][3] for l in page_lines)
            else:
                x0, y0, x1, y1 = 0, 0, page.rect.width, page.rect.height
            pages.append({
                "page": n,
                "width": _r(page.rect.width), "height": _r(page.rect.height), "rotation": page.rotation,
                "content_box": [_r(x0), _r(y0), _r(x1), _r(y1)],
                "margins": {"left": _r(x0), "top": _r(y0), "right": _r(page.rect.width - x1),
                            "bottom": _r(page.rect.height - y1)},
            })
            all_rules += _rules(page, n, x1 - x0, page_lines, headings)
            rows_by_page[n] = _rows(page_lines)
        fonts = _fonts(doc)
        flags = _ats_flags(doc, manifest, grouped, list(lines.values()), fonts)

    right_items = [lines[i] for i, r in roles.items() if r.endswith("_right")]
    left_items = [lines[i] for i, r in roles.items() if r in {"company", "role_title", "project_title", "education_title", "entry_title"}]
    pairs = {
        "count": len(right_items),
        "right_edge_x": _med([l["bbox"][2] for l in right_items]),
        "left_edge_x": _med([l["bbox"][0] for l in left_items]),
        "roles": sorted({roles[l["id"]] for l in right_items}),
    }

    headings_style = typography.get("section_heading")
    separators = [r for r in all_rules if r["kind"] == "separator"]

    return {
        "version": "1.0",
        "document": {"page_count": manifest["document"]["page_count"],
                     "page_size": manifest["document"]["page_size"],
                     "layout_type": manifest["document"]["layout_type"]},
        "pages": pages,
        "typography": typography,
        "section_headings": {
            "style": headings_style,
            "order": [s["name"] for s in grouped["sections"]],
            "rule_under_every_heading": bool(separators) and len(
                {r["section"] for r in separators if r.get("placement") == "under_heading"}) == len(headings),
        },
        "rules": all_rules,
        "bullets": _bullet_style(grouped, lines, spans),
        "spacing": _spacing(rows_by_page, roles),
        "pairs": pairs,
        "fonts": fonts,
        "ats": {"risk": _risk(flags), "flags": flags},
        "role_of_line": roles,
    }
