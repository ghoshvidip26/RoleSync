"""Phase 2: turn the manifest's individual lines into resume entries.

Input  : the dict returned by ``build_layout_manifest``.
Output : header, plus sections whose content is grouped into meaningful
         entries (company -> roles -> bullets, projects, education, ...).

Nothing here is specific to one resume. Structure is inferred from geometry
(rows, left/right pairs, bullets) and style (bold, size, caps), not from a
fixed layout. Sections the draft has no field for (Certifications,
Publications, Awards, ...) are kept as optional ``custom`` content so they are
never lost. Every element carries ``line_ids`` so later phases can map it back
to the original geometry.
"""

from __future__ import annotations

import re
from typing import Any

from .layout_manifest import SECTION_NAMES

KIND_BY_NAME = {
    "Summary": "summary",
    "Technical Skills": "skills",
    "Experience": "experience",
    "Projects": "projects",
    "Achievements": "achievements",
    "Education": "education",
    "Certifications": "certifications",
    "Publications": "publications",
}

DATE_RE = re.compile(
    r"\b(?:19|20)\d{2}\b|\b(?:present|current|ongoing|till date|now)\b", re.I
)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
URL_RE = re.compile(r"(?:https?://|www\.|linkedin\.com|github\.com|[\w-]+\.(?:com|io|dev|me|in|net|org)\b)", re.I)
# Bullet glyphs differ between resume tools (• ● ▪ · ∙ ■ ➢ ...). Dashes only count
# when followed by whitespace, so "-" or "–" inside text or ranges is not a bullet.
BULLET_PATTERN = re.compile(r"^(?:[•●▪◦‣⁃·∙■□○➢➤*]\s*|[-–—]\s+)")
SEPARATORS = {"|", "•", "·", ",", ";", "/", "–", "-", "—"}


def has_date(text: str) -> bool:
    return bool(DATE_RE.search(text or ""))


# --------------------------------------------------------------------- rows

def _same_row(a: dict, b: dict) -> bool:
    top, bottom = max(a["y0"], b["y0"]), min(a["y1"], b["y1"])
    overlap = bottom - top
    smaller = min(a["y1"] - a["y0"], b["y1"] - b["y0"])
    return smaller > 0 and overlap > 0.5 * smaller


def _geometry_text(line: dict, spans: dict[str, dict]) -> str:
    """Line text from its spans: a space only where the PDF has a gap (or the span has one).
    Joining with a blanket space turns "LocalPay," into "LocalPay ,"."""
    out, prev = "", None
    for sid in line["spans"]:
        sp = spans.get(sid)
        if not sp or not sp["text"]:
            continue
        text = sp["text"]
        if prev is not None and text.strip():
            gap = sp["bbox"][0] - prev["bbox"][2]
            if gap > 0.12 * sp["font_size"] and not out.endswith(" ") and not text.startswith(" "):
                out += " "
        out += text
        if text.strip():
            prev = sp
    return re.sub(r"\s+", " ", out).strip() or line["text"]


def _line_style(line: dict, spans: dict[str, dict]) -> dict:
    parts = [spans[s] for s in line["spans"] if s in spans and spans[s]["text"].strip()]
    if not parts:
        return {"bold": False, "italic": False, "size": 0.0, "color": "#000000"}
    return {
        "bold": all(p["style"]["bold"] for p in parts),
        "italic": all(p["style"]["italic"] for p in parts),
        "size": max(p["font_size"] for p in parts),
        "color": parts[0]["color"],
    }


def _build_rows(manifest: dict) -> list[dict]:
    spans = {s["id"]: s for p in manifest["pages"] for s in p["spans"]}
    rows: list[dict] = []
    for page in manifest["pages"]:
        page_no = page["page_number"]
        ordered = sorted(page["lines"], key=lambda l: (l["bbox"][1], l["bbox"][0]))
        page_rows: list[dict] = []
        for line in ordered:
            x0, y0, x1, y1 = line["bbox"]
            item = dict(line, style=_line_style(line, spans), text=_geometry_text(line, spans))
            for row in page_rows:
                if _same_row(row, {"y0": y0, "y1": y1}):
                    row["items"].append(item)
                    row["y0"], row["y1"] = min(row["y0"], y0), max(row["y1"], y1)
                    break
            else:
                page_rows.append({"page": page_no, "y0": y0, "y1": y1, "items": [item]})
        for row in page_rows:
            row["items"].sort(key=lambda i: i["bbox"][0])
            # drop pure separator fragments such as a lone "|"
            real = [i for i in row["items"] if i["text"].strip() not in SEPARATORS]
            row["items"] = real or row["items"]
            row["x0"] = row["items"][0]["bbox"][0]
            row["x1"] = row["items"][-1]["bbox"][2]
        rows.extend(sorted(page_rows, key=lambda r: r["y0"]))
    return rows


def _row_text(row: dict) -> str:
    return " ".join(i["text"] for i in row["items"]).strip()


def _row_ids(row: dict) -> list[str]:
    return [i["id"] for i in row["items"]]


# ----------------------------------------------------------------- headings

def _canonical(text: str) -> str | None:
    return SECTION_NAMES.get(re.sub(r"\s+", " ", text).strip().lower().rstrip(":"))


def _is_caps(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


def _signature(item: dict) -> tuple:
    s = item["style"]
    return (round(s["size"]), s["bold"], _is_caps(item["text"]), s["color"])


def _find_headings(rows: list[dict]) -> dict[int, str]:
    """Row index -> heading name. Known names first, then same-looking rows."""
    known: dict[int, str] = {}
    for i, row in enumerate(rows):
        if len(row["items"]) == 1 and (name := _canonical(row["items"][0]["text"])):
            known[i] = name
    if not known:
        return {}
    signatures = {_signature(rows[i]["items"][0]) for i in known}
    found = dict(known)
    for i, row in enumerate(rows):
        if i in found or len(row["items"]) != 1:
            continue
        item = row["items"][0]
        text = item["text"].strip()
        words = text.split()
        if (
            1 <= len(words) <= 4
            and len(text) <= 40
            and not any(c.isdigit() for c in text)
            and not BULLET_PATTERN.match(text)
            and _signature(item) in signatures
        ):
            found[i] = text.rstrip(":").title() if _is_caps(text) else text.rstrip(":")
    return found


# ------------------------------------------------------------------- blocks

def _bullet(row: dict) -> dict:
    text = _row_text(row)
    marker = BULLET_PATTERN.match(text)
    return {
        "type": "bullet",
        "marker": marker.group(0).strip() if marker else "",
        "text": text[marker.end():].strip() if marker else text,
        "line_ids": _row_ids(row),
        "page": row["page"],
        "x0": row["x0"],
    }


def _is_head(row: dict) -> bool:
    first = row["items"][0]
    if BULLET_PATTERN.match(first["text"]):
        return False
    return len(row["items"]) > 1 or first["style"]["bold"]


def _head(row: dict) -> dict:
    items = row["items"]
    left, right = items[0], items[1:]
    return {
        "left": left["text"].strip(),
        "right": " ".join(i["text"] for i in right).strip(),
        "bold": left["style"]["bold"],
        "italic": left["style"]["italic"],
        "line_ids": _row_ids(row),
        "page": row["page"],
        "y0": row["y0"],
    }


def _entries(rows: list[dict]) -> list[dict]:
    """Generic grouping: each head row starts an entry; bullets and text
    that follow belong to it. Wrapped lines join the item they continue."""
    entries: list[dict] = []
    current: dict | None = None
    last_item: dict | None = None
    for row in rows:
        if _is_head(row):
            current = {"head": _head(row), "items": []}
            entries.append(current)
            last_item = None
            continue
        if current is None:
            current = {"head": None, "items": []}
            entries.append(current)
        if BULLET_PATTERN.match(row["items"][0]["text"]):
            last_item = _bullet(row)
            current["items"].append(last_item)
        elif last_item is not None and last_item["type"] == "bullet":
            last_item["text"] = f"{last_item['text']} {_row_text(row)}".strip()
            last_item["line_ids"] += _row_ids(row)
        else:
            text = _row_text(row)
            if last_item is not None and last_item["type"] == "text" and last_item.get("open"):
                last_item["text"] += " " + text
                last_item["line_ids"] += _row_ids(row)
            else:
                last_item = {"type": "text", "text": text, "line_ids": _row_ids(row),
                             "page": row["page"], "open": True}
                current["items"].append(last_item)
    for e in entries:
        for item in e["items"]:
            item.pop("open", None)
    return entries


# ------------------------------------------------------------ section shapes

def _skills(rows: list[dict]) -> dict:
    groups = []
    for row in rows:
        for item in row["items"]:
            text = item["text"].strip()
            label, sep, values = text.partition(":")
            groups.append({
                "label": label.strip() if sep else "",
                "values": (values if sep else text).strip(),
                "line_ids": [item["id"]],
            })
    return {"groups": groups}


def _ids(entry: dict) -> list[str]:
    out = list(entry["head"]["line_ids"]) if entry["head"] else []
    for item in entry["items"]:
        out += item["line_ids"]
    return out


def _bullets(entry: dict) -> list[dict]:
    return [i for i in entry["items"] if i["type"] == "bullet"]


def _texts(entry: dict) -> list[dict]:
    return [i for i in entry["items"] if i["type"] == "text"]


def _experience(entries: list[dict]) -> dict:
    companies: list[dict] = []
    for entry in entries:
        head = entry["head"]
        if head is None:                                  # stray bullets before any head
            if companies and companies[-1]["roles"]:
                companies[-1]["roles"][-1]["bullets"] += _bullets(entry)
            else:
                companies.append({"company": "", "location": "", "line_ids": [], "roles": [
                    {"title": "", "duration": "", "bullets": _bullets(entry), "details": [],
                     "line_ids": _ids(entry)}]})
            continue
        dated = has_date(head["right"]) or (not head["right"] and has_date(head["left"]))
        if not dated:                                     # company row: name | location
            companies.append({"company": head["left"], "location": head["right"],
                              "line_ids": head["line_ids"], "roles": []})
            if entry["items"]:                            # undated role under it
                companies[-1]["roles"].append({
                    "title": "", "duration": "", "bullets": _bullets(entry),
                    "details": _texts(entry), "line_ids": _ids(entry)})
            continue
        texts = _texts(entry)
        if companies and not companies[-1]["roles"]:      # role row under a company row
            role = {"title": head["left"], "duration": head["right"],
                    "bullets": _bullets(entry), "details": texts, "line_ids": _ids(entry)}
            companies[-1]["roles"].append(role)
        elif companies and companies[-1]["roles"] and _title_only(head):
            role = {"title": head["left"], "duration": head["right"],
                    "bullets": _bullets(entry), "details": texts, "line_ids": _ids(entry)}
            companies[-1]["roles"].append(role)          # another role, same company
        else:
            # one-row format: "Company ... dates" with the title on the next line,
            # or "Title, Company ... dates"
            title = texts[0]["text"] if texts else ""
            companies.append({
                "company": head["left"], "location": "", "line_ids": head["line_ids"],
                "roles": [{"title": title, "duration": head["right"], "bullets": _bullets(entry),
                           "details": texts[1:] if title else texts, "line_ids": _ids(entry)}]})
    return {"companies": companies}


def _title_only(head: dict) -> bool:
    """A dated head that does not look like a new company (not bold when the
    company rows are bold) continues the previous company."""
    return not head["bold"] or head["italic"]


def _projects(entries: list[dict]) -> dict:
    projects = []
    for entry in entries:
        head = entry["head"]
        if head is None:
            if projects:
                projects[-1]["bullets"] += _bullets(entry)
            continue
        name, _, tech = head["left"].partition("|")
        projects.append({
            "name": name.strip(),
            "technologies": tech.strip(),
            "dates": head["right"],
            "description": _texts(entry),
            "bullets": _bullets(entry),
            "line_ids": _ids(entry),
        })
    return {"projects": projects}


def _education(entries: list[dict]) -> dict:
    out = []
    for entry in entries:
        head = entry["head"]
        if head is None:
            if out:
                out[-1]["details"] += _texts(entry) + _bullets(entry)
            continue
        parts = [p.strip() for p in head["right"].split("|") if p.strip()]
        dates = " | ".join(p for p in parts if has_date(p))
        location = " | ".join(p for p in parts if not has_date(p))
        out.append({"institution": head["left"], "location": location, "dates": dates,
                    "details": _texts(entry) + _bullets(entry), "line_ids": _ids(entry)})
    return {"entries": out}


def _generic(entries: list[dict]) -> dict:
    return {"entries": [
        {"head": e["head"], "items": e["items"], "line_ids": _ids(e)} for e in entries
    ]}


# ------------------------------------------------------------------- header

def _header(rows: list[dict]) -> dict:
    if not rows:
        return {"name": None, "contacts": [], "other": [], "line_ids": []}
    name_row = max(rows, key=lambda r: max(i["style"]["size"] for i in r["items"]))
    contacts, other = [], []
    for row in rows:
        if row is name_row:
            continue
        for item in row["items"]:
            for piece in re.split(r"\s*\|\s*", item["text"]):
                piece = piece.strip()
                if not piece:
                    continue
                if EMAIL_RE.search(piece):
                    kind = "email"
                elif URL_RE.search(piece):
                    kind = "url"
                elif PHONE_RE.fullmatch(piece):
                    kind = "phone"
                else:
                    kind = None
                (contacts if kind else other).append(
                    {"kind": kind, "text": piece, "line_ids": [item["id"]]} if kind
                    else {"text": piece, "line_ids": [item["id"]]})
    return {
        "name": {"text": _row_text(name_row), "line_ids": _row_ids(name_row)},
        "contacts": contacts,
        "other": other,
        "line_ids": [i for r in rows for i in _row_ids(r)],
    }


# --------------------------------------------------------------------- main

def group_entries(manifest: dict[str, Any]) -> dict[str, Any]:
    """Group manifest lines into header + sections of semantic entries."""
    if not manifest.get("pages"):
        raise ValueError("Manifest has no pages")
    rows = _build_rows(manifest)
    headings = _find_headings(rows)
    order = sorted(headings)
    header_rows = rows[: order[0]] if order else rows

    sections = []
    for n, start in enumerate(order):
        end = order[n + 1] if n + 1 < len(order) else len(rows)
        body = rows[start + 1: end]
        name = headings[start]
        kind = KIND_BY_NAME.get(name, "custom")
        if kind == "summary":
            content = {"paragraph": {"text": " ".join(_row_text(r) for r in body),
                                     "line_ids": [i for r in body for i in _row_ids(r)]}}
        elif kind == "skills":
            content = _skills(body)
        else:
            entries = _entries(body)
            shape = {"experience": _experience, "projects": _projects,
                     "education": _education}.get(kind, _generic)
            content = shape(entries)
        heading_item = rows[start]["items"][0]
        sections.append({
            "name": name,
            "kind": kind,
            "heading": {"text": heading_item["text"], "line_id": heading_item["id"]},
            "content": content,
            "line_ids": [i for r in body for i in _row_ids(r)],
        })

    warnings = []
    if not order:
        warnings.append("No section headings were recognised.")
    if not header_rows or not header_rows[0]["items"]:
        warnings.append("No header text found.")
    return {"version": "1.0", "header": _header(header_rows),
            "sections": sections, "warnings": warnings}
