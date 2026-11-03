"""Phase 4: map an approved tailored draft onto the original resume's structure.

Input : ``group_entries(manifest)`` output (what the original contains) and the
        approved draft dict (headline, summary, skills, experience, projects,
        achievements, education).
Output: a render plan. It follows the ORIGINAL resume's sections, order and
        slots, with the text of each slot taken from the draft where the draft
        provides it. Every slot says which original lines it replaces, so the
        renderer can reuse their position and style, and whether it changed.

Rules (each is reported in ``plan["report"]``, nothing happens silently):

* Layout wins. The plan never adds a section, row or field the original does
  not have. Draft content with no home is listed under ``unplaced``.
* Missing or empty draft fields mean "no opinion": the original is kept.
* Facts are never invented. Company, title, dates and location come from the
  draft only when the draft supplies them, otherwise from the original.
* Original experience/projects the draft leaves out are NOT restored; they are
  listed under ``omitted``. Draft entries with no original match are kept but
  marked ``added`` for review.
* Skills: the draft's skills are the job-relevant ones, so they are moved to
  the front of the matching original group. No original skill is removed.
* Education and header/contact details are facts: the original text is kept
  and any disagreement with the draft is reported.
* Sections the draft has no field for (Certifications, Publications, Awards,
  ...) are kept unchanged, unless the draft supplies them through an optional
  ``custom_sections`` mapping (section name -> list of text).
"""

from __future__ import annotations

import re
from typing import Any

MATCH_THRESHOLD = 0.55
_STOP = {"the", "and", "of", "at", "inc", "ltd", "llc", "pvt", "limited", "private", "co", "corp"}


# ---------------------------------------------------------------- text helpers

def _clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(_clean(v) for v in value if _clean(v))
    return re.sub(r"\s+", " ", str(value)).strip()


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP}


def _sim(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    return max(inter / len(ta | tb), 0.9 * inter / min(len(ta), len(tb)))


def _fact(new: str, original: str) -> str:
    """For facts (company, title, dates, ...): the draft's text counts only when it says
    something different. Same words with different case, punctuation or dash keep the
    original spelling, so the resume does not change for no reason."""
    new, original = _clean(new), _clean(original)
    if not new:
        return original
    if original and _tokens(new) == _tokens(original):
        return original
    return new


def _skill_key(skill: str) -> str:
    return re.sub(r"[^a-z0-9+#]", "", skill.lower())


def _split_top(text: str) -> list[str]:
    """Split on commas/semicolons that are not inside brackets."""
    out, depth, cur = [], 0, ""
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        if ch in ",;" and depth == 0:
            if cur.strip():
                out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def _slot(text: str, original: str | None, template: list[str]) -> dict:
    text, original_clean = _clean(text), _clean(original) if original is not None else None
    return {
        "text": text,
        "original": original_clean,
        "changed": original_clean is None or text != original_clean,
        "template": list(template),
    }


SAME_TEXT = 0.8      # at least this similar to an original bullet: same content
REWORDED_TEXT = 0.45  # between the two: a rewording of an original bullet


def _provenance(text: str, originals: list[dict]) -> str:
    """Does this bullet come from the original resume? 'original', 'reworded' or 'not_in_original'."""
    best = max((_sim(text, o["text"]) for o in originals), default=0.0)
    return "original" if best >= SAME_TEXT else "reworded" if best >= REWORDED_TEXT else "not_in_original"


def _bullet_slots(new: list[str], original: list[dict]) -> list[dict]:
    """New bullet texts, each styled after the original bullet at the same
    position (the last original bullet when there are more new bullets)."""
    slots = []
    for i, text in enumerate(new):
        src = original[i] if i < len(original) else (original[-1] if original else None)
        slots.append({
            "text": _clean(text),
            "original": _clean(src["text"]) if i < len(original) else None,
            "changed": i >= len(original) or _clean(text) != _clean(original[i]["text"]),
            "template": list(src["line_ids"]) if src else [],
            "provenance": _provenance(_clean(text), original) if original else "not_in_original",
        })
    return slots


def _items(values: Any) -> list[str]:
    if not values:
        return []
    if isinstance(values, str):
        return [values] if _clean(values) else []
    return [t for t in (_clean(v.get("details") if isinstance(v, dict) and "details" in v else v)
                        for v in values) if t]


# ---------------------------------------------------------------------- report

class _Report:
    def __init__(self):
        self.data: dict[str, list[dict]] = {k: [] for k in
                                             ("changed", "added", "omitted", "unplaced", "kept_original", "warnings")}

    def add(self, bucket: str, section: str, what: str, detail: str = ""):
        self.data[bucket].append({"section": section, "what": what, "detail": detail})


# -------------------------------------------------------------------- sections

def _summary(sec: dict, draft: dict, rep: _Report) -> dict:
    para = sec["content"]["paragraph"]
    new = _clean(draft.get("summary"))
    if not new:
        rep.add("kept_original", sec["name"], "summary", "draft has no summary")
        new = para["text"]
    slot = _slot(new, para["text"], para["line_ids"])
    if slot["changed"]:
        rep.add("changed", sec["name"], "summary")
    return {"paragraph": slot}


def _skills(sec: dict, draft: dict, rep: _Report) -> dict:
    wanted = [_clean(s) for s in (draft.get("skills") or []) if _clean(s)]
    order = {_skill_key(s): i for i, s in reversed(list(enumerate(wanted)))}
    found: set[str] = set()
    groups = []
    for g in sec["content"]["groups"]:
        values = _split_top(g["values"])
        sep = "; " if g["values"].count(";") > g["values"].count(",") else ", "
        hit = [v for v in values if _skill_key(v) in order]
        hit.sort(key=lambda v: order[_skill_key(v)])
        rest = [v for v in values if _skill_key(v) not in order]
        found.update(_skill_key(v) for v in hit)
        new_values = sep.join(hit + rest)
        slot = _slot(new_values, g["values"], g["line_ids"])
        if slot["changed"]:
            rep.add("changed", sec["name"], f"{g['label'] or 'skills'}: relevant skills moved first")
        groups.append({"label": g["label"], "values": slot, "highlighted": hit, "template": g["line_ids"]})
    for s in wanted:
        if _skill_key(s) not in found:
            rep.add("unplaced", sec["name"], s, "skill is not in any skill group of the original resume")
    return {"groups": groups}


def _match(draft_items: list[dict], originals: list[dict], score) -> dict[int, int]:
    """One-to-one greedy matching, best score first. Returns draft idx -> original idx."""
    pairs = sorted(
        ((score(d, o), i, j) for i, d in enumerate(draft_items) for j, o in enumerate(originals)),
        key=lambda t: -t[0])
    used_d, used_o, out = set(), set(), {}
    for s, i, j in pairs:
        if s < MATCH_THRESHOLD or i in used_d or j in used_o:
            continue
        out[i] = j
        used_d.add(i)
        used_o.add(j)
    return out


def _experience(sec: dict, draft: dict, rep: _Report) -> dict:
    companies = sec["content"]["companies"]
    flat = [(ci, ri, co, r) for ci, co in enumerate(companies) for ri, r in enumerate(co["roles"])]
    entries = [e for e in (draft.get("experience") or []) if isinstance(e, dict)]
    if not entries:
        rep.add("kept_original", sec["name"], "experience", "draft has no experience entries")
        entries = []

    def score(d, o):
        _, _, co, r = o
        c = _sim(_clean(d.get("company")), co["company"])
        t = _sim(_clean(d.get("job_title")), r["title"])
        return 0.6 * c + 0.4 * t if _clean(d.get("job_title")) and r["title"] else c

    matches = _match(entries, flat, score) if entries else {}
    by_original = {j: i for i, j in matches.items()}

    plan_companies: list[dict] = []
    for ci, co in enumerate(companies):
        roles = []
        for ri, r in enumerate(co["roles"]):
            j = next(k for k, f in enumerate(flat) if f[0] == ci and f[1] == ri)
            if entries and j not in by_original:
                rep.add("omitted", sec["name"], f"{r['title'] or co['company']} @ {co['company']}",
                        "not in the approved draft; not restored")
                continue
            d = entries[by_original[j]] if entries else {}
            title = _fact(d.get("job_title"), r["title"])
            duration = _fact(d.get("duration"), r["duration"])
            bullets = [_clean(b) for b in (d.get("bullets") or []) if _clean(b)]
            if not bullets:
                bullets = [b["text"] for b in r["bullets"]]
                if entries:
                    rep.add("kept_original", sec["name"], f"bullets of {title}", "draft has no bullets")
            role = {
                "status": "mapped",
                "title": _slot(title, r["title"], r["line_ids"][:1]),
                "duration": _slot(duration, r["duration"], r["line_ids"][1:2]),
                "bullets": _bullet_slots(bullets, r["bullets"]),
                "details": r["details"],
                "template": r["line_ids"],
            }
            if any(role[k]["changed"] for k in ("title", "duration")) or any(b["changed"] for b in role["bullets"]):
                rep.add("changed", sec["name"], f"{title} @ {co['company']}")
            roles.append(role)
        if not roles and entries:
            continue                                   # every role of this company was omitted
        d_first = next((entries[by_original[j]] for j, f in enumerate(flat)
                        if f[0] == ci and j in by_original), {}) if entries else {}
        plan_companies.append({
            "status": "mapped",
            "company": _slot(_fact(d_first.get("company"), co["company"]), co["company"], co["line_ids"][:1]),
            "location": _slot(_fact(d_first.get("location"), co["location"]), co["location"], co["line_ids"][1:]),
            "roles": roles,
            "template": co["line_ids"],
        })

    # draft entries without an original counterpart: keep, but flag for review
    for i, d in enumerate(entries):
        if i in matches:
            continue
        company = _clean(d.get("company"))
        title = _clean(d.get("job_title"))
        bullets = [_clean(b) for b in (d.get("bullets") or []) if _clean(b)]
        host = next((pc for pc in plan_companies if _sim(company, pc["company"]["text"]) >= 0.6), None)
        model = (host or (plan_companies[-1] if plan_companies else None))
        model_role = model["roles"][-1] if model and model["roles"] else None
        role = {
            "status": "added",
            "title": _slot(title, None, model_role["title"]["template"] if model_role else []),
            "duration": _slot(_clean(d.get("duration")), None, model_role["duration"]["template"] if model_role else []),
            "bullets": [{"text": b, "original": None, "changed": True, "provenance": "not_in_original",
                         "template": model_role["bullets"][-1]["template"] if model_role and model_role["bullets"] else []}
                        for b in bullets],
            "details": [],
            "template": model_role["template"] if model_role else [],
        }
        if host:
            host["roles"].append(role)
        else:
            plan_companies.append({
                "status": "added",
                "company": _slot(company, None, model["company"]["template"] if model else []),
                "location": _slot(_clean(d.get("location")), None, model["location"]["template"] if model else []),
                "roles": [role], "template": model["template"] if model else [],
            })
        rep.add("added", sec["name"], f"{title or 'role'} @ {company or '?'}",
                "not in the original resume; review before sending")
    return {"companies": plan_companies}


def _projects(sec: dict, draft: dict, rep: _Report) -> dict:
    originals = sec["content"]["projects"]
    entries = [e for e in (draft.get("projects") or []) if isinstance(e, dict)]
    if not entries:
        rep.add("kept_original", sec["name"], "projects", "draft has no projects")
        entries = [{"name": p["name"]} for p in originals]
        keep_all = True
    else:
        keep_all = False

    matches = _match(entries, originals, lambda d, o: _sim(_clean(d.get("name")), o["name"]))
    plan, used = [], set()
    for i, d in enumerate(entries):
        j = matches.get(i)
        if j is None:
            name = _clean(d.get("name"))
            plan.append({"status": "added", "name": _slot(name, None, []), "technologies": _slot(
                _clean(d.get("technologies")), None, []), "dates": _slot(_clean(d.get("dates") or d.get("duration")), None, []),
                         "description": [], "bullets": _bullet_slots(
                             [_clean(b) for b in (d.get("bullets") or []) if _clean(b)], []), "template": []})
            rep.add("added", sec["name"], name or "project", "not in the original resume; review before sending")
            continue
        used.add(j)
        o = originals[j]
        tech_new = _clean(d.get("technologies"))
        same_tech = {_skill_key(t) for t in _split_top(tech_new)} == {_skill_key(t) for t in _split_top(o["technologies"])}
        tech = o["technologies"] if (not tech_new or same_tech) else tech_new
        bullets = [_clean(b) for b in (d.get("bullets") or []) if _clean(b)] or [b["text"] for b in o["bullets"]]
        desc_new = _clean(d.get("description"))
        if desc_new and not o["description"]:
            rep.add("unplaced", sec["name"], f"description of {o['name']}",
                    "the original layout has no description line for projects")
        desc = [dict(_slot(desc_new if k == 0 and desc_new else x["text"], x["text"], x["line_ids"]))
                for k, x in enumerate(o["description"])]
        item = {
            "status": "mapped",
            "name": _slot(_fact(d.get("name"), o["name"]), o["name"], o["line_ids"][:1]),
            "technologies": _slot(tech, o["technologies"], o["line_ids"][:1]),
            "dates": _slot(_fact(d.get("dates") or d.get("duration"), o["dates"]), o["dates"], o["line_ids"][1:2]),
            "description": desc,
            "bullets": _bullet_slots(bullets, o["bullets"]),
            "template": o["line_ids"],
        }
        if any(item[k]["changed"] for k in ("name", "technologies", "dates")) or \
                any(b["changed"] for b in item["bullets"]) or any(x["changed"] for x in desc):
            rep.add("changed", sec["name"], item["name"]["text"])
        plan.append(item)
    if not keep_all:
        for j, o in enumerate(originals):
            if j not in used:
                rep.add("omitted", sec["name"], o["name"], "not in the approved draft; not restored")
    return {"projects": plan}


def _flat_bullets(sec: dict, new_texts: list[str], rep: _Report, label: str) -> dict:
    entries = sec["content"]["entries"]
    single = len(entries) == 1 and entries[0].get("head") is None
    if not new_texts:
        rep.add("kept_original", sec["name"], label, "no replacement text supplied")
        return {"entries": entries, "status": "unchanged"}
    if not single:
        rep.add("warnings", sec["name"], label, "section has sub-headings; kept unchanged")
        return {"entries": entries, "status": "unchanged"}
    originals = [i for i in entries[0]["items"] if i["type"] == "bullet"]
    slots = _bullet_slots(new_texts, originals)
    if any(s["changed"] for s in slots) or len(slots) != len(originals):
        rep.add("changed", sec["name"], label)
    return {"bullets": slots, "status": "mapped"}


def _education(sec: dict, draft: dict, rep: _Report) -> dict:
    entries = sec["content"]["entries"]
    for text in _items(draft.get("education")):
        best = max((_sim(text, f"{e['institution']} {e['location']} {e['dates']}") for e in entries), default=0)
        best = max(best, max((_sim(text, e["institution"]) for e in entries), default=0))
        if best < MATCH_THRESHOLD:
            rep.add("warnings", sec["name"], text, "not found in the original education; original kept")
    rep.add("kept_original", sec["name"], "education", "facts are kept exactly as in the original")
    return {"entries": entries, "status": "unchanged"}


def _custom(sec: dict, draft: dict, rep: _Report) -> dict:
    supplied = draft.get("custom_sections") or {}
    key = next((k for k in supplied if _sim(k, sec["name"]) >= 0.9 or k.lower() == sec["kind"]), None)
    texts = _items(supplied.get(key)) if key else []
    if texts:
        return _flat_bullets(sec, texts, rep, sec["name"])
    rep.add("kept_original", sec["name"], sec["name"], "no draft field for this section; kept unchanged")
    return {"entries": sec["content"]["entries"], "status": "unchanged"}


def _all_bullets(sections: list[dict]):
    for sec in sections:
        c = sec["content"]
        for co in c.get("companies", []):
            for r in co["roles"]:
                if r["status"] == "mapped":
                    yield sec["name"], r["title"]["text"], r["bullets"]
        for p in c.get("projects", []):
            if p["status"] == "mapped":
                yield sec["name"], p["name"]["text"], p["bullets"]
        if c.get("bullets"):
            yield sec["name"], sec["name"], c["bullets"]


def _check_source(sections: list[dict], rep: _Report) -> None:
    """Draft wording that cannot be traced to the uploaded PDF means the draft was
    built from different resume text (e.g. an older upload). Say so clearly."""
    total = new = 0
    for section, label, bullets in _all_bullets(sections):
        fresh = [b for b in bullets if b["provenance"] == "not_in_original"]
        total += len(bullets)
        new += len(fresh)
        if fresh:
            rep.add("warnings", section, f"{label}: {len(fresh)} of {len(bullets)} bullets are not in the uploaded PDF",
                    "check that the draft was built from this resume")
    if total and new / total >= 0.5:
        rep.add("warnings", "Draft", "source_mismatch",
                f"{new} of {total} bullets cannot be found in the uploaded PDF; the saved profile may come from a different upload")


# ------------------------------------------------------------------------ main

def map_content(grouped: dict[str, Any], draft: dict[str, Any]) -> dict[str, Any]:
    """Build the render plan (see module docstring)."""
    if not isinstance(draft, dict):
        raise ValueError("Draft must be a dictionary")
    rep = _Report()
    present = {s["kind"] for s in grouped["sections"]}

    # header: facts stay as in the original; the headline only fits a tagline row
    header = dict(grouped["header"])
    headline = _clean(draft.get("headline"))
    plan_header = {"name": header["name"], "contacts": header["contacts"], "other": header["other"],
                   "headline": None}
    if headline:
        if header["other"]:
            first = header["other"][0]
            plan_header["headline"] = _slot(headline, first["text"], first["line_ids"])
            if plan_header["headline"]["changed"]:
                rep.add("changed", "Header", "headline")
        else:
            rep.add("unplaced", "Header", headline, "the original has no headline/tagline row")

    # draft sections the original does not have
    for kind, label in (("summary", "summary"), ("skills", "skills"), ("experience", "experience"),
                        ("projects", "projects"), ("achievements", "achievements")):
        value = draft.get(label)
        if value and kind not in present:
            rep.add("unplaced", label.title(), _clean(value)[:80],
                    f"the original has no {label} section; adding one would change the layout")

    sections = []
    for sec in grouped["sections"]:
        kind = sec["kind"]
        if kind == "summary":
            content = _summary(sec, draft, rep)
        elif kind == "skills":
            content = _skills(sec, draft, rep)
        elif kind == "experience":
            content = _experience(sec, draft, rep)
        elif kind == "projects":
            content = _projects(sec, draft, rep)
        elif kind == "achievements":
            content = _flat_bullets(sec, _items(draft.get("achievements")), rep, "achievements")
        elif kind == "education":
            content = _education(sec, draft, rep)
        else:
            content = _custom(sec, draft, rep)
        sections.append({"name": sec["name"], "kind": kind, "heading": sec["heading"], "content": content})

    _check_source(sections, rep)

    for note in _items(draft.get("claims_requiring_confirmation")):
        rep.add("warnings", "Draft", note, "claim flagged for confirmation in the draft")

    return {"version": "1.0", "header": plan_header, "sections": sections, "report": rep.data}
