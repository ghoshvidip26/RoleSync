"""Phase 5/6: rebuild a resume PDF with the ORIGINAL design and the APPROVED content.

The original PDF is analysed (manifest -> entries -> layout representation), the
approved draft is mapped onto it (content mapper), and each page is rebuilt from
the measured design: same page size, fonts, sizes, colors, alignment, bullet
glyph and indents, section rules, left/right columns and vertical rhythm.
Only the words change. Text is re-wrapped, later content moves down, and a new
page is added when the content no longer fits.

Scope (anything else raises ``ValueError`` so the caller can use a fallback
renderer): single-column resumes whose experience entries use a company row
(company | location) followed by role rows (title | dates).

Fonts, in order: a font file with the original's name (``RESUME_FONT_DIRS``,
``app/assets/fonts`` or the system), the font embedded in the original PDF
(only when it contains every character to be drawn - embedded fonts are
subsets), then a base-14 look-alike. The choice is made per font family so a
resume never mixes two families, and is reported in the result.
"""

from __future__ import annotations

import copy
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import pymupdf

from .base import BaseResumeRenderer
from .content_mapper import map_content
from .entry_grouper import group_entries
from .layout_manifest import build_layout_manifest
from .layout_representation import _rows, build_layout_representation

MAX_SHRINK = 0.20               # word spaces may compress by this much to fit a line (TeX does the same)
DEFAULT_MARGIN_BETWEEN = 8.0    # minimum gap between left text and right-aligned text
MAX_PAGES = 6
BASELINE_ROOM = 6.0             # pt kept free below the last baseline (descenders, tall glyph boxes)
MIN_LEADING = 1.12              # never set lines closer than this multiple of the font size
# (line-spacing scale, font scale), gentlest first; used to fit the original page count
FIT_STEPS = [(1.0, 1.0), (0.96, 1.0), (0.92, 1.0), (0.92, 0.97), (0.88, 0.95), (0.86, 0.93)]
ASSET_FONT_DIR = Path(__file__).resolve().parents[2] / "assets" / "fonts"
_FONT_EXT = {".ttf", ".otf"}


def _med(values, default=0.0):
    values = [v for v in values if v is not None]
    return float(median(values)) if values else default


def _rgb(hex_color: str | None) -> tuple[float, float, float]:
    h = (hex_color or "#000000").lstrip("#")
    if not re.fullmatch(r"[0-9a-fA-F]{6}", h):
        h = "000000"
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def _clean(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


# ============================================================================
# fonts
# ============================================================================

class FontBook:
    """Chooses one font per (font name) for everything that will be drawn."""

    def __init__(self, original_pdf: bytes, chars_by_font: dict[str, set[str]], font_dirs: list[Path]):
        self.report: list[dict] = []
        self.faces: dict[str, tuple[pymupdf.Font, str]] = {}
        self._base: dict[str, pymupdf.Font] = {}
        embedded = self._embedded(original_pdf)
        files = self._find_files(font_dirs, set(chars_by_font))

        def letters(name):
            chars = chars_by_font[name]
            return sum(c.isalpha() for c in chars) / max(1, len(chars))

        text_fonts = [n for n in chars_by_font if letters(n) >= 0.4]
        glyph_fonts = [n for n in chars_by_font if n not in text_fonts]

        families: dict[str, list[str]] = defaultdict(list)
        for name in text_fonts:
            families[name.split("-")[0].lower()].append(name)

        for fam, names in families.items():
            level, loaded = "base14", {}
            for candidate, source in (("file", files), ("embedded", embedded)):
                trial = {}
                for n in names:
                    font = source.get(n)
                    if font is None or not self._covers(font, chars_by_font[n]):
                        trial = None
                        break
                    trial[n] = font
                if trial is not None:
                    level, loaded = candidate, trial
                    break
            for n in names:
                if level == "base14":
                    self.faces[n] = (self._substitute(n), "base14")
                else:
                    self.faces[n] = (loaded[n], level)
        for n in glyph_fonts:                      # separators, bullet glyphs: judged one by one
            font = files.get(n) or embedded.get(n)
            if font is not None and self._covers(font, chars_by_font[n]):
                self.faces[n] = (font, "file" if n in files else "embedded")
            else:
                self.faces[n] = (self._substitute(n, sans=True), "base14")
        for n, (font, level) in sorted(self.faces.items()):
            self.report.append({"font": n, "source": level, "used_face": font.name,
                                "kind": "text" if n in text_fonts else "symbol"})

    @staticmethod
    def _covers(font: pymupdf.Font, chars: set[str]) -> bool:
        return all(font.has_glyph(ord(c)) for c in chars if not c.isspace())

    @staticmethod
    def _embedded(pdf: bytes) -> dict[str, pymupdf.Font]:
        out: dict[str, pymupdf.Font] = {}
        with pymupdf.open(stream=pdf, filetype="pdf") as doc:
            for page in doc:
                for f in page.get_fonts(full=True):
                    clean = f[3].split("+", 1)[-1]
                    if clean in out:
                        continue
                    try:
                        _, _, _, buf = doc.extract_font(f[0])
                        if buf:
                            out[clean] = pymupdf.Font(fontbuffer=buf)
                    except Exception:
                        continue
        return out

    @staticmethod
    def _find_files(dirs: list[Path], wanted: set[str]) -> dict[str, pymupdf.Font]:
        keys = {re.sub(r"[\s_]", "", w).lower(): w for w in wanted}
        found: dict[str, pymupdf.Font] = {}
        for d in dirs:
            if not d.is_dir():
                continue
            try:
                for path in d.rglob("*"):
                    if path.suffix.lower() not in _FONT_EXT:
                        continue
                    key = re.sub(r"[\s_]", "", path.stem).lower()
                    if key in keys and keys[key] not in found:
                        try:
                            found[keys[key]] = pymupdf.Font(fontfile=str(path))
                        except Exception:
                            continue
            except OSError:
                continue
        return found

    def _substitute(self, name: str, sans: bool = False) -> pymupdf.Font:
        low = name.lower()
        bold = "bold" in low or "black" in low or "heavy" in low or low.endswith("-b")
        italic = "italic" in low or "oblique" in low
        if sans or not re.search(r"times|serif|georgia|garamond|cambria|palatino|minion|book|roman|mono|cour|consol|menlo", low) \
                or "sans" in low:
            base = ["helv", "hebo", "heit", "hebi"]
        elif re.search(r"mono|cour|consol|menlo", low):
            base = ["cour", "cobo", "coit", "cobi"]
        else:
            base = ["tiro", "tibo", "tiit", "tibi"]
        code = base[(1 if bold else 0) + (2 if italic else 0)]
        if code not in self._base:
            self._base[code] = pymupdf.Font(code)
        return self._base[code]

    def font(self, name: str) -> pymupdf.Font:
        if name not in self.faces:
            self.faces[name] = (self._substitute(name), "base14")
        return self.faces[name][0]


def _font_dirs(extra: list[Path] | None) -> list[Path]:
    dirs = [Path(p) for p in (extra or [])]
    dirs += [Path(p) for p in os.environ.get("RESUME_FONT_DIRS", "").split(os.pathsep) if p]
    dirs += [ASSET_FONT_DIR, Path.home() / "Library/Fonts", Path("/Library/Fonts"),
             Path("/System/Library/Fonts/Supplemental"), Path("/usr/share/fonts"), Path("/usr/local/share/fonts")]
    return dirs


# ============================================================================
# layout units
# ============================================================================

class Seg:
    """A run of text in one style, positioned relative to the previous run by `gap`."""

    def __init__(self, text: str, style: dict, gap: float | None = 0.0, orig: tuple[float, float] | None = None,
                 lead: bool = False, trail: bool = False):
        # gap None = one normal space; lead/trail = a space that sits inside the original span
        self.text, self.style, self.gap, self.orig = _clean(text), style, gap, orig
        self.lead, self.trail = lead, trail


class Unit:
    """One logical line (or wrapped paragraph) of the resume."""

    def __init__(self, role: str, left: list[Seg], right: list[Seg] | None = None, *, align: str = "left",
                 x: float = 0.0, wrap: bool = True, text_x: float | None = None, marker: Seg | None = None,
                 rule: bool = False, keep_next: bool = False, orig_line: dict | None = None,
                 row_items: list | None = None):
        self.role, self.left, self.right = role, left, right or []
        self.align, self.x, self.wrap, self.text_x = align, x, wrap, text_x
        self.marker, self.rule, self.keep_next = marker, rule, keep_next
        self.orig_line, self.row_items = orig_line, row_items or []


class _Original:
    """Everything measured from the original PDF that the renderer needs."""

    def __init__(self, pdf: bytes, manifest: dict, grouped: dict, rep: dict):
        self.pdf, self.manifest, self.grouped, self.rep = pdf, manifest, grouped, rep
        self.lines = {l["id"]: l for p in manifest["pages"] for l in p["lines"]}
        self.spans = {s["id"]: s for p in manifest["pages"] for s in p["spans"]}
        self.roles = rep["role_of_line"]
        self.typo = rep["typography"]
        page = rep["pages"][0]
        self.page_w, self.page_h = page["width"], page["height"]
        self.margin_top = page["margins"]["top"]
        self.right_edge = rep["pairs"]["right_edge_x"] or page["content_box"][2]
        self.left_edge = page["content_box"][0]
        self.baseline = {i: self._baseline(l) for i, l in self.lines.items()}
        self.role_x = self._role_x()
        self.pattern = self._patterns()
        self.pitch = self._pitches()
        self.rule_style = self._rule_style()
        self.pitch_scale = 1.0

    def _baseline(self, line: dict) -> float:
        for sid in line["spans"]:
            sp = self.spans.get(sid)
            if sp and sp["text"].strip():
                return sp["origin"][1]
        return line["bbox"][3]

    def runs(self, line: dict) -> list[dict]:
        """Spans merged into runs of one style, with x extent and the gap before each."""
        runs: list[dict] = []
        for sid in line["spans"]:
            sp = self.spans.get(sid)
            if not sp or not sp["text"].strip():
                continue
            key = (sp["font"], round(sp["font_size"], 2), sp["color"])
            x0, x1 = sp["bbox"][0], sp["bbox"][2]
            gap = x0 - runs[-1]["x1"] if runs else 0.0
            if runs and runs[-1]["key"] == key and gap < 0.45 * sp["font_size"]:
                # same style and a word-space or less apart: one run (space only where there is a gap)
                joiner = " " if gap > 0.12 * sp["font_size"] and not runs[-1]["text"][-1:].isspace() \
                    and not sp["text"][:1].isspace() else ""
                runs[-1]["text"] += joiner + sp["text"]
                runs[-1]["x1"] = x1
            else:
                runs.append({"key": key, "text": sp["text"], "x0": x0, "x1": x1,
                             "style": {"font": sp["font"], "size": sp["font_size"], "color": sp["color"],
                                       "bold": sp["style"]["bold"], "italic": sp["style"]["italic"]}})
        for i, r in enumerate(runs):
            r["lead"], r["trail"] = r["text"][:1].isspace(), r["text"][-1:].isspace()
            r["text"] = _clean(r["text"])
            r["gap"] = max(0.0, r["x0"] - runs[i - 1]["x1"]) if i else 0.0
        return [r for r in runs if r["text"]]

    def _role_x(self) -> dict[str, float]:
        xs = defaultdict(list)
        for i, role in self.roles.items():
            if i in self.lines:
                xs[role].append(self.lines[i]["bbox"][0])
        return {r: _med(v) for r, v in xs.items()}

    def _patterns(self) -> dict[str, list[dict]]:
        """Most common run-style sequence per role, with measured gaps."""
        by_role: dict[str, list[list[dict]]] = defaultdict(list)
        for i, role in self.roles.items():
            if i in self.lines:
                runs = self.runs(self.lines[i])
                if runs:
                    by_role[role].append(runs)
        out = {}
        for role, seqs in by_role.items():
            sig, _ = Counter(tuple(r["key"] for r in s) for s in seqs).most_common(1)[0]
            pick = [s for s in seqs if tuple(r["key"] for r in s) == sig]
            out[role] = [{"style": r["style"], "gap": _med([s[k]["gap"] for s in pick]),
                          "lead": pick[0][k]["lead"], "trail": pick[0][k]["trail"]} for k, r in enumerate(pick[0])]
        return out

    def _pitches(self) -> dict[tuple[str, str], float]:
        table = defaultdict(list)
        for page in self.manifest["pages"]:
            rows = _rows(page["lines"])
            for a, b in zip(rows, rows[1:]):
                ra, rb = self.roles.get(a["ids"][0]), self.roles.get(b["ids"][0])
                if ra and rb:
                    table[(ra, rb)].append(self.baseline[b["ids"][0]] - self.baseline[a["ids"][0]])
        return {k: _med(v) for k, v in table.items()}

    def _rule_style(self) -> dict | None:
        seps = [r for r in self.rep["rules"] if r["kind"] == "separator" and r.get("placement") == "under_heading"]
        if not seps or not self.rep["section_headings"]["rule_under_every_heading"]:
            return None
        heads = [(self.baseline[i], i) for i, role in self.roles.items()
                 if role == "section_heading" and i in self.baseline]
        offsets = []
        for r in seps:
            above = [(r["y"] - b) for b, _ in heads if 0 <= r["y"] - b <= 14]
            if above:
                offsets.append(min(above))
        return {"x0": _med([r["x0"] for r in seps]), "x1": _med([r["x1"] for r in seps]),
                "thickness": _med([r["thickness"] for r in seps], 0.5),
                "color": seps[0]["color"], "offset": _med(offsets, 2.5)}

    def separator(self) -> dict | None:
        """Style and spacing of the '|' that sits between side-by-side items (skills rows)."""
        for l in self.lines.values():
            if l["text"].strip() == "|":
                runs = self.runs(l)
                if not runs:
                    continue
                row = [m for m in self.lines.values()
                       if m is not l and abs(m["bbox"][1] - l["bbox"][1]) < 4]
                left = [m["bbox"][2] for m in row if m["bbox"][2] <= l["bbox"][0] + 0.5]
                right = [m["bbox"][0] for m in row if m["bbox"][0] >= l["bbox"][2] - 0.5]
                return {"style": runs[0]["style"],
                        "before": l["bbox"][0] - max(left) if left else 8.0,
                        "after": min(right) - l["bbox"][2] if right else 8.0}
        return None

    def pitch_for(self, prev_role: str | None, role: str, size: float) -> float:
        if prev_role is None:
            return 0.0
        if (prev_role, role) in self.pitch:
            value = self.pitch[(prev_role, role)]
        else:
            same = [v for (a, b), v in self.pitch.items() if b == role]
            value = _med(same, size * 1.3)
        # tighten by the fit scale, but never below a readable leading (or above the original)
        return max(value * self.pitch_scale, min(value, MIN_LEADING * size))

    def style_for(self, role: str) -> dict:
        t = self.typo.get(role) or self.typo.get("summary_text") or next(iter(self.typo.values()))
        return {"font": t["font"], "size": t["size"], "color": t["color"], "bold": t["bold"], "italic": t["italic"]}

    def pattern_for(self, role: str) -> list[dict]:
        return self.pattern.get(role) or [{"style": self.style_for(role), "gap": 0.0, "lead": False, "trail": False}]


# ============================================================================
# composing units from the render plan
# ============================================================================

class _Composer:
    def __init__(self, o: _Original):
        self.o = o

    # ---- helpers
    def _segs(self, role: str, pieces: list[str]) -> list[Seg]:
        """Place texts onto the role's run pattern (e.g. bold name, separator, italic tech)."""
        pat = self.o.pattern_for(role)
        if len(pieces) > len(pat):                   # fewer runs in the original than pieces: one run
            head, tail = list(pieces[:len(pat) - 1]), pieces[len(pat) - 1:]
            merged = tail[0]
            for extra in tail[1:]:
                merged += ("" if extra[:1] in ":,;.)" else " ") + extra
            pieces = head + [merged]
        segs = []
        for k, text in enumerate(pieces):
            p = pat[min(k, len(pat) - 1)]
            segs.append(Seg(text, p["style"], p["gap"] if k < len(pat) else 4.0,
                            lead=p.get("lead", False) and k > 0, trail=p.get("trail", False) and k < len(pieces) - 1))
        return [s for s in segs if s.text]

    def _single(self, role: str, text: str) -> list[Seg]:
        st = self.o.style_for(role)
        return [Seg(text, st)] if _clean(text) else []

    def copy_units(self, line_ids: list[str], headline: dict | None = None) -> list[Unit]:
        """Reproduce original lines unchanged: runs, styles, alignment, right-hand items."""
        o = self.o
        lines = [o.lines[i] for i in line_ids if i in o.lines]
        units = []
        for row in _rows(lines):
            items = sorted((o.lines[i] for i in row["ids"]), key=lambda l: l["bbox"][0])
            left_items = [l for l in items if l["alignment"] != "right"] or items[:1]
            right_items = [l for l in items if l not in left_items]
            left, right = [], []
            for target, group in ((left, left_items), (right, right_items)):
                prev_x1 = None
                for l in group:
                    for n, r in enumerate(o.runs(l)):
                        gap = r["gap"] if n else (max(0.0, r["x0"] - prev_x1) if prev_x1 is not None else 0.0)
                        text = headline["text"] if headline and l["id"] == headline["line_id"] else r["text"]
                        target.append(Seg(text, r["style"], gap, (r["x0"], r["x1"]), r["lead"], r["trail"]))
                        prev_x1 = r["x1"]
            first = left_items[0]
            center = (first["bbox"][0] + first["bbox"][2]) / 2
            align = "center" if first["alignment"] == "center" else "left"
            role = o.roles.get(first["id"], "entry_text")
            units.append(Unit(role, left, right, align=align,
                              x=center if align == "center" else first["bbox"][0],
                              wrap=False, orig_line=first, row_items=items))
        return units

    def emphasis(self) -> list[tuple[str, dict]]:
        """Bold/italic phrases the original uses inside bullets, longest first."""
        if hasattr(self, "_emph"):
            return self._emph
        o = self.o
        base = o.style_for("bullet")
        groups: list[list[dict]] = []
        ordered = sorted((l for l in o.lines.values() if o.roles.get(l["id"]) in {"bullet", "bullet_cont"}),
                         key=lambda l: (int(l["id"].split("_")[0][1:]), l["bbox"][1], l["bbox"][0]))
        for l in ordered:
            runs = [r for r in o.runs(l) if r["text"] not in {"•", "-", "·", "–", "—", "*"}]
            if o.roles[l["id"]] == "bullet" or not groups:
                groups.append([])
            groups[-1] += runs
        phrases: dict[str, dict] = {}
        for runs in groups:
            cur, style = [], None
            for r in runs + [None]:
                emph = r is not None and (r["style"]["bold"] != base["bold"] or r["style"]["italic"] != base["italic"])
                if emph:
                    cur.append(r["text"])
                    style = style or r["style"]
                elif cur:
                    phrase = " ".join(cur)
                    if len(phrase) >= 3:
                        phrases.setdefault(phrase, style)
                    cur, style = [], None
        self._emph = sorted(phrases.items(), key=lambda kv: -len(kv[0]))
        return self._emph

    def _emphasize(self, text: str, base: dict) -> list[Seg]:
        """Split text into plain and emphasised pieces using the original's phrases."""
        claimed: list[tuple[int, int, dict]] = []
        for phrase, style in self.emphasis():
            pat = r"\s+".join(re.escape(w) for w in phrase.split())
            pat = (r"(?<![A-Za-z0-9])" if phrase[0].isalnum() else "") + pat + (r"(?![A-Za-z0-9])" if phrase[-1].isalnum() else "")
            for m in re.finditer(pat, text):
                if all(m.end() <= a or m.start() >= b for a, b, _ in claimed):
                    claimed.append((m.start(), m.end(), style))
        claimed.sort()
        cuts, pieces, pos = [], [], 0
        for a, b, style in claimed:
            if a > pos:
                pieces.append((text[pos:a], base))
            pieces.append((text[a:b], style))
            pos = b
        if pos < len(text):
            pieces.append((text[pos:], base))
        segs, prev_raw = [], ""
        for raw, style in pieces:
            if not raw.strip():
                prev_raw = raw
                continue
            spaced = bool(segs) and (raw[:1].isspace() or prev_raw[-1:].isspace())
            segs.append(Seg(raw, style, None if spaced else 0.0))
            prev_raw = raw
        return segs

    def bullet(self, slot: dict, role_hint: str = "bullet") -> Unit:
        o = self.o
        st = o.style_for("bullet")
        b = o.rep["bullets"] or {}
        text = _clean(slot["text"])
        segs = None
        # keep a bold lead-in ("Winner: ...") when the original bullet had one
        tmpl = slot.get("template") or []
        runs = o.runs(o.lines[tmpl[0]]) if tmpl and tmpl[0] in o.lines else []
        body = [r for r in runs if r["text"] not in {"•", "-", "·", "–"}]
        if len(body) >= 2 and body[0]["style"]["bold"] and not body[1]["style"]["bold"]:
            head, sep, rest = text.partition(":")
            if sep and len(head) <= 70:
                segs = [Seg(head + ":", body[0]["style"])] + self._emphasize(rest, body[1]["style"])
                if len(segs) > 1:
                    segs[1].gap = None             # the space after the colon
        if segs is None:
            segs = self._emphasize(text, st)
        marker = None
        if b.get("marker"):
            msize = b.get("marker_size", st["size"])
            marker = Seg(b["marker"], {"font": b.get("marker_font", st["font"]), "size": msize,
                                       "color": b.get("marker_color", st["color"]),
                                       "bold": False, "italic": False})
        line_x = b.get("line_x0") or o.role_x.get("bullet", o.left_edge)
        text_x = b.get("wrapped_line_x0") or o.role_x.get("bullet_cont", line_x + 9)
        return Unit("bullet", [s for s in segs if s.text], x=line_x, text_x=text_x, marker=marker)

    def text_block(self, role: str, text: str) -> Unit:
        return Unit(role, self._single(role, text), x=self.o.role_x.get(role, self.o.left_edge))

    # ---- sections
    def section(self, sec: dict) -> list[Unit]:
        o = self.o
        units = [Unit("section_heading", self._single("section_heading", sec["heading"]["text"]),
                      x=o.role_x.get("section_heading", o.left_edge), rule=True, keep_next=True)]
        kind, c = sec["kind"], sec["content"]
        if kind == "summary":
            units.append(self.text_block("summary_text", c["paragraph"]["text"]))
        elif kind == "skills":
            units += self._skills(c)
        elif kind == "experience":
            units += self._experience(c)
        elif kind == "projects":
            units += self._projects(c)
        elif c.get("bullets") is not None:
            units += [self.bullet(b) for b in c["bullets"]]
        else:                                                    # unchanged content: copy as is
            ids = []
            if kind == "education":
                ids = [i for e in c["entries"] for i in e["line_ids"]]
            else:
                for e in c.get("entries", []):
                    if e.get("head"):
                        ids += e["head"]["line_ids"]
                    ids += [i for it in e["items"] for i in it["line_ids"]]
            units += self.copy_units(ids)
        return units

    def _skills(self, c: dict) -> list[Unit]:
        o = self.o
        rows: list[list[dict]] = []
        for g in c["groups"]:
            y = o.baseline.get(g["template"][0]) if g["template"] else None
            if rows and y is not None and rows[-1][0].get("_y") is not None and abs(rows[-1][0]["_y"] - y) < 2.5:
                rows[-1].append(dict(g, _y=y))
            else:
                rows.append([dict(g, _y=y)])
        # The rows that cover the most of what the job asks for come first, right after the first row (Languages): a frontend job
        # moves the Frontend row up. A row is kept whole (groups that share a line stay together), so the page length does not change.
        if len(rows) > 2:
            hits = lambda row: sum(len(g.get("highlighted") or []) for g in row)
            rows = [rows[0], *sorted(rows[1:], key=lambda row: -hits(row))]           # a stable sort: equal rows keep their order
        units = []
        for row in rows:
            segs: list[Seg] = []
            sep = o.separator()
            for k, g in enumerate(row):
                pieces = [g["label"], ": " + g["values"]["text"]] if g["label"] else [g["values"]["text"]]
                gsegs = self._segs("skills_row", pieces)
                if k and gsegs:
                    style = sep["style"] if sep else gsegs[0].style
                    segs.append(Seg("|", style, sep["before"] if sep else 8.0))
                    gsegs[0].gap = sep["after"] if sep else 8.0
                segs += gsegs
            units.append(Unit("skills_row", segs, x=o.role_x.get("skills_row", o.left_edge)))
        return units

    def _experience(self, c: dict) -> list[Unit]:
        o = self.o
        units = []
        for co in c["companies"]:
            if len(co["template"]) >= 2 and not co["location"]["original"] and co["status"] == "mapped":
                raise ValueError("Experience entries with company and dates on one row are not supported yet")
            units.append(Unit("company", self._segs("company", [co["company"]["text"]]),
                              self._single("company_right", co["location"]["text"]),
                              x=o.role_x.get("company", o.left_edge), keep_next=True))
            for r in co["roles"]:
                units.append(Unit("role_title", self._segs("role_title", [r["title"]["text"]]),
                                  self._single("role_right", r["duration"]["text"]),
                                  x=o.role_x.get("role_title", o.left_edge), keep_next=True))
                detail_ids = [i for d in r["details"] for i in d["line_ids"]]
                units += self.copy_units(detail_ids)
                units += [self.bullet(b) for b in r["bullets"]]
        return units

    def _projects(self, c: dict) -> list[Unit]:
        o = self.o
        units = []
        for p in c["projects"]:
            pieces = [p["name"]["text"]] + (["|", p["technologies"]["text"]] if p["technologies"]["text"] else [])
            units.append(Unit("project_title", self._segs("project_title", pieces),
                              self._single("project_right", p["dates"]["text"]),
                              x=o.role_x.get("project_title", o.left_edge), keep_next=True))
            for d in p["description"]:
                units.append(self.text_block("project_description", d["text"]))
            units += [self.bullet(b) for b in p["bullets"]]
        return units

    def header(self, plan_header: dict) -> list[Unit]:
        ids = self.o.grouped["header"]["line_ids"]
        headline = None
        if plan_header.get("headline") and plan_header["headline"]["template"]:
            headline = {"line_id": plan_header["headline"]["template"][0], "text": plan_header["headline"]["text"]}
        return self.copy_units(ids, headline)


# ============================================================================
# placement and drawing
# ============================================================================

# Several code points can share one glyph (a hyphen glyph is also U+2010; a space glyph is also U+00A0), and
# the text map PyMuPDF writes picks the wrong one: "real-time" would extract as "real\u2010time" and "React Native"
# as "React\u00a0Native", and an ATS keyword search would miss both. Map them back to the ordinary characters.
_LOOKALIKES = {"002d": ("2010", "2011", "00ad"), "0020": ("00a0", "202f", "2007")}
_TARGET_FOR = {alt: plain for plain, alts in _LOOKALIKES.items() for alt in alts}
_ALT = "|".join(sorted(_TARGET_FOR))
_BFRANGE = re.compile(rb"<([0-9a-fA-F]+)>\s*<([0-9a-fA-F]+)>\s*<(" + _ALT.encode() + rb")>", re.I)
_BFCHAR = re.compile(rb"^(<[0-9a-fA-F]+>)\s*<(" + _ALT.encode() + rb")>\s*$", re.I | re.M)


def _normalize_cmap(data: bytes) -> tuple[bytes, int]:
    """A ToUnicode CMap with hyphen/space look-alike targets replaced by the plain character, and how many."""
    count = 0

    def rng(m):
        nonlocal count
        if m.group(1).lower() != m.group(2).lower():          # a range of several codes: leave it alone
            return m.group(0)
        count += 1
        return m.group(0)[: m.start(3) - m.start(0)] + _TARGET_FOR[m.group(3).decode().lower()].encode() + b">"

    def chr_(m):
        nonlocal count
        count += 1
        return m.group(1) + b" <" + _TARGET_FOR[m.group(2).decode().lower()].encode() + b">"

    return _BFCHAR.sub(chr_, _BFRANGE.sub(rng, data)), count


def _fix_text_maps(doc: pymupdf.Document) -> int:
    """Apply ``_normalize_cmap`` to every font's ToUnicode map in the document. Returns how many were changed."""
    changed = 0
    for xref in range(1, doc.xref_length()):
        try:
            kind, value = doc.xref_get_key(xref, "ToUnicode")
        except Exception:
            continue
        if kind != "xref":
            continue
        ref = int(value.split()[0])
        data = doc.xref_stream(ref)
        if not data:
            continue
        new, n = _normalize_cmap(data)
        if n:
            doc.update_stream(ref, new)
            changed += n
    return changed


_fix_hyphen_maps = _fix_text_maps          # earlier name, kept for callers that still import it


# things the renderer cannot reproduce: it redraws text, rules and links only
_BLOCKERS = {"multi_column", "images", "tables"}


class PdfPreservingRenderer(BaseResumeRenderer):
    """Rebuilds the original resume PDF with the approved draft's content."""

    def __init__(self, original_pdf: bytes, font_dirs: list[Path] | None = None):
        if not isinstance(original_pdf, (bytes, bytearray)) or not original_pdf.startswith(b"%PDF"):
            raise ValueError("original_pdf must be PDF bytes")
        self.original_pdf = bytes(original_pdf)
        self.font_dirs = _font_dirs(font_dirs)
        self.last_report: dict[str, Any] = {}

    @staticmethod
    def _scaled(units: list[Unit], font_scale: float) -> list[Unit]:
        if font_scale == 1.0:
            return units
        out = copy.deepcopy(units)
        for u in out:
            for seg in u.left + u.right + ([u.marker] if u.marker else []):
                seg.style = dict(seg.style, size=seg.style["size"] * font_scale)
                if seg.gap:
                    seg.gap *= font_scale
        return out

    # BaseResumeRenderer API
    def render(self, profile: dict[str, Any], draft: dict[str, Any], layout_metadata: dict[str, Any]) -> bytes:
        return self.render_with_report(draft)[0]

    def render_with_report(self, draft: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
        manifest = build_layout_manifest(self.original_pdf)
        grouped = group_entries(manifest)
        rep = build_layout_representation(self.original_pdf, manifest, grouped)
        blockers = sorted({f["code"] for f in rep["ats"]["flags"] if f["code"] in _BLOCKERS})
        if blockers:
            raise ValueError("Only single-column resumes without images or tables can be rebuilt with their "
                             f"original design (found: {', '.join(blockers)})")
        if not grouped["sections"]:
            raise ValueError("No resume sections were recognised in the original PDF")
        plan = map_content(grouped, draft)
        o = _Original(self.original_pdf, manifest, grouped, rep)

        composer = _Composer(o)
        units = composer.header(plan["header"])
        for sec in plan["sections"]:
            units += composer.section(sec)

        chars: dict[str, set[str]] = defaultdict(set)
        for u in units:
            for s in u.left + u.right + ([u.marker] if u.marker else []):
                chars[s.style["font"]].update(s.text)
        fonts = FontBook(self.original_pdf, chars, self.font_dirs)

        target = manifest["document"]["page_count"]
        best = None
        for pitch_scale, font_scale in FIT_STEPS:
            o.pitch_scale = pitch_scale
            doc = pymupdf.open()
            try:
                report = _Drawer(o, fonts, doc).draw(self._scaled(units, font_scale))
            except ValueError:                       # too many pages at this step; try a tighter one
                doc.close()
                continue
            try:
                doc.subset_fonts()               # embed only the glyphs used (1 MB -> ~120 KB); optional
            except Exception:
                pass
            _fix_text_maps(doc)
            report["fit"] = {"line_spacing_scale": pitch_scale, "font_scale": font_scale,
                             "fits_original_page_count": report["pages"] <= target,
                             "extra_pages": max(0, report["pages"] - target)}
            if best is None:
                best = (doc, report)                 # the untouched design, used if nothing fits
            if report["pages"] <= target:
                if best[0] is not doc:
                    best[0].close()
                best = (doc, report)
                break
            if best[0] is not doc:
                doc.close()
        if best is None:
            raise ValueError("The resume does not fit in the allowed number of pages")
        doc, report = best
        report.update({"fonts": fonts.report, "plan_report": plan["report"], "original_pages": target})
        out = doc.tobytes(garbage=3, deflate=True)
        doc.close()
        with pymupdf.open(stream=out, filetype="pdf") as check:
            if not any(p.get_text().strip() for p in check):
                raise ValueError("Generated PDF has no extractable text")
        self.last_report = report
        return out, report


class _Drawer:
    def __init__(self, o: _Original, fonts: FontBook, doc: pymupdf.Document):
        self.o, self.fonts, self.doc = o, fonts, doc
        # The usable page ends one top-margin above the bottom edge (the validator's rule too). Baselines stop a
        # little earlier so descenders and tall glyphs such as "|" stay inside it; an original that itself runs lower
        # keeps its own extent.
        lowest = max(o.baseline.values(), default=0.0)
        self.bottom = max(o.page_h - o.margin_top - BASELINE_ROOM, lowest + 2.0)
        first = min(o.baseline.values()) if o.baseline else o.margin_top + 10
        self.first_baseline = first
        self.page = None
        self.baseline = 0.0
        self.prev_role: str | None = None
        self.pages = 0
        self.warnings: list[str] = []
        self._marker_cache: dict = {}
        self.src = pymupdf.open(stream=o.pdf, filetype="pdf")
        self.text_font = o.style_for("bullet")["font"]          # supplies the space after symbol glyphs
        # hyperlinks are re-attached by the text they sat on (header links, project names, ...)
        self.link_anchors: list[tuple[str, str]] = []
        for page in self.src:
            for link in page.get_links():
                if link.get("uri"):
                    anchor = re.sub(r"\s+", " ", page.get_textbox(pymupdf.Rect(link["from"]))).strip()
                    if anchor:
                        self.link_anchors.append((anchor, link["uri"]))

    # ---- measuring
    def width(self, seg: Seg) -> float:
        return self.fonts.font(seg.style["font"]).text_length(seg.text, fontsize=seg.style["size"])

    def advance(self, seg: Seg) -> float:
        return self.width(seg) + self.space(seg.style) * (int(seg.lead) + int(seg.trail))

    def space(self, style: dict) -> float:
        return self.fonts.font(style["font"]).text_length(" ", fontsize=style["size"])

    def _marker_style(self, seg: Seg) -> Seg:
        """If the marker font was substituted, match the original glyph advance."""
        key = (seg.style["font"], seg.text)
        if key not in self._marker_cache:
            self._marker_cache[key] = self._marker_style_uncached(seg)
        return self._marker_cache[key]

    def _marker_style_uncached(self, seg: Seg) -> Seg:
        font, level = self.fonts.faces.get(seg.style["font"], (None, ""))
        if level == "base14" and font is not None:
            target = None
            for line in self.o.lines.values():
                for sid in line["spans"]:
                    sp = self.o.spans[sid]
                    if sp["font"] == seg.style["font"] and sp["text"].strip() == seg.text:
                        target = sp["bbox"][2] - sp["bbox"][0]
                        break
                if target:
                    break
            unit = font.text_length(seg.text, fontsize=1)
            if target and unit:
                return Seg(seg.text, dict(seg.style, size=target / unit))
        return seg

    # ---- wrapping
    def _wrap(self, segs: list[Seg], width: float, first_width: float) -> list[list[tuple[Seg, float]]]:
        tokens, prev = [], None
        for seg in segs:
            for i, word in enumerate(seg.text.split()):
                gap = None
                if i == 0:
                    gap = seg.gap
                    if gap is not None:
                        gap += self.space(seg.style) * (int(seg.lead) + int(bool(prev and prev.trail)))
                tokens.append((word, seg.style, gap))
            prev = seg
        lines: list[list[tuple[Seg, float]]] = [[]]
        limits = [first_width]
        words_w = gaps_w = 0.0

        def close(line_index: int):
            """A line slightly too wide is fitted by compressing its word spaces."""
            line = lines[line_index]
            natural = words_w + gaps_w
            if line_index < len(limits) and natural > limits[line_index] and gaps_w > 0:
                scale = max(1 - MAX_SHRINK, (limits[line_index] - words_w) / gaps_w)
                lines[line_index] = [(sg, g * scale if i else 0.0) for i, (sg, g) in enumerate(line)]

        for word, style, gap in tokens:
            w = Seg(word, style)
            ww = self.width(w)
            gap_w = (gap if gap is not None else self.space(style)) if lines[-1] else 0.0
            limit = limits[-1]
            room = MAX_SHRINK * (gaps_w + gap_w)            # how much this line could still be squeezed
            if lines[-1] and words_w + gaps_w + gap_w + ww > limit + room:
                close(len(lines) - 1)
                lines.append([])
                limits.append(width)
                words_w = gaps_w = 0.0
                gap_w, limit = 0.0, width
            while ww > limit and not lines[-1]:             # a single word wider than the line
                cut = len(w.text) - 1
                while cut > 1 and self.width(Seg(w.text[:cut], style)) > limit:
                    cut -= 1
                lines[-1].append((Seg(w.text[:cut], style), 0.0))
                lines.append([])
                limits.append(width)
                w = Seg(w.text[cut:], style)
                ww = self.width(w)
                words_w = gaps_w = 0.0
            lines[-1].append((w, gap_w))
            words_w += ww
            gaps_w += gap_w
        close(len(lines) - 1)
        return [l for l in lines if l]

    def _merge(self, line: list[tuple[Seg, float]]) -> list[tuple[Seg, float]]:
        """Join consecutive words of one style into a single text run."""
        out: list[tuple[Seg, float]] = []
        for seg, gap in line:
            if out and out[-1][0].style == seg.style and abs(gap - self.space(seg.style)) < 0.01:
                prev = out[-1][0]
                out[-1] = (Seg(prev.text + " " + seg.text, prev.style), out[-1][1])
            else:
                out.append((seg, gap))
        return out

    # ---- pages
    def new_page(self):
        if self.pages >= MAX_PAGES:
            raise ValueError(f"Resume would exceed {MAX_PAGES} pages")
        self.page = self.doc.new_page(width=self.o.page_w, height=self.o.page_h)
        self.pages += 1
        self.baseline = self.first_baseline
        self.prev_role = None

    def _text(self, x: float, y: float, seg: Seg, trailing_space: bool = False):
        font = self.fonts.font(seg.style["font"])
        size = seg.style["size"]
        # written in reading order, one run at a time: batching by color would pull blue links away
        # from the black text beside them and split every row into several lines on extraction
        tw = pymupdf.TextWriter(self.page.rect)
        if trailing_space and font.has_glyph(32):
            tw.append((x, y), seg.text + " ", font=font, fontsize=size)
        else:
            tw.append((x, y), seg.text, font=font, fontsize=size)
            if trailing_space:      # symbol fonts have no space glyph; take it from the body font
                tw.append((x + font.text_length(seg.text, fontsize=size), y), " ",
                          font=self.fonts.font(self.text_font), fontsize=size)
        tw.write_text(self.page, color=_rgb(seg.style["color"]))
        text = re.sub(r"\s+", " ", seg.text).strip()
        for anchor, uri in self.link_anchors:
            # a name must match the whole run; a URL/e-mail may sit inside a longer run ("a@b.com | github.com/x")
            inside = bool(re.search(r"[./@]", anchor)) and len(anchor) >= 4
            idx = text.find(anchor) if inside else (0 if anchor == text else -1)
            if idx < 0:
                continue
            x0 = x + font.text_length(text[:idx], fontsize=size)
            rect = pymupdf.Rect(x0, y - size * 0.95, x0 + font.text_length(anchor, fontsize=size), y + size * 0.3)
            self.page.insert_link({"kind": pymupdf.LINK_URI, "from": rect, "uri": uri})

    # ---- main loop
    def _lead(self, u: Unit) -> float:
        return 0.0 if self.prev_role is None else self.o.pitch_for(self.prev_role, u.role, self._size(u))

    def _inner(self, u: Unit, lines) -> float:
        return sum(self.o.pitch_for(self._line_role(u, k - 1), self._line_role(u, k), self._size(u))
                   for k in range(1, len(lines)))

    def draw(self, units: list[Unit]) -> dict:
        self.new_page()
        for i, u in enumerate(units):
            lines = self._layout(u)
            fresh = self.prev_role is None
            start = self.first_baseline if fresh else self.baseline + self._lead(u)
            span = self._inner(u, lines)
            if u.keep_next and i + 1 < len(units):          # a heading never sits alone at the page bottom
                nxt = units[i + 1]
                nl = self._layout(nxt)
                last = self._line_role(u, len(lines) - 1)
                span += self.o.pitch_for(last, nxt.role, self._size(nxt)) + self._inner(nxt, nl[:2])
            if not fresh and start + span > self.bottom:
                self.new_page()
                start = self.first_baseline
            self._place(u, lines, start)
        self.src.close()
        return {"pages": self.pages, "warnings": self.warnings}

    @staticmethod
    def _size(u: Unit) -> float:
        segs = u.left or u.right
        return segs[0].style["size"] if segs else 9.0

    def _layout(self, u: Unit):
        """Split a unit into lines of (seg, gap) with the available widths."""
        o = self.o
        right_w = sum(self.advance(s) + (s.gap or 0) for s in u.right)
        if u.marker or u.wrap:
            x0 = u.text_x if u.text_x is not None else u.x
            width = max(10.0, o.right_edge - x0)
            first = width - (right_w + DEFAULT_MARGIN_BETWEEN if right_w else 0)
            return self._wrap(u.left, width, max(10.0, first))
        # copied lines keep their runs; re-wrap only when they no longer fit
        total = sum(self.advance(s) + (s.gap or 0) for s in u.left) + (right_w + DEFAULT_MARGIN_BETWEEN if right_w else 0)
        avail = (o.right_edge - u.x) if u.align != "center" else (o.right_edge - o.left_edge)
        if total <= avail + 0.5:
            return [[(s, s.gap or 0.0) for s in u.left]]
        return self._wrap(u.left, max(10.0, o.right_edge - u.x), max(10.0, o.right_edge - u.x - right_w))

    def _line_role(self, u: Unit, index: int) -> str:
        if index == 0:
            return u.role
        return "bullet_cont" if u.role == "bullet" else u.role

    def _place(self, u: Unit, lines, start: float):
        for k, line in enumerate(lines):
            role = self._line_role(u, k)
            if k == 0:
                y = start
            else:
                y = self.baseline + self.o.pitch_for(self._line_role(u, k - 1), role, self._size(u))
                if y > self.bottom:                          # a long paragraph continues on the next page
                    self.new_page()
                    y = self.first_baseline
            self.baseline = y
            self._draw_line(u, line, y, first=(k == 0))
            self.prev_role = role

    def _draw_line(self, u: Unit, line: list[tuple[Seg, float]], y: float, first: bool):
        o = self.o
        line = self._merge(line) if not u.orig_line else line
        o = self.o
        total = sum(self.advance(sg) + (g if i else 0) for i, (sg, g) in enumerate(line))
        if u.marker:
            x0 = u.text_x
            if first:
                self._text(u.x, y, self._marker_style(u.marker), trailing_space=True)
        elif u.align == "center":
            x0 = u.x - total / 2
        else:
            x0 = u.x
        x = x0
        spans = []
        for i, (seg, gap) in enumerate(line):
            if i:
                x += gap
            if seg.lead:
                x += self.space(seg.style)
            nxt_gap = line[i + 1][1] if i + 1 < len(line) else 0.0
            # a real space character keeps words separate for text extraction (ATS) even when
            # the visible gap was squeezed; it is invisible and the next word is drawn over its advance
            self._text(x, y, seg, trailing_space=nxt_gap >= 0.5 * self.space(seg.style) and not seg.trail)
            w = self.width(seg)
            spans.append((seg, x, x + w))
            x += w + (self.space(seg.style) if seg.trail else 0.0)
        if first and u.right:
            rsegs = u.right
            rw = sum(self.advance(sg) + ((sg.gap or 0) if i else 0) for i, sg in enumerate(rsegs))
            x = o.right_edge - rw
            for i, sg in enumerate(rsegs):
                if i:
                    x += sg.gap or 0
                if sg.lead:
                    x += self.space(sg.style)
                self._text(x, y, sg)
                x += self.width(sg) + (self.space(sg.style) if sg.trail else 0.0)
        if first and u.orig_line:
            self._decorate(u, spans, y)
        if first and u.rule and o.rule_style:
            r = o.rule_style
            self.page.draw_line((r["x0"], y + r["offset"]), (r["x1"], y + r["offset"]),
                                color=_rgb(r["color"]), width=r["thickness"])

    def _decorate(self, u: Unit, spans, y: float):
        """Re-create link underlines and hyperlinks that sat on a copied line."""
        o = self.o
        ob = o.baseline.get(u.orig_line["id"], y)
        for rule in o.rep["rules"]:
            if rule["kind"] != "link_underline" or rule["page"] != 1:
                continue
            if not (0 <= rule["y"] - ob <= 8):
                continue
            hit = [(s, a, b) for s, a, b in spans if s.orig and s.orig[0] - 1 <= rule["x1"] and rule["x0"] <= s.orig[1] + 1
                   and min(rule["x1"], s.orig[1]) - max(rule["x0"], s.orig[0]) > 0.5 * (s.orig[1] - s.orig[0])]
            if hit:
                a, b = hit[0][1], hit[-1][2]
                self.page.draw_line((a, y + rule["y"] - ob), (b, y + rule["y"] - ob),
                                    color=_rgb(rule["color"]), width=rule["thickness"])
