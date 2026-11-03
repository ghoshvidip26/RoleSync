from pathlib import Path

import pymupdf
import pytest

from app.services.resume_pdf_renderers.layout_manifest import build_layout_manifest

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
W, H = 612, 792
LEFT, RIGHT = 30, 582


def _text(page, x, y, s, size=9):
    page.insert_text((x, y), s, fontname="helv", fontsize=size)


def _width(s, size=9):
    return pymupdf.get_text_length(s, fontname="helv", fontsize=size)


def _fit(prefix, filler, max_width, size=9):
    """Longest prefix + filler words that still fits in max_width."""
    s = prefix
    while _width(s + filler, size) <= max_width:
        s += filler
    return s.strip()


def _synthetic_pdf() -> bytes:
    doc = pymupdf.open()
    p = doc.new_page(width=W, height=H)
    name = "Asha Rao"
    _text(p, (W - _width(name, 20)) / 2, 40, name, 20)           # centered
    _text(p, LEFT, 80, _fit("Summary paragraph starting at the left margin ", "word ", RIGHT - LEFT))
    _text(p, LEFT, 120, "Acme Corp")                              # left
    d = "Jul 2026 - Present"
    _text(p, RIGHT - _width(d), 120, d)                           # right
    _text(p, LEFT + 4, 150, _fit("- Bullet first line that is quite long ", "word ", RIGHT - LEFT - 4))
    wrapped = _fit("wrapped continuation that runs to the right margin ", "word ", RIGHT - LEFT - 13)
    _text(p, LEFT + 13, 162, wrapped)                             # indented, ends near right edge
    _text(p, LEFT + 13, 174, "short wrapped tail")                # indented, short
    out = doc.tobytes()
    doc.close()
    return out


def _align(manifest):
    return {l["text"]: l["alignment"] for pg in manifest["pages"] for l in pg["lines"]}


def _by_prefix(aligns, prefix):
    matches = [v for k, v in aligns.items() if k.startswith(prefix)]
    assert len(matches) == 1, (prefix, list(aligns))
    return matches[0]


def test_synthetic_alignment_cases():
    a = _align(build_layout_manifest(_synthetic_pdf()))
    assert _by_prefix(a, "Asha Rao") == "center"
    assert _by_prefix(a, "Summary paragraph") == "left"
    assert _by_prefix(a, "Acme") == "left"
    assert _by_prefix(a, "Jul 2026") == "right"
    assert _by_prefix(a, "wrapped continuation") == "indented_left"   # was misread as right
    assert _by_prefix(a, "short wrapped") == "indented_left"


def test_wide_line_near_page_center_is_not_center():
    doc = pymupdf.open()
    p = doc.new_page(width=W, height=H)
    _text(p, LEFT, 60, "Left anchor line")
    _text(p, RIGHT - _width("Right anchor"), 60, "Right anchor")
    s = _fit("indented wide line ", "more ", 480)                 # centred on the page, > 70% wide
    _text(p, (W - _width(s)) / 2, 100, s)
    a = _align(build_layout_manifest(doc.tobytes()))
    assert _by_prefix(a, "indented wide") == "indented_left"


@pytest.mark.skipif(not (FIXTURES / "Vidip_Resume.pdf").exists(), reason="sample resume not present")
def test_real_resume_wrapped_bullets_and_right_items():
    m = build_layout_manifest((FIXTURES / "Vidip_Resume.pdf").read_bytes())
    lines = {l["id"]: l for pg in m["pages"] for l in pg["lines"]}
    a = _align(m)
    assert _by_prefix(a, "FCM/WebSocket alerts") == "indented_left"
    assert _by_prefix(a, "shift management") == "indented_left"
    assert _by_prefix(a, "plus Google Maps") == "indented_left"
    assert _by_prefix(a, "Vidip Ghosh") == "center"
    assert _by_prefix(a, "Jul 2026 – Present") == "right"
    # no line is classed as right unless it sits in the right half of the page
    for l in lines.values():
        if l["alignment"] == "right":
            assert l["bbox"][0] >= 612 / 2
