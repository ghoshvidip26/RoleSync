from pathlib import Path

import pymupdf
import pytest

from app.services.resume_pdf_renderers.layout_representation import build_layout_representation
from test_entry_grouper import Page, _design_a, _design_b

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
LETTER = (612, 792)


def _flags(rep):
    return {f["code"]: f for f in rep["ats"]["flags"]}


def _body(p):
    """Enough ordinary resume text to be a real page."""
    p.center("Alex Doe")
    p.center("alex@example.com | github.com/alex", 9, "helv")
    p.heading("Experience")
    p.pair("Acme", "Jan 2022 - Present", "hebo", right_font="hebo")
    for i in range(6):
        p.bullet(f"Delivered feature number {i} with measurable results for customers and the wider "
                 "business across several regions", "continuing on a second wrapped line of text")


# ------------------------------------------------------------ real resume

@pytest.mark.skipif(not (FIXTURES / "Vidip_Resume.pdf").exists(), reason="sample resume not present")
def test_real_resume_representation():
    rep = build_layout_representation((FIXTURES / "Vidip_Resume.pdf").read_bytes())
    page = rep["pages"][0]
    assert (page["width"], page["height"]) == (612.0, 792.0)
    assert rep["document"]["page_size"] == "Letter"
    # section headings: bold 10.96pt, one full-width rule under each of the six
    h = rep["section_headings"]
    assert h["style"]["bold"] and h["style"]["size"] == 10.96 and h["style"]["consistent"]
    assert h["rule_under_every_heading"] is True
    seps = [r for r in rep["rules"] if r["kind"] == "separator"]
    assert len(seps) == 6 and all(r["placement"] == "under_heading" for r in seps)
    # the blue underlines below the contact links are not separators
    assert sum(r["kind"] == "link_underline" for r in rep["rules"]) == 3
    # bullets: glyph + indents
    b = rep["bullets"]
    assert b["count"] == 19 and b["marker"] == "•"
    assert b["hanging_indent"] == pytest.approx(8.65, abs=0.1)
    assert b["marker_font"] == "CMSY5"
    # left/right pairs
    assert rep["pairs"]["right_edge_x"] == pytest.approx(583.2, abs=0.2)
    assert {"company_right", "role_right", "project_right", "education_right"} <= set(rep["pairs"]["roles"])
    # typography per role, including mixed-style lines
    t = rep["typography"]
    assert t["company"]["bold"] and t["role_title"]["italic"]
    assert t["name"]["size"] > 20 and t["name"]["alignment"] == "center"
    fonts = [r["font"] for r in t["project_title"]["run_pattern"]["runs"]]
    assert fonts == ["Lato-Bold", "CMSY9", "Lato-Italic"]
    # spacing is recorded per transition
    assert rep["spacing"]["bullet->bullet"]["gap_median"] == pytest.approx(2.6, abs=0.3)
    assert rep["spacing"]["section_heading->company"]["count"] == 1
    # fonts are embedded subsets, and the resume is ATS-clean
    assert all(f["embedded"] and f["subset"] for f in rep["fonts"])
    assert rep["ats"] == {"risk": "low", "flags": []}


# ----------------------------------------------------------- other designs

def test_design_a_dash_bullets_and_rules():
    rep = build_layout_representation(_design_a())
    assert rep["bullets"]["marker"] == "-"
    assert rep["section_headings"]["style"]["caps"] is True
    assert rep["section_headings"]["rule_under_every_heading"] is True
    assert rep["document"]["page_size"] == "A4"
    # custom headings are kept and reported, not treated as an error
    f = _flags(rep)
    assert f["nonstandard_headings"]["severity"] == "info"
    assert "Leadership" in f["nonstandard_headings"]["evidence"]["headings"]
    assert rep["ats"]["risk"] == "low"


def test_design_b_projects_pairs_and_centered_header():
    rep = build_layout_representation(_design_b())
    assert rep["typography"]["name"]["alignment"] == "center"
    assert "project_right" in rep["pairs"]["roles"]
    assert rep["bullets"]["marker"] == "·"


def test_every_line_has_a_role():
    from app.services.resume_pdf_renderers.layout_manifest import build_layout_manifest
    for pdf in (_design_a(), _design_b()):
        m = build_layout_manifest(pdf)
        rep = build_layout_representation(pdf, manifest=m)
        ids = {l["id"] for p in m["pages"] for l in p["lines"] if l["text"].strip() not in {"|", "-", "·"}}
        assert not ids - set(rep["role_of_line"]), ids - set(rep["role_of_line"])


# ------------------------------------------------------------- ATS flags

def test_clean_resume_has_no_flags():
    p = Page(LETTER, left=40)
    _body(p)
    ats = build_layout_representation(p.pdf())["ats"]
    assert ats["risk"] == "low"
    assert not [f for f in ats["flags"] if f["severity"] == "warning"]


def test_flag_images():
    p = Page(LETTER, left=40)
    _body(p)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 8, 8), False)
    pix.set_rect(pix.irect, (200, 0, 0))
    p.page.insert_image(pymupdf.Rect(500, 40, 540, 80), pixmap=pix)
    rep = build_layout_representation(p.pdf())
    assert _flags(rep)["images"]["severity"] == "warning"
    assert rep["ats"]["risk"] == "medium"


def test_flag_table():
    p = Page(LETTER, left=40)
    _body(p)
    top, left = 400, 40
    for r in range(3):
        p.page.draw_line((left, top + r * 20), (left + 300, top + r * 20))
    for c in range(3):
        p.page.draw_line((left + c * 150, top), (left + c * 150, top + 40))
    p.page.insert_text((left + 5, top + 14), "Python", fontsize=9)
    p.page.insert_text((left + 155, top + 14), "Go", fontsize=9)
    p.page.insert_text((left + 5, top + 34), "Docker", fontsize=9)
    p.page.insert_text((left + 155, top + 34), "AWS", fontsize=9)
    assert "tables" in _flags(build_layout_representation(p.pdf()))


def test_horizontal_rules_alone_are_not_a_table():
    p = Page(LETTER, left=40)
    _body(p)
    assert "tables" not in _flags(build_layout_representation(p.pdf()))


def test_flag_scanned_or_empty_page():
    p = Page(LETTER, left=40)
    p.center("Alex Doe")
    p.line("alex@example.com")
    p.line("Experience")
    flags = _flags(build_layout_representation(p.pdf()))
    assert "little_text" in flags


def test_flag_contact_in_page_edge_zone():
    p = Page(LETTER, left=40)
    _body(p)
    p.page.insert_text((40, 10), "me@example.com", fontsize=8)
    assert "contact_in_margin" in _flags(build_layout_representation(p.pdf()))


def test_flag_two_columns():
    p = Page(LETTER, left=40)
    p.y = 60
    for i in range(6):
        p.page.insert_text((40, 60 + i * 40), f"Left column paragraph number {i} with enough words in it", fontsize=9)
        p.page.insert_text((330, 74 + i * 40), f"Right column paragraph number {i} with enough words in it", fontsize=9)
    flags = _flags(build_layout_representation(p.pdf()))
    assert flags["multi_column"]["severity"] == "warning"


def test_flag_unreadable_characters(monkeypatch):
    p = Page(LETTER, left=40)
    _body(p)
    pdf = p.pdf()
    original = pymupdf.Page.get_text

    def with_private_use(self, *args, **kwargs):      # built-in fonts cannot draw these, so inject
        out = original(self, *args, **kwargs)
        return out + "\ue000\ue001" if not args and not kwargs else out

    monkeypatch.setattr(pymupdf.Page, "get_text", with_private_use)
    flags = _flags(build_layout_representation(pdf))
    assert flags["unreadable_characters"]["evidence"]["count"] == 2


def test_no_contact_details_flag():
    p = Page(LETTER, left=40)
    p.center("Alex Doe")
    p.heading("Experience")
    p.pair("Acme", "2022 - Present", "hebo")
    for i in range(8):
        p.bullet(f"Delivered feature number {i} with measurable results for customers and more words")
    assert "no_contact_details" in _flags(build_layout_representation(p.pdf()))
