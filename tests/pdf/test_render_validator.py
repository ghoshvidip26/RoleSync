from pathlib import Path

import pymupdf
import pytest

from app.services.resume_pdf_renderers.pdf_preserving_renderer import PdfPreservingRenderer
from app.services.resume_pdf_renderers.render_validator import (
    comparison_image, expected_texts, validate_rendering)
from test_content_mapper import grouped, identity_draft
from test_pdf_preserving_renderer import design_c, exp
from test_entry_grouper import Page

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
LETTER = (612, 792)


def status(report, name):
    return next(c for c in report["checks"] if c["name"] == name)["status"]


def rendered(pdf, draft):
    return PdfPreservingRenderer(pdf).render_with_report(draft)[0]


def edit(pdf, fn):
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    fn(doc)
    out = doc.tobytes()
    doc.close()
    return out


# --------------------------------------------------------------- good output

def test_identity_render_passes_every_check():
    pdf = design_c()
    draft = identity_draft(grouped(pdf))
    report = validate_rendering(pdf, rendered(pdf, draft), draft)
    assert report["passed"] and report["summary"]["fail"] == 0
    assert report["summary"]["warn"] == 0, [c for c in report["checks"] if c["status"] != "pass"]
    assert report["metrics"]["visual_similarity"] > 0.9


def test_tailored_render_passes():
    pdf = design_c()
    draft = {"summary": "Platform engineer focused on payments reliability.", "skills": ["Kafka"],
             "experience": [exp("Acme Corp", "Senior Engineer", ["Rebuilt the ledger service", "Mentored engineers"]),
                            exp("Beta Inc", "Software Engineer", ["Cut checkout errors by half"])]}
    report = validate_rendering(pdf, rendered(pdf, draft), draft)
    assert report["passed"], [c for c in report["checks"] if c["status"] == "fail"]


@pytest.mark.skipif(not (FIXTURES / "Vidip_Resume.pdf").exists(), reason="sample resume not present")
def test_real_resume_identity_passes_with_high_similarity():
    pdf = (FIXTURES / "Vidip_Resume.pdf").read_bytes()
    draft = identity_draft(grouped(pdf))
    report = validate_rendering(pdf, rendered(pdf, draft), draft)
    assert report["passed"], [c for c in report["checks"] if c["status"] == "fail"]
    assert status(report, "links_preserved") == "pass"          # header links and project-name links
    assert report["metrics"]["visual_similarity"] > 0.95


def test_plan_can_be_given_instead_of_draft():
    from app.services.resume_pdf_renderers.content_mapper import map_content
    pdf = design_c()
    draft = identity_draft(grouped(pdf))
    plan = map_content(grouped(pdf), draft)
    assert validate_rendering(pdf, rendered(pdf, draft), plan=plan)["passed"]


# ------------------------------------------------------------ bad output

def test_missing_text_fails():
    pdf = design_c()
    draft = identity_draft(grouped(pdf))
    good = rendered(pdf, draft)

    def remove(doc):
        page = doc[0]
        for r in page.search_for("Shipped the mobile checkout"):
            page.add_redact_annot(r)
        page.apply_redactions()

    report = validate_rendering(pdf, edit(good, remove), draft)
    assert not report["passed"] and status(report, "text_complete") == "fail"
    assert any("Shipped" in m for m in next(c for c in report["checks"] if c["name"] == "text_complete")["evidence"]["missing"])


def test_overlapping_text_fails():
    pdf = design_c()
    draft = identity_draft(grouped(pdf))
    good = rendered(pdf, draft)
    spot = pymupdf.open(stream=good, filetype="pdf")[0].search_for("Backend engineer")[0]
    bad = edit(good, lambda d: d[0].insert_text((spot.x0 + 2, spot.y1 - 2), "Overlapping words here", fontsize=9))
    report = validate_rendering(pdf, bad, draft)
    assert status(report, "no_overlap") == "fail" and not report["passed"]


def test_text_outside_margins_fails():
    pdf = design_c()
    draft = identity_draft(grouped(pdf))
    bad = edit(rendered(pdf, draft), lambda d: d[0].insert_text((3, 400), "stray margin text", fontsize=9))
    assert status(validate_rendering(pdf, bad, draft), "inside_margins") == "fail"


def test_unreadable_characters_fail(monkeypatch):
    pdf = design_c()
    draft = identity_draft(grouped(pdf))
    good = rendered(pdf, draft)
    original = pymupdf.Page.get_text

    def dirty(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        return out + "real‐time �" if not args and not kwargs else out

    monkeypatch.setattr(pymupdf.Page, "get_text", dirty)
    report = validate_rendering(pdf, good, draft)
    ext = next(c for c in report["checks"] if c["name"] == "extraction_clean")
    assert ext["status"] == "fail" and "unicode hyphen" in ext["evidence"]["found"]


def test_reordered_sections_fail():
    pdf = design_c()
    draft = identity_draft(grouped(pdf))
    p = Page(LETTER, left=40)                     # same words, Education moved above Experience
    p.center("Alex Doe", 20)
    p.center("alex@example.com | github.com/alex", 9, "helv")
    p.heading("Summary")
    p.line("Backend engineer who builds reliable services and data pipelines for payments.")
    p.heading("Education")
    p.pair("University of Leeds, BSc Computer Science", "Leeds | 2016 - 2019", "hebo")
    p.heading("Technical Skills")
    p.line("Languages: Python, Go, SQL")
    p.line("Tools: Docker, Terraform, Kafka")
    p.heading("Experience")
    report = validate_rendering(pdf, p.pdf(), draft)
    assert status(report, "section_order") == "fail"


def test_lost_hyperlink_fails():
    p = Page(LETTER, left=40)
    p.center("Alex Doe", 20)
    p.center("alex@example.com | github.com/alex", 9, "helv")
    p.heading("Summary")
    p.line("Backend engineer who builds reliable services and data pipelines for payments.")
    with_link = edit(p.pdf(), lambda d: d[0].insert_link({
        "kind": pymupdf.LINK_URI, "uri": "https://github.com/alex",
        "from": d[0].search_for("github.com/alex")[0]}))
    without_link = p.pdf() if False else edit(with_link, lambda d: [d[0].delete_link(l) for l in d[0].get_links()])
    report = validate_rendering(with_link, without_link, identity_draft(grouped(with_link)))
    assert status(report, "links_preserved") == "fail"


def test_links_survive_a_real_render():
    p = Page(LETTER, left=40)
    p.center("Alex Doe", 20)
    p.center("alex@example.com | github.com/alex", 9, "helv")
    p.heading("Summary")
    p.line("Backend engineer who builds reliable services and data pipelines for payments.")
    with_link = edit(p.pdf(), lambda d: d[0].insert_link({
        "kind": pymupdf.LINK_URI, "uri": "https://github.com/alex",
        "from": d[0].search_for("github.com/alex")[0]}))
    draft = identity_draft(grouped(with_link))
    report = validate_rendering(with_link, rendered(with_link, draft), draft)
    assert status(report, "links_preserved") == "pass"


def test_extra_pages_warn_not_fail():
    pdf = design_c()
    bullets = [f"Delivered milestone number {i} with measurable impact across several regions" for i in range(60)]
    draft = {"experience": [exp("Acme Corp", "Senior Engineer", bullets), exp("Beta Inc", "Software Engineer", ["x"])]}
    report = validate_rendering(pdf, rendered(pdf, draft), draft)
    assert status(report, "page_count") == "warn" and report["passed"]


def test_cramped_lines_warn():
    pdf = design_c()
    draft = identity_draft(grouped(pdf))
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    for i in range(12):
        page.insert_text((40, 60 + i * 8), f"A tightly packed line of text number {i}", fontsize=9)
    assert status(validate_rendering(pdf, doc.tobytes(), draft), "line_spacing") == "warn"


# -------------------------------------------------------------- helpers

def test_expected_texts_use_clean_education_text():
    from app.services.resume_pdf_renderers.content_mapper import map_content
    from app.services.resume_pdf_renderers.layout_manifest import build_layout_manifest
    pdf = design_c()
    m = build_layout_manifest(pdf)
    plan = map_content(grouped(pdf), identity_draft(grouped(pdf)))
    lines = {l["id"]: l for p in m["pages"] for l in p["lines"]}
    spans = {s["id"]: s for p in m["pages"] for s in p["spans"]}
    texts = expected_texts(plan, lines, spans)
    assert "Acme Corp" in texts and any("University of Leeds" in t for t in texts)


def test_comparison_image_is_a_png():
    pdf = design_c()
    png = comparison_image(pdf, rendered(pdf, identity_draft(grouped(pdf))))
    assert png.startswith(b"\x89PNG")


def test_input_validation():
    pdf = design_c()
    with pytest.raises(ValueError):
        validate_rendering(b"nope", pdf, {})
    with pytest.raises(ValueError):
        validate_rendering(pdf, pdf)


def test_non_breaking_spaces_are_reported(monkeypatch):
    pdf = design_c()
    draft = identity_draft(grouped(pdf))
    good = rendered(pdf, draft)
    original = pymupdf.Page.get_text

    def with_nbsp(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        return out.replace("React Native", "React Native") + "\nReact Native" if not args and not kwargs else out

    monkeypatch.setattr(pymupdf.Page, "get_text", with_nbsp)
    ext = next(c for c in validate_rendering(pdf, good, draft)["checks"] if c["name"] == "extraction_clean")
    assert ext["status"] == "fail" and "non-breaking space" in ext["evidence"]["found"]
