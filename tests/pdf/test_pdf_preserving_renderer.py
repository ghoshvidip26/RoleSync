import re
from pathlib import Path

import pymupdf
import pytest

from app.services.resume_pdf_renderers.pdf_preserving_renderer import PdfPreservingRenderer
from test_content_mapper import grouped, identity_draft
from test_entry_grouper import Page, _design_a, _design_b

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
LETTER = (612, 792)


def norm(text):
    return re.sub(r"\s+", " ", text).strip()


def text_of(pdf):
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        return norm(" ".join(p.get_text() for p in doc))


def pages_of(pdf):
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        return doc.page_count


def design_c():
    """Letter, Helvetica: company | location, role | dates, bullets, skills, education."""
    p = Page(LETTER, left=40)
    p.center("Alex Doe", 20)
    p.center("alex@example.com | github.com/alex", 9, "helv")
    p.heading("Summary")
    p.line("Backend engineer who builds reliable services and data pipelines for payments.")
    p.heading("Technical Skills")
    p.line("Languages: Python, Go, SQL")
    p.line("Tools: Docker, Terraform, Kafka")
    p.heading("Experience")
    p.pair("Acme Corp", "Berlin", "hebo", right_font="hebo")
    p.pair("Senior Engineer", "Jan 2022 - Present", "heit", right_font="heit")
    p.bullet("Built payment APIs serving two million users across Europe", "with strict latency budgets.")
    p.bullet("Led a team of four engineers through a platform migration")
    p.pair("Beta Inc", "London", "hebo", right_font="hebo")
    p.pair("Software Engineer", "Jun 2019 - Dec 2021", "heit", right_font="heit")
    p.bullet("Shipped the mobile checkout used by every customer")
    p.heading("Education")
    p.pair("University of Leeds, BSc Computer Science", "Leeds | 2016 - 2019", "hebo")
    return p.pdf()


def exp(company, title, bullets, **kw):
    return dict({"company": company, "job_title": title, "duration": "", "location": "", "bullets": bullets}, **kw)


def render(pdf, draft):
    return PdfPreservingRenderer(pdf).render_with_report(draft)


# --------------------------------------------------------------- identity

@pytest.mark.skipif(not (FIXTURES / "Vidip_Resume.pdf").exists(), reason="sample resume not present")
def test_real_resume_identity_keeps_text_and_page_count():
    original = (FIXTURES / "Vidip_Resume.pdf").read_bytes()
    out, report = render(original, identity_draft(grouped(original)))
    assert report["pages"] == 1 and report["fit"]["fits_original_page_count"]
    assert text_of(out) == text_of(original)               # same words, same order, plain hyphens


@pytest.mark.skipif(not (FIXTURES / "Vidip_Resume.pdf").exists(), reason="sample resume not present")
def test_real_resume_identity_keeps_geometry():
    from app.services.resume_pdf_renderers.layout_manifest import build_layout_manifest
    original = (FIXTURES / "Vidip_Resume.pdf").read_bytes()
    out, _ = render(original, identity_draft(grouped(original)))

    def edges(pdf):
        m = build_layout_manifest(pdf)
        return {(l["text"].strip(), round(l["bbox"][1] / 20)): l["bbox"] for l in m["pages"][0]["lines"]}

    a, b = edges(original), edges(out)
    shared = [k for k in a if k in b and len(k[0]) > 12]
    assert len(shared) >= 20
    # at the original font sizes, left edges do not move and lines stay within a couple of points
    assert max(abs(a[k][0] - b[k][0]) for k in shared) < 1.5


def test_design_c_identity_round_trip():
    pdf = design_c()
    out, report = render(pdf, identity_draft(grouped(pdf)))
    assert report["pages"] == 1
    assert text_of(out) == text_of(pdf)


def test_design_b_identity_round_trip():
    pdf = _design_b()
    out, _ = render(pdf, identity_draft(grouped(pdf)))
    assert text_of(out) == text_of(pdf)


# --------------------------------------------------------------- tailoring

def test_new_content_replaces_old_and_structure_is_kept():
    pdf = design_c()
    draft = {
        "summary": "Platform engineer focused on payments reliability.",
        "skills": ["Kafka", "Go"],
        "experience": [
            exp("Beta Inc", "Software Engineer", ["Cut checkout errors by half"]),
            exp("Acme Corp", "Senior Engineer", ["Rebuilt the ledger service", "Mentored three engineers"]),
        ],
    }
    out, report = render(pdf, draft)
    text = text_of(out)
    for want in ("Platform engineer focused on payments reliability.", "Rebuilt the ledger service",
                 "Mentored three engineers", "Cut checkout errors by half", "Berlin", "Jan 2022 - Present",
                 "Languages: Go, Python, SQL", "Tools: Kafka, Docker, Terraform"):
        assert want in text, want
    for gone in ("Built payment APIs", "Led a team of four", "Shipped the mobile checkout"):
        assert gone not in text, gone
    # sections keep their order and company order stays as in the original
    order = [text.index(h) for h in ("Summary", "Technical Skills", "Experience", "Education")]
    assert order == sorted(order)
    assert text.index("Acme Corp") < text.index("Beta Inc")
    assert report["plan_report"]["omitted"] == []


def test_omitted_entry_is_not_drawn_and_is_reported():
    pdf = design_c()
    out, report = render(pdf, {"experience": [exp("Acme Corp", "Senior Engineer", ["Only this"])]})
    text = text_of(out)
    assert "Only this" in text and "Beta Inc" not in text
    assert any("Beta Inc" in r["what"] for r in report["plan_report"]["omitted"])


def test_added_entry_is_drawn_in_the_same_style_and_flagged():
    pdf = design_c()
    draft = {"experience": [exp("Acme Corp", "Senior Engineer", ["a"]), exp("Beta Inc", "Software Engineer", ["b"]),
                            exp("Gamma LLC", "Intern", ["c"], duration="2018")]}
    out, report = render(pdf, draft)
    assert "Gamma LLC" in text_of(out)
    assert any("Gamma LLC" in r["what"] for r in report["plan_report"]["added"])


# ---------------------------------------------------------------- reflow

def test_long_content_flows_onto_more_pages_without_losing_text_or_overflowing():
    pdf = design_c()
    bullets = [f"Delivered milestone number {i} with measurable impact across several regions and teams" for i in range(60)]
    out, report = render(pdf, {"experience": [exp("Acme Corp", "Senior Engineer", bullets),
                                              exp("Beta Inc", "Software Engineer", ["x"])]})
    assert report["pages"] > 1 and not report["fit"]["fits_original_page_count"]
    text = text_of(out)
    assert all(f"milestone number {i} " in text for i in range(60))
    with pymupdf.open(stream=out, filetype="pdf") as doc:
        for page in doc:
            lowest = max(b[3] for b in (w[:4] for w in page.get_text("words")))
            assert lowest <= page.rect.height - 20


def test_headings_are_not_left_alone_at_page_bottom():
    pdf = design_c()
    bullets = [f"Delivered milestone number {i} with measurable impact across several regions" for i in range(40)]
    out, _ = render(pdf, {"experience": [exp("Acme Corp", "Senior Engineer", bullets),
                                         exp("Beta Inc", "Software Engineer", ["x"])]})
    with pymupdf.open(stream=out, filetype="pdf") as doc:
        for page in doc:
            lines = [l for l in page.get_text().splitlines() if l.strip()]
            assert lines[-1].strip() not in {"Summary", "Technical Skills", "Experience", "Education"}


def test_slightly_longer_content_is_fitted_to_the_original_page_count():
    """Adding lines until the page overflows at full size must trigger spacing/font tightening, not a new page."""
    pdf = design_c()
    original_pages = pages_of(pdf)
    fitted = []
    for extra in range(4, 40, 2):
        draft = identity_draft(grouped(pdf))
        for e in draft["experience"]:
            e["bullets"] = e["bullets"] + ["An additional supporting bullet that adds one more line of content"] * extra
        out, report = render(pdf, draft)
        if report["fit"]["fits_original_page_count"] and (report["fit"]["line_spacing_scale"] < 1.0
                                                          or report["fit"]["font_scale"] < 1.0):
            fitted.append(extra)
            assert pages_of(out) == original_pages
    assert fitted, "no input size needed (and got) a tighter fit"


# ------------------------------------------------------------- refusal / API

def test_unsupported_experience_format_is_refused():
    with pytest.raises(ValueError, match="one row"):
        render(_design_a(), {"experience": [exp("Acme Corp", "Senior Engineer", ["x"])]})


def test_two_column_original_is_refused():
    p = Page(LETTER, left=40)
    for i in range(6):
        p.page.insert_text((40, 60 + i * 40), f"Left column paragraph number {i} with enough words in it", fontsize=9)
        p.page.insert_text((330, 74 + i * 40), f"Right column paragraph number {i} with enough words in it", fontsize=9)
    with pytest.raises(ValueError, match="single-column"):
        render(p.pdf(), {})


def test_rejects_non_pdf_input():
    with pytest.raises(ValueError):
        PdfPreservingRenderer(b"not a pdf")


def test_render_api_returns_pdf_bytes():
    pdf = design_c()
    out = PdfPreservingRenderer(pdf).render({}, identity_draft(grouped(pdf)), {})
    assert out.startswith(b"%PDF")


def test_hyphens_extract_as_plain_hyphens():
    pdf = design_c()
    out, _ = render(pdf, {"summary": "Real-time, low-latency and event-driven systems."})
    assert "Real-time, low-latency and event-driven" in text_of(out)


# ------------------------------------------------------ text-map repair (hyphen and space)

CMAP = (b"/CIDInit /ProcSet findresource begin\n1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n"
        b"3 beginbfchar\n<0003> <00a0>\n<0010> <2010>\n<0011> <0041>\nendbfchar\n"
        b"3 beginbfrange\n<0020> <0020> <00AD>\n<0030> <0032> <2010>\n<0040> <0040> <202f>\nendbfrange\nendcmap\n")


def test_cmap_lookalikes_become_plain_characters():
    from app.services.resume_pdf_renderers.pdf_preserving_renderer import _normalize_cmap
    out, n = _normalize_cmap(CMAP)
    assert n == 4
    assert b"<0003> <0020>" in out and b"<0010> <002d>" in out and b"<0011> <0041>" in out
    assert b"<0020> <0020> <002d>" in out and b"<0040> <0040> <0020>" in out
    assert b"<0030> <0032> <2010>" in out                 # a range of several codes is not rewritten
    assert _normalize_cmap(out)[1] == 0                   # idempotent


def test_a_real_font_file_extracts_with_plain_spaces_and_hyphens():
    """Any installed TrueType font that maps look-alikes to one glyph (Lato, Arial, ...) used to leak
    U+00A0 / U+2010 into the PDF's text; the finished PDF must not."""
    import glob
    candidates = glob.glob("/System/Library/Fonts/Supplemental/Arial.ttf") + glob.glob("/Library/Fonts/*.ttf") \
        + glob.glob("/usr/share/fonts/**/DejaVuSans.ttf", recursive=True)
    if not candidates:
        pytest.skip("no TrueType font installed")
    from app.services.resume_pdf_renderers.pdf_preserving_renderer import _fix_text_maps
    doc = pymupdf.open()
    page = doc.new_page()
    tw = pymupdf.TextWriter(page.rect)
    tw.append((50, 50), "real-time React Native", font=pymupdf.Font(fontfile=candidates[0]), fontsize=10)
    tw.write_text(page)
    _fix_text_maps(doc)
    text = page.get_text()
    assert "real-time React Native" in text and " " not in text and "‐" not in text


# ------------------------------- the user's own design must survive ordinary edits (found by trying edits on a real resume)

def _edited(pdf, change):
    draft = identity_draft(grouped(pdf))
    change(draft)
    out, report = render(pdf, draft)
    return out, report, draft


def _lowest_text(pdf):
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        return [max(w[3] for w in page.get_text("words")) for page in doc]


def test_text_stays_inside_the_page_even_when_content_is_added_until_it_overflows():
    pdf = design_c()                                         # a short resume on a tall page: it may grow to fill the page
    out, report, _ = _edited(pdf, lambda d: d["experience"][0]["bullets"].extend(
        ["An additional supporting bullet that fills lines of the page with more content"] * 90))
    with pymupdf.open(stream=out, filetype="pdf") as doc:
        assert doc.page_count > 1
        for page in doc:
            assert max(w[3] for w in page.get_text("words")) <= page.rect.height - 25, "text runs into the bottom margin"


@pytest.mark.skipif(not (FIXTURES / "Vidip_Resume.pdf").exists(), reason="sample resume not present")
@pytest.mark.parametrize("name, change", [
    ("a project removed", lambda d: d["projects"].pop()),
    ("a role removed", lambda d: d["experience"].pop()),
    ("long achievement", lambda d: d["achievements"].append("An achievement that goes on for a long time " * 12)),
    ("symbols", lambda d: d["experience"][0]["bullets"].append("Cut costs by ≈30% → faster “checkout” — résumé café ✓")),
    ("new summary", lambda d: d.update(summary="Mobile engineer who ships React Native apps with real-time features.")),
    ("extra bullets", lambda d: [e["bullets"].extend(["An extra supporting line of experience text that wraps fully"] * 3) for e in d["experience"]]),
])
def test_ordinary_edits_to_the_real_resume_still_give_a_valid_pdf_in_the_users_own_design(name, change):
    from app.services.resume_pdf_renderers.render_validator import validate_rendering
    original = (FIXTURES / "Vidip_Resume.pdf").read_bytes()
    out, _, draft = _edited(original, change)
    verdict = validate_rendering(original, out, draft)
    failed = [(c["name"], c["detail"]) for c in verdict["checks"] if c["status"] == "fail"]
    assert verdict["passed"], (name, failed)
