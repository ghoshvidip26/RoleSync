import re
from dataclasses import replace

import pymupdf
import pytest

from app.services.resume_pdf_renderers.layout_representation import build_layout_representation
from app.services.resume_pdf_renderers.render_validator import _overlaps, _words
from app.services.resume_pdf_renderers.template_renderer import (
    TemplateRenderer, preview_png, sample_draft, sample_profile)
from app.services.resume_pdf_renderers.templates import (
    SECTION_TITLES, TEMPLATES, get_template, list_templates, validate_spec)

IDS = list(TEMPLATES)


def norm(t):
    return re.sub(r"\s+", " ", t).strip()


def text_of(pdf):
    with pymupdf.open(stream=pdf, filetype="pdf") as d:
        return norm(" ".join(p.get_text() for p in d))


def render(tid, profile=None, draft=None):
    return TemplateRenderer(tid).render_with_report(profile or sample_profile(), draft or sample_draft())


def test_there_are_four_distinct_templates():
    assert len(IDS) == 4
    assert len({(t.family, t.header_align, t.section_order, t.entry_layout, t.body_size) for t in TEMPLATES.values()}) == 4


@pytest.mark.parametrize("tid", IDS)
def test_every_template_is_ats_safe_by_construction(tid):
    validate_spec(TEMPLATES[tid])


def test_unsafe_specs_are_rejected():
    base = TEMPLATES["classic"]
    for bad, why in [(replace(base, margins=(20, 43, 36, 36)), "margins"), (replace(base, leading=1.0), "leading"),
                     (replace(base, bullet="★"), "bullet"), (replace(base, body_size=6.0), "body size"),
                     (replace(base, section_order=("summary", "summary")), "twice"),
                     (replace(base, section_order=("summary", "hobbies")), "unknown"),
                     (replace(base, family="script"), "font")]:
        with pytest.raises(ValueError, match="not ATS-safe"):
            validate_spec(bad)


@pytest.mark.parametrize("tid", IDS)
def test_sample_renders_on_one_page_with_all_content(tid):
    pdf, report = render(tid)
    assert report["pages"] == 1 and pdf.startswith(b"%PDF")
    text = text_of(pdf)
    d, p = sample_draft(), sample_profile()
    must = [p["name"], p["email"], d["summary"], d["achievements"][0], "Northwind Labs", "Software Engineer",
            "Jan 2023 – Present", "Austin, TX", "StreamKit", "Go, Kafka, PostgreSQL", "Languages:", "FastAPI"]
    must += [b for e in d["experience"] for b in e["bullets"]] + [b for pr in d["projects"] for b in pr["bullets"]]
    for want in must:
        assert norm(want) in text, (tid, want)
    assert "B.S. in Computer Science | State University, Austin, TX" in text and "Aug 2017 – May 2021" in text


@pytest.mark.parametrize("tid", IDS)
def test_section_order_follows_the_template(tid):
    pdf, report = render(tid)
    with pymupdf.open(stream=pdf, filetype="pdf") as d:
        lines = [l.strip().lower() for l in d[0].get_text().splitlines()]
    spec = TEMPLATES[tid]
    pos = [lines.index(SECTION_TITLES[s].lower()) for s in spec.section_order]     # heading lines only
    assert pos == sorted(pos)
    assert report["sections"] == [SECTION_TITLES[s] for s in spec.section_order]


@pytest.mark.parametrize("tid", IDS)
def test_output_is_clean_for_an_ats(tid):
    pdf, _ = render(tid)
    text = "\n".join(p.get_text() for p in pymupdf.open(stream=pdf, filetype="pdf"))
    assert not any(c in text for c in "�­‐‑\x00")
    assert not any("" <= c <= "" for c in text)
    rep = build_layout_representation(pdf)
    assert not [f for f in rep["ats"]["flags"] if f["severity"] == "warning"], rep["ats"]
    assert rep["ats"]["risk"] == "low"
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    assert not doc[0].get_images() and not doc[0].find_tables(strategy="lines").tables
    assert not _overlaps(_words(doc)[0])


@pytest.mark.parametrize("tid", IDS)
def test_text_stays_inside_the_margins(tid):
    spec = TEMPLATES[tid]
    pdf, _ = render(tid)
    with pymupdf.open(stream=pdf, filetype="pdf") as d:
        for w in d[0].get_text("words"):
            assert w[0] >= spec.margins[0] - 1 and w[2] <= 612 - spec.margins[1] + 1, (tid, w[4])
            assert w[3] <= 792 - spec.margins[3] + 2


@pytest.mark.parametrize("tid", IDS)
def test_contact_details_are_clickable(tid):
    pdf, _ = render(tid)
    with pymupdf.open(stream=pdf, filetype="pdf") as d:
        uris = {l["uri"] for l in d[0].get_links()}
    assert {"mailto:alex.morgan@example.com", "https://linkedin.com/in/alexmorgan",
            "https://github.com/alexmorgan"} <= uris


def test_skills_put_job_relevant_ones_first_and_remove_nothing():
    pdf, _ = render("classic", draft=dict(sample_draft(), skills=["Redis", "Go"]))
    text = text_of(pdf)
    assert "Languages: Go, Python, TypeScript, SQL" in text
    assert "Databases & Tools: Redis, PostgreSQL, Docker, AWS, Git" in text


def test_only_draft_skills_when_profile_has_none():
    pdf, _ = render("modern", profile={"name": "Sam"}, draft={"skills": ["Rust", "Go"], "summary": "Systems engineer."})
    assert "Skills: Rust, Go" in text_of(pdf)


def test_sections_without_content_are_left_out_not_invented():
    pdf, report = render("classic", profile={"name": "Sam", "email": "sam@example.com"},
                         draft={"summary": "Backend engineer."})
    assert report["sections"] == ["Summary"]
    text = text_of(pdf)
    assert "Experience" not in text and "Education" not in text


def test_long_content_flows_to_more_pages_without_losing_text():
    d = sample_draft()
    d["experience"] = [dict(d["experience"][0], bullets=[f"Delivered milestone number {i} with measurable impact across regions"
                                                         for i in range(70)])] + d["experience"][1:]
    for tid in IDS:
        pdf, report = render(tid, draft=d)
        assert report["pages"] > 1
        text = text_of(pdf)
        assert all(f"milestone number {i} " in text for i in range(70)), tid
        with pymupdf.open(stream=pdf, filetype="pdf") as doc:
            for page in doc:
                lines = [l.strip() for l in page.get_text().splitlines() if l.strip()]
                assert lines[-1].lower() not in {t.lower() for t in SECTION_TITLES.values()}, (tid, lines[-1])


def test_education_date_goes_to_the_right_column():
    pdf, _ = render("classic")
    with pymupdf.open(stream=pdf, filetype="pdf") as d:
        words = {w[4]: w for w in d[0].get_text("words")}
    assert words["May"][0] > 400                 # "Aug 2017 – May 2021" is right-aligned


def test_errors():
    with pytest.raises(ValueError, match="Unknown template"):
        TemplateRenderer("fancy")
    with pytest.raises(ValueError, match="name is required"):
        TemplateRenderer("classic").render({}, sample_draft())
    with pytest.raises(ValueError, match="no renderable content"):
        TemplateRenderer("classic").render({"name": "Sam"}, {})
    with pytest.raises(ValueError):
        TemplateRenderer("classic").render(None, {})


def test_gallery_listing_and_previews():
    listing = list_templates()
    assert [t["id"] for t in listing] == IDS
    assert all(t["description"] and t["best_for"] and t["ats_notes"] and t["sections"] for t in listing)
    for tid in IDS:
        assert preview_png(tid).startswith(b"\x89PNG")
    assert get_template(" Classic ").id == "classic"


def test_base_class_api():
    assert TemplateRenderer("compact").render(sample_profile(), sample_draft(), {}).startswith(b"%PDF")


# ------------------------------------------------------------------ fitting

def _longer_draft(extra_bullets):
    d = sample_draft()
    d["experience"][0]["bullets"] = d["experience"][0]["bullets"] + [
        f"Additional supporting achievement number {i} with enough words to wrap onto a second line of text for sure"
        for i in range(extra_bullets)]
    return d


@pytest.mark.parametrize("tid", IDS)
def test_slightly_long_resumes_are_tightened_onto_one_page_within_safety_limits(tid):
    from app.services.resume_pdf_renderers.templates import MIN_BODY_SIZE, MIN_LEADING
    tightened = []
    for extra in range(1, 45, 4):
        pdf, report = render(tid, draft=_longer_draft(extra))
        if report["fit"]["fits_target"] and (report["fit"]["type_scale"] < 1 or report["fit"]["spacing_scale"] < 1):
            tightened.append(extra)
            with pymupdf.open(stream=pdf, filetype="pdf") as d:
                sizes = {round(s["size"], 1) for b in d[0].get_text("dict")["blocks"] for l in b.get("lines", [])
                         for s in l["spans"]}
            assert min(sizes) >= MIN_BODY_SIZE - 1.0           # smallest text is the right-hand dates (-0.3pt) and similar
            assert report["pages"] == 1
    assert tightened, f"{tid}: no input needed (and got) a tighter fit"


def test_content_that_cannot_fit_keeps_the_templates_own_sizes_and_adds_pages():
    pdf, report = render("classic", draft=_longer_draft(60))
    assert report["pages"] > 1 and report["fit"]["fits_target"] is False
    assert report["fit"]["type_scale"] == 1.0 and report["fit"]["spacing_scale"] == 1.0


def test_two_page_target_is_respected():
    pdf, report = TemplateRenderer("classic", target_pages=2).render_with_report(sample_profile(), _longer_draft(30))
    assert report["pages"] <= 2


# ---------------------------------------------------- contacts from the original

def test_contact_details_are_read_from_the_uploaded_resume_and_only_fill_gaps():
    from app.services.resume_pdf_renderers import original_resume
    from test_entry_grouper import Page
    p = Page((612, 792), left=40)
    p.center("Alex Doe", 20)
    p.center("+1 555 010 1234 | alex@old.example.com | linkedin.com/in/alexdoe | github.com/alexdoe", 9, "helv")
    p.heading("Summary")
    p.line("Backend engineer who builds reliable services and data pipelines for payments and more.")
    found = original_resume.contacts_of(p.pdf())
    assert found == {"phone": "+1 555 010 1234", "email": "alex@old.example.com",
                     "linkedin": "linkedin.com/in/alexdoe", "github": "github.com/alexdoe"}
    filled = original_resume.fill_contacts({"name": "Alex", "email": "alex@new.example.com"}, {"contacts": found})
    assert filled["email"] == "alex@new.example.com"             # the profile wins
    assert filled["phone"] == "+1 555 010 1234" and filled["github"] == "github.com/alexdoe"
    assert original_resume.fill_contacts({"name": "A"}, None) == {"name": "A"}
