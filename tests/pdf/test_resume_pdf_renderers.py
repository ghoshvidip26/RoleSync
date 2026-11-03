import pymupdf
import pytest

from app.services.resume_pdf_renderers.renderers.single_column import SingleColumnRenderer
from app.services.resume_pdf_renderers.renderers.two_column import TwoColumnRenderer
from app.services.resume_pdf_renderers.router import get_renderer

PROFILE = {"name": "Asha Rao", "email": "asha@example.com", "phone": "123"}
DRAFT = {
    "headline": "Backend Engineer",
    "summary": "Built services – I’ve shipped “fast” APIs.",
    "skills": ["Python", "Go"],
    "experience": [{"company": "Acme", "job_title": "Engineer", "duration": "Jan 2024 – Present",
                    "location": "Remote", "bullets": ["Built APIs", "Cut latency 30%"]}],
    "projects": [{"name": "Tool", "technologies": ["Python"], "description": "A tool", "bullets": ["Did x"]}],
    "achievements": ["Won a hackathon"],
    "education": [{"details": "B.Tech Computer Engineering"}],
}


def text_of(pdf):
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        return "\n".join(p.get_text() for p in doc)


# ---------- router ----------

@pytest.mark.parametrize("layout,cls", [("single_column", SingleColumnRenderer), ("two_column", TwoColumnRenderer)])
def test_router_picks_renderer(layout, cls):
    assert isinstance(get_renderer({"layout_type": layout, "confidence": 0.9}), cls)


def test_router_normalises_layout_type():
    assert isinstance(get_renderer({"layout_type": "  Two_Column ", "confidence": 0.9}), TwoColumnRenderer)


def test_router_confidence_boundary():
    assert isinstance(get_renderer({"layout_type": "single_column", "confidence": 0.65}), SingleColumnRenderer)
    with pytest.raises(ValueError, match="too low"):
        get_renderer({"layout_type": "single_column", "confidence": 0.649})


def test_router_accepts_numeric_string_confidence():
    assert isinstance(get_renderer({"layout_type": "single_column", "confidence": "0.8"}), SingleColumnRenderer)


@pytest.mark.parametrize("bad", [None, "high", True, [], {}])
def test_router_bad_confidence_is_value_error(bad):
    with pytest.raises(ValueError, match="confidence"):
        get_renderer({"layout_type": "single_column", "confidence": bad})


def test_router_missing_confidence_and_nan():
    with pytest.raises(ValueError):
        get_renderer({"layout_type": "single_column"})
    with pytest.raises(ValueError):
        get_renderer({"layout_type": "single_column", "confidence": float("nan")})


def test_router_unsupported_and_non_dict():
    with pytest.raises(ValueError, match="not supported"):
        get_renderer({"layout_type": "complex", "confidence": 0.99})
    with pytest.raises(ValueError, match="dictionary"):
        get_renderer(None)


# ---------- single column ----------

def single_column_metadata():
    return {"layout_type": "single_column", "confidence": 0.8, "pages": [{
        "width": 595.28, "height": 841.89,
        "margins_points": {"left": 40, "right": 40, "top": 40, "bottom": 40},
        "dominant_fonts": ["Helvetica"], "dominant_font_sizes": [9.0],
        "dominant_text_colors": ["#000000", "#0046a0"]}]}


def test_single_column_rejects_other_layouts():
    with pytest.raises(ValueError, match="single_column"):
        SingleColumnRenderer().render(PROFILE, DRAFT, {})


def test_single_column_renders_text_and_keeps_unicode():
    pdf = SingleColumnRenderer().render(PROFILE, DRAFT, single_column_metadata())
    assert pdf.startswith(b"%PDF")
    text = text_of(pdf)
    assert "Asha Rao" in text and "Cut latency 30%" in text
    assert "–" in text and "’" in text and "“" in text
    assert "B.Tech Computer Engineering" in text
    assert "details:" not in text


# ---------- two column ----------

def test_two_column_sidebar_left_of_main():
    pdf = TwoColumnRenderer().render(PROFILE, DRAFT, {})
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        words = doc[0].get_text("words")
    x = {}
    for w in words:
        x.setdefault(w[4].upper(), w[0])
    assert x["SKILLS"] < x["EXPERIENCE"] and x["EDUCATION"] < x["EXPERIENCE"]
    assert x["SKILLS"] < 100 < 200 < x["EXPERIENCE"]


def test_two_column_text_and_no_details_leak():
    text = text_of(TwoColumnRenderer().render(PROFILE, DRAFT, {}))
    assert "Asha Rao" in text and "Built APIs" in text and "B.Tech Computer Engineering" in text
    assert "details:" not in text


def test_two_column_flows_onto_more_pages():
    draft = dict(DRAFT, experience=[{"company": f"Co{i}", "job_title": "Dev", "duration": "2020",
                                     "bullets": ["Built a thing that matters " * 6] * 5} for i in range(8)])
    pdf = TwoColumnRenderer().render(PROFILE, draft, {})
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        assert doc.page_count > 1
    assert "Co7" in text_of(pdf)


def test_two_column_escapes_html_and_long_words():
    draft = dict(DRAFT, summary="Used <b>& tags</b> " + "x" * 300)
    text = text_of(TwoColumnRenderer().render(PROFILE, draft, {}))
    assert "<b>& tags</b>" in text


def test_two_column_requires_name_and_content():
    with pytest.raises(ValueError):
        TwoColumnRenderer().render({"name": ""}, DRAFT, {})
    with pytest.raises(ValueError):
        TwoColumnRenderer().render(PROFILE, {"headline": "x"}, {})
