from pathlib import Path

import pymupdf
import pytest

from app.services.resume_pdf_renderers.entry_grouper import group_entries
from app.services.resume_pdf_renderers.layout_manifest import build_layout_manifest

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


class Page:
    """Tiny resume-drawing helper so tests can build very different designs."""

    def __init__(self, size, font="helv", bold="hebo", italic="heit", left=50):
        self.doc = pymupdf.open()
        self.w, self.h = size
        self.page = self.doc.new_page(width=self.w, height=self.h)
        self.font, self.bold, self.italic = font, bold, italic
        self.left, self.right, self.y = left, self.w - left, 50

    def _t(self, x, s, font, size, color=(0, 0, 0)):
        self.page.insert_text((x, self.y), s, fontname=font, fontsize=size, color=color)

    def w_of(self, s, font, size):
        return pymupdf.get_text_length(s, fontname=font, fontsize=size)

    def center(self, s, size=18, font=None):
        font = font or self.bold
        self._t((self.w - self.w_of(s, font, size)) / 2, s, font, size)
        self.y += size + 6

    def line(self, s, size=9, font=None, x=None):
        self._t(x if x is not None else self.left, s, font or self.font, size)
        self.y += size + 4

    def heading(self, s, size=12, rule=True):
        self.y += 6
        self._t(self.left, s, self.bold, size)
        if rule:
            self.page.draw_line((self.left, self.y + 3), (self.right, self.y + 3))
        self.y += size + 6

    def pair(self, left, right, font=None, size=9, right_font=None):
        font = font or self.font
        self._t(self.left, left, font, size)
        self._t(self.right - self.w_of(right, right_font or self.font, size), right, right_font or self.font, size)
        self.y += size + 4

    def bullet(self, first, cont=None, marker="•", size=9):
        self._t(self.left + 6, f"{marker} {first}", self.font, size)
        self.y += size + 3
        if cont:
            self._t(self.left + 6 + self.w_of(marker + " ", self.font, size), cont, self.font, size)
            self.y += size + 3

    def pdf(self):
        out = self.doc.tobytes()
        self.doc.close()
        return out


def _group(pdf):
    return group_entries(build_layout_manifest(pdf))


def _by_name(g):
    return {s["name"]: s for s in g["sections"]}


def _design_a():
    """A4, Times, left header, caps headings, dash bullets, dates on the company
    row with the title on its own line, plus sections the draft has no field for."""
    p = Page(pymupdf.paper_size("a4"), "tiro", "tibo", "tiit", left=56)
    p.line("Priya Nair", 20, "tibo")
    p.line("priya@example.com | +44 7700 900123 | github.com/priya", 9)
    p.heading("EDUCATION")
    p.pair("University of Leeds, BSc Computer Science", "Leeds | 2016 - 2019", "tibo")
    p.heading("EXPERIENCE")
    p.pair("Acme Corp", "Jan 2022 - Present", "tibo", right_font="tibo")
    p.line("Senior Engineer", 9, "tiit")
    p.bullet("Built payments APIs serving 2M users", "and cut latency by 30 percent.", "-")
    p.bullet("Led a team of four engineers", None, "-")
    p.pair("Beta Inc", "Jun 2019 - Dec 2021", "tibo", right_font="tibo")
    p.line("Software Engineer", 9, "tiit")
    p.bullet("Shipped the mobile checkout", None, "-")
    p.heading("CERTIFICATIONS")
    p.bullet("AWS Solutions Architect Associate", None, "-")
    p.bullet("Certified Kubernetes Administrator", None, "-")
    p.heading("LEADERSHIP")
    p.bullet("Mentor, Women in Tech chapter", None, "-")
    p.heading("SKILLS")
    p.line("Languages: Python, Go, SQL")
    p.line("Tools: Docker, Terraform")
    return p.pdf()


def _design_b():
    """US Letter, Helvetica, centered header, label-style skills, projects."""
    p = Page((612, 792), left=40)
    p.center("Sam Okafor")
    p.center("sam@example.com | linkedin.com/in/samokafor", 9, "helv")
    p.heading("Summary")
    p.line("Backend engineer who ships reliable data pipelines.")
    p.heading("Projects")
    p.pair("Pipeliner | Python, Kafka, Postgres", "2024 - Present", "hebo", right_font="hebo")
    p.bullet("Processed 5M events per day", "with exactly-once delivery.")
    p.pair("Dashy | React, D3", "2023", "hebo", right_font="hebo")
    p.bullet("Built a live metrics dashboard")
    p.heading("Awards")
    p.bullet("Hackathon winner 2023")
    return p.pdf()


def test_design_a_sections_order_and_custom_sections_kept():
    g = _group(_design_a())
    names = [s["name"] for s in g["sections"]]
    assert names == ["Education", "Experience", "Certifications", "Leadership", "Skills"] or \
        names == ["Education", "Experience", "Certifications", "Leadership", "Technical Skills"]
    kinds = {s["name"]: s["kind"] for s in g["sections"]}
    assert kinds["Leadership"] == "custom"            # unknown heading is kept, not merged away
    assert kinds["Certifications"] == "certifications"


def test_design_a_custom_content_is_not_lost():
    s = _by_name(_group(_design_a()))
    certs = [i["text"] for e in s["Certifications"]["content"]["entries"] for i in e["items"]]
    assert certs == ["AWS Solutions Architect Associate", "Certified Kubernetes Administrator"]
    lead = [i["text"] for e in s["Leadership"]["content"]["entries"] for i in e["items"]]
    assert lead == ["Mentor, Women in Tech chapter"]


def test_design_a_experience_one_row_format():
    exp = _by_name(_group(_design_a()))["Experience"]["content"]["companies"]
    assert [c["company"] for c in exp] == ["Acme Corp", "Beta Inc"]
    role = exp[0]["roles"][0]
    assert role["title"] == "Senior Engineer"
    assert role["duration"] == "Jan 2022 - Present"
    # wrapped bullet text is rejoined and attached to the right job
    assert role["bullets"][0]["text"] == "Built payments APIs serving 2M users and cut latency by 30 percent."
    assert len(role["bullets"]) == 2
    assert [b["text"] for b in exp[1]["roles"][0]["bullets"]] == ["Shipped the mobile checkout"]


def test_design_a_header_left_aligned_and_education():
    g = _group(_design_a())
    assert g["header"]["name"]["text"] == "Priya Nair"
    assert {c["kind"] for c in g["header"]["contacts"]} == {"email", "phone", "url"}
    edu = _by_name(g)["Education"]["content"]["entries"][0]
    assert edu["institution"].startswith("University of Leeds")
    assert edu["location"] == "Leeds" and "2016" in edu["dates"]


def test_design_b_projects_and_custom():
    s = _by_name(_group(_design_b()))
    assert s["Summary"]["content"]["paragraph"]["text"].startswith("Backend engineer")
    projects = s["Projects"]["content"]["projects"]
    assert [p["name"] for p in projects] == ["Pipeliner", "Dashy"]
    assert projects[0]["technologies"] == "Python, Kafka, Postgres"
    assert projects[0]["dates"] == "2024 - Present"
    assert projects[0]["bullets"][0]["text"] == "Processed 5M events per day with exactly-once delivery."
    assert s["Awards"]["kind"] == "custom"


def test_every_line_belongs_to_exactly_one_place():
    """No text may be dropped or duplicated by grouping."""
    for pdf in (_design_a(), _design_b()):
        m = build_layout_manifest(pdf)
        g = group_entries(m)
        seen = list(g["header"]["line_ids"])
        for s in g["sections"]:
            seen.append(s["heading"]["line_id"])
            seen += s["line_ids"]
        all_lines = {l["id"] for p in m["pages"] for l in p["lines"]}
        # separator-only fragments such as a lone "|" are intentionally dropped
        texts = {l["id"]: l["text"].strip() for p in m["pages"] for l in p["lines"]}
        missing = {i for i in all_lines - set(seen) if texts[i] not in {"|", "•", "-", ","}}
        assert not missing, missing
        assert len(seen) == len(set(seen))


@pytest.mark.skipif(not (FIXTURES / "Vidip_Resume.pdf").exists(), reason="sample resume not present")
def test_real_resume_tree():
    g = _group((FIXTURES / "Vidip_Resume.pdf").read_bytes())
    s = _by_name(g)
    exp = s["Experience"]["content"]["companies"]
    assert [(c["company"], c["location"], len(c["roles"])) for c in exp] == [
        ("Aforro", "Bengaluru", 2), ("TechKareer", "Bengaluru", 1), ("Tata Consultancy Services", "Noida", 1)]
    assert [len(r["bullets"]) for r in exp[0]["roles"]] == [3, 3]
    assert exp[0]["roles"][1]["title"] == "React Native Intern"
    assert exp[0]["roles"][1]["duration"] == "Apr 2026 – Jul 2026"
    assert [p["name"] for p in s["Projects"]["content"]["projects"]] == ["StackSearch", "AegisSRE"]
    assert len(s["Technical Skills"]["content"]["groups"]) == 7
    assert len(s["Achievements"]["content"]["entries"][0]["items"]) == 3
    assert g["warnings"] == []
