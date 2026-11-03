from pathlib import Path

import pytest

from app.services.resume_pdf_renderers.content_mapper import map_content
from app.services.resume_pdf_renderers.entry_grouper import group_entries
from app.services.resume_pdf_renderers.layout_manifest import build_layout_manifest
from test_entry_grouper import Page, _design_a, _design_b

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
LETTER = (612, 792)


def grouped(pdf):
    return group_entries(build_layout_manifest(pdf))


def section(plan, name):
    return next(s for s in plan["sections"] if s["name"] == name)


def report(plan, bucket):
    return [f"{r['section']}:{r['what']}" for r in plan["report"][bucket]]


def identity_draft(g):
    """A draft that says exactly what the original resume says."""
    d = {"headline": "", "summary": "", "skills": [], "experience": [], "projects": [],
         "achievements": [], "education": []}
    for s in g["sections"]:
        c = s["content"]
        if s["kind"] == "summary":
            d["summary"] = c["paragraph"]["text"]
        elif s["kind"] == "skills":
            d["skills"] = []                       # nothing to highlight
        elif s["kind"] == "experience":
            for co in c["companies"]:
                for r in co["roles"]:
                    d["experience"].append({"company": co["company"], "job_title": r["title"],
                                            "duration": r["duration"], "location": co["location"],
                                            "bullets": [b["text"] for b in r["bullets"]]})
        elif s["kind"] == "projects":
            d["projects"] = [{"name": p["name"], "technologies": p["technologies"].split(", "),
                              "bullets": [b["text"] for b in p["bullets"]]} for p in c["projects"]]
        elif s["kind"] == "achievements":
            d["achievements"] = [i["text"] for e in c["entries"] for i in e["items"]]
        elif s["kind"] == "education":
            d["education"] = [{"details": f"{e['institution']} {e['location']} {e['dates']}"} for e in c["entries"]]
    return d


# ------------------------------------------------------------- identity

@pytest.mark.parametrize("make", [_design_a, _design_b])
def test_draft_equal_to_original_changes_nothing(make):
    g = grouped(make())
    plan = map_content(g, identity_draft(g))
    for bucket in ("changed", "added", "omitted", "unplaced"):
        assert report(plan, bucket) == [], bucket


@pytest.mark.skipif(not (FIXTURES / "Vidip_Resume.pdf").exists(), reason="sample resume not present")
def test_real_resume_identity():
    g = grouped((FIXTURES / "Vidip_Resume.pdf").read_bytes())
    plan = map_content(g, identity_draft(g))
    assert report(plan, "changed") == [] and report(plan, "omitted") == [] and report(plan, "added") == []


# ---------------------------------------------------------------- summary

def test_summary_replaced_and_kept():
    g = grouped(_design_b())
    plan = map_content(g, {"summary": "A sharper summary for this job."})
    slot = section(plan, "Summary")["content"]["paragraph"]
    assert slot["text"] == "A sharper summary for this job." and slot["changed"]
    assert slot["original"].startswith("Backend engineer") and slot["template"]
    kept = map_content(g, {"summary": ""})
    assert not section(kept, "Summary")["content"]["paragraph"]["changed"]


def test_draft_section_missing_from_original_is_unplaced_not_added():
    plan = map_content(grouped(_design_a()), {"summary": "Summary text the original has no place for."})
    assert [s["name"] for s in plan["sections"]] == ["Education", "Experience", "Certifications", "Leadership", "Technical Skills"]
    assert any(r["section"] == "Summary" for r in plan["report"]["unplaced"])


# ----------------------------------------------------------------- skills

def test_skills_reordered_not_removed_and_unknown_reported():
    plan = map_content(grouped(_design_a()), {"skills": ["Terraform", "Go", "Rust"]})
    groups = {g["label"]: g for g in section(plan, "Technical Skills")["content"]["groups"]}
    assert groups["Languages"]["values"]["text"] == "Go, Python, SQL"       # Go first, nothing dropped
    assert groups["Tools"]["values"]["text"] == "Terraform, Docker"
    assert groups["Tools"]["highlighted"] == ["Terraform"]
    assert "Technical Skills:Rust" in report(plan, "unplaced")                        # not in any original group


def test_skill_match_ignores_case_and_punctuation():
    plan = map_content(grouped(_design_a()), {"skills": ["sql", "GO"]})
    groups = {g["label"]: g for g in section(plan, "Technical Skills")["content"]["groups"]}
    assert groups["Languages"]["values"]["text"] == "SQL, Go, Python"


# ------------------------------------------------------------- experience

def _exp(company, title, bullets, **kw):
    return dict({"company": company, "job_title": title, "duration": "", "location": "", "bullets": bullets}, **kw)


def test_experience_matched_by_name_with_original_order_and_new_bullets():
    draft = {"experience": [
        _exp("Beta Inc.", "Software Engineer", ["Shipped checkout v2", "Cut crashes by half"]),
        _exp("ACME corp", "Senior Engineer", ["New first bullet"]),
    ]}
    plan = map_content(grouped(_design_a()), draft)
    cos = section(plan, "Experience")["content"]["companies"]
    assert [c["company"]["text"] for c in cos] == ["Acme Corp", "Beta Inc"]       # original order and spelling
    assert [b["text"] for b in cos[0]["roles"][0]["bullets"]] == ["New first bullet"]
    assert [b["text"] for b in cos[1]["roles"][0]["bullets"]] == ["Shipped checkout v2", "Cut crashes by half"]
    # dates/location the draft did not supply stay as in the original
    assert cos[0]["roles"][0]["duration"]["text"] == "Jan 2022 - Present"
    assert cos[0]["roles"][0]["bullets"][0]["template"]       # styled after an original bullet


def test_omitted_original_entries_are_reported_and_not_restored():
    draft = {"experience": [_exp("Acme Corp", "Senior Engineer", ["Only this one"])]}
    plan = map_content(grouped(_design_a()), draft)
    cos = section(plan, "Experience")["content"]["companies"]
    assert [c["company"]["text"] for c in cos] == ["Acme Corp"]
    assert any("Beta Inc" in r for r in report(plan, "omitted"))


def test_draft_entry_without_original_is_added_and_flagged():
    draft = {"experience": [
        _exp("Acme Corp", "Senior Engineer", ["a"]),
        _exp("Beta Inc", "Software Engineer", ["b"]),
        _exp("Gamma LLC", "Intern", ["c"], duration="2018"),
    ]}
    plan = map_content(grouped(_design_a()), draft)
    cos = section(plan, "Experience")["content"]["companies"]
    assert cos[-1]["status"] == "added" and cos[-1]["company"]["text"] == "Gamma LLC"
    assert cos[-1]["roles"][0]["bullets"][0]["provenance"] == "not_in_original"
    assert any("Gamma LLC" in r for r in report(plan, "added"))


def test_second_role_at_known_company_joins_that_company():
    draft = {"experience": [
        _exp("Acme Corp", "Senior Engineer", ["a"]),
        _exp("Acme Corp", "Tech Lead", ["extra role"]),
        _exp("Beta Inc", "Software Engineer", ["b"]),
    ]}
    plan = map_content(grouped(_design_a()), draft)
    acme = section(plan, "Experience")["content"]["companies"][0]
    assert [r["status"] for r in acme["roles"]] == ["mapped", "added"]


def test_empty_draft_bullets_keep_original():
    draft = {"experience": [_exp("Acme Corp", "Senior Engineer", []), _exp("Beta Inc", "Software Engineer", [])]}
    plan = map_content(grouped(_design_a()), draft)
    texts = [b["text"] for b in section(plan, "Experience")["content"]["companies"][0]["roles"][0]["bullets"]]
    assert texts[0].startswith("Built payments APIs") and len(texts) == 2


# --------------------------------------------------------------- projects

def test_projects_follow_draft_order_and_report_omitted_added_and_description():
    draft = {"projects": [
        {"name": "Dashy", "technologies": ["D3", "React"], "description": "A dashboard", "bullets": ["Live charts"]},
        {"name": "Brand New", "technologies": ["Rust"], "bullets": ["x"]},
    ]}
    plan = map_content(grouped(_design_b()), draft)
    projects = section(plan, "Projects")["content"]["projects"]
    assert [p["name"]["text"] for p in projects] == ["Dashy", "Brand New"]
    assert projects[0]["technologies"]["text"] == "React, D3"            # same set -> original spelling kept
    assert projects[0]["technologies"]["changed"] is False
    assert projects[1]["status"] == "added"
    assert any("Pipeliner" in r for r in report(plan, "omitted"))
    assert any("description of Dashy" in r for r in report(plan, "unplaced"))


# ------------------------------------------------- flat bullet sections

def _with_achievements():
    p = Page(LETTER, left=40)
    p.center("Alex Doe")
    p.center("alex@example.com", 9, "helv")
    p.heading("Achievements")
    p.bullet("Won the 2023 hackathon")
    p.bullet("Ranked top ten in a contest")
    p.heading("Awards")
    p.bullet("Employee of the year")
    return p.pdf()


def test_achievements_replaced_with_draft_text():
    plan = map_content(grouped(_with_achievements()), {"achievements": ["Won the 2023 hackathon", "Brand new honor"]})
    sec = section(plan, "Achievements")["content"]
    assert [b["text"] for b in sec["bullets"]] == ["Won the 2023 hackathon", "Brand new honor"]
    assert [b["changed"] for b in sec["bullets"]] == [False, True]
    assert sec["bullets"][1]["template"]


def test_custom_section_kept_unless_draft_supplies_it():
    g = grouped(_with_achievements())
    kept = map_content(g, {"achievements": ["x"]})
    assert section(kept, "Awards")["content"]["status"] == "unchanged"
    assert any("Awards" in r for r in report(kept, "kept_original"))
    supplied = map_content(g, {"custom_sections": {"Awards": ["Employee of the year", "Mentor award"]}})
    assert [b["text"] for b in section(supplied, "Awards")["content"]["bullets"]] == [
        "Employee of the year", "Mentor award"]


# ---------------------------------------------------- facts and header

def test_education_kept_and_disagreement_reported():
    plan = map_content(grouped(_design_a()), {"education": [{"details": "MIT PhD Quantum Basket Weaving 2030"}]})
    edu = section(plan, "Education")["content"]
    assert edu["status"] == "unchanged" and edu["entries"][0]["institution"].startswith("University of Leeds")
    assert any("MIT" in r for r in report(plan, "warnings"))


def test_headline_needs_a_tagline_row():
    assert any(r["section"] == "Header" for r in map_content(grouped(_design_b()), {"headline": "Staff Engineer"})["report"]["unplaced"])
    p = Page(LETTER, left=40)
    p.center("Alex Doe")
    p.center("Backend Engineer", 10, "helv")
    p.center("alex@example.com | github.com/alex", 9, "helv")
    p.heading("Summary")
    p.line("Some summary text.")
    plan = map_content(grouped(p.pdf()), {"headline": "Platform Engineer"})
    assert plan["header"]["headline"]["text"] == "Platform Engineer"
    assert plan["header"]["headline"]["original"] == "Backend Engineer"


def test_claims_requiring_confirmation_are_passed_through():
    plan = map_content(grouped(_design_b()), {"claims_requiring_confirmation": ["Kubernetes: add only if used"]})
    assert any("Kubernetes" in r for r in report(plan, "warnings"))


# -------------------------------------------------------------- provenance

def test_bullet_provenance_and_source_mismatch_warning():
    g = grouped(_design_a())
    reworded = _exp("Acme Corp", "Senior Engineer",
                    ["Built payments APIs that serve two million users", "Totally unrelated quantum widget"])
    plan = map_content(g, {"experience": [reworded]})
    prov = [b["provenance"] for b in section(plan, "Experience")["content"]["companies"][0]["roles"][0]["bullets"]]
    assert prov == ["reworded", "not_in_original"] or prov == ["original", "not_in_original"]
    mismatch = map_content(g, {"experience": [_exp("Acme Corp", "Senior Engineer", ["Alpha beta gamma", "Delta epsilon zeta"])]})
    assert any(r["what"] == "source_mismatch" for r in mismatch["report"]["warnings"])


def test_rejects_non_dict_draft():
    with pytest.raises(ValueError):
        map_content(grouped(_design_b()), None)
