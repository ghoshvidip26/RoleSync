from langchain_core.embeddings import DeterministicFakeEmbedding

from app.models.profile import Profile
from app.services import outreach_agent as oa

LEGACY_JOB = ("Junior Mobile Developer | Aforro, Bengaluru/nJul 2026 – Present/n"
              "• Built end-to-end Rider app workflows in React Native over REST APIs: pickup, delivery,\n"
              "returns and cancellations, with live rider location tracking./n"
              "• Implemented background geofencing for proximity-based state changes.")
PROJECT = ("StackSearch | GitHub | Node.js, TypeScript, FAISS\nJan 2026 – Present\n"
           "• Built a semantic documentation search engine using RAG: answers are ranked\nby relevance.\n"
           "• Reached >80% answer relevance across 50+ test queries.")


def profile(**kw):
    return Profile(id="p", name="N", email="n@example.com", **kw)


def docs_for(section, **kw):
    return [d for d in oa.build_documents_from_profile(profile(**kw)) if d.metadata["source_section"] == section]


def test_an_experience_entry_stays_whole_with_title_company_dates_and_every_bullet():
    (doc,) = docs_for("experience", experience=[LEGACY_JOB])
    assert doc.page_content.startswith("Work Experience: Junior Mobile Developer | Aforro, Bengaluru\nJul 2026 – Present\n")
    assert "/n" not in doc.page_content
    assert "returns and cancellations, with live rider location tracking." in doc.page_content   # wrapped bullet rejoined
    assert "background geofencing" in doc.page_content
    assert (doc.metadata["job_title"], doc.metadata["company"], doc.metadata["dates"]) == (
        "Junior Mobile Developer", "Aforro", "Jul 2026 – Present")


def test_project_metadata_has_the_real_name_and_technologies_never_the_generic_word_project():
    (doc,) = docs_for("projects", projects=[PROJECT])
    assert doc.metadata["project_name"] == "StackSearch"
    assert doc.metadata["technologies"] == "Node.js, TypeScript, FAISS"          # the "GitHub" link label is not a technology
    assert ">80% answer relevance" in doc.page_content and "Project: StackSearch" in doc.page_content


def test_metadata_is_left_out_when_the_text_does_not_provide_it():
    unnamed = "Built a small tool for fun with Python and some other things that has a very long unnamed description."
    (doc,) = docs_for("projects", projects=[unnamed])
    assert "project_name" not in doc.metadata
    (doc,) = docs_for("experience", experience=["Worked on various things for a while.\n• Helped the team."])
    assert "job_title" not in doc.metadata and "company" not in doc.metadata


def test_name_before_colon_and_title_at_company_formats_are_understood():
    (p,) = docs_for("projects", projects=["StreamLite: Open-source real-time event engine in Go."])
    assert p.metadata["project_name"] == "StreamLite"
    (e,) = docs_for("experience", experience=["Senior Backend Engineer at FinTech Global (2022-Present):\n- Built a pipeline."])
    assert (e.metadata["job_title"], e.metadata["company"], e.metadata["dates"]) == ("Senior Backend Engineer", "FinTech Global", "2022-Present")


def test_a_very_long_entry_splits_between_bullets_and_every_part_repeats_the_header():
    bullets = "\n".join(f"• Delivered payments platform feature {i} with measurable impact on reliability and latency." for i in range(30))
    docs = docs_for("experience", experience=["Platform Engineer | BigCo, Delhi\nJun 2019 – Dec 2021\n" + bullets])
    assert len(docs) > 1
    for d in docs:
        assert d.page_content.startswith("Work Experience: Platform Engineer | BigCo, Delhi\nJun 2019 – Dec 2021\n• ")
        assert d.page_content.rstrip().endswith(".")            # never cut mid-sentence
        assert d.metadata["company"] == "BigCo" and "part" in d.metadata


def test_the_index_keeps_entries_whole_instead_of_resplitting_them_at_500_characters(tmp_path, monkeypatch):
    monkeypatch.setattr(oa, "INDEX_DIR", tmp_path)
    monkeypatch.setattr(oa, "get_embeddings", lambda: DeterministicFakeEmbedding(size=32))
    oa.clear_index_cache()
    long_job = "Backend Engineer | PayLoop, Pune\nJan 2022 – Present\n" + "\n".join(
        f"• Built service number {i} that handles high volume traffic with retries and monitoring in place." for i in range(8))
    assert len(long_job) > 500
    p = profile(experience=[long_job], projects=[PROJECT])
    oa.index_resume("p1", oa.build_documents_from_profile(p, raw_text=None))
    chunks = [d for d in oa._VECTOR_STORE_CACHE["p1"].docstore._dict.values() if d.metadata["source_section"] in ("experience", "projects")]
    assert len(chunks) == 2                                            # one chunk per entry, not 4+ fragments
    assert all(c.page_content.startswith(("Work Experience:", "Project:")) for c in chunks)
    oa.clear_index_cache()


def test_unstructured_resume_text_is_chunked_on_line_boundaries():
    raw = "\n".join(f"Line number {i} describes one complete accomplishment in this resume." for i in range(40))
    chunks = [d for d in oa.build_documents_from_profile(profile(), raw_text=raw) if d.metadata["source_section"] == "raw_resume"]
    assert len(chunks) > 1 and all(c.page_content.startswith("Line number") for c in chunks)


def test_the_structured_profile_survives_saving_so_the_review_can_read_it(monkeypatch):
    """Regression: confirm_and_persist_profile filled `resume` with the summary, which made the
    database layer drop experience, projects, skills and achievements."""
    import app.db.models as db
    from app.services import resume_service

    stored = {}

    class Table:
        def insert(self, row):
            stored["row"] = row
            return self

        def select(self, *_):
            return self

        def eq(self, *_):
            return self

        def limit(self, *_):
            return self

        def execute(self):
            return type("R", (), {"data": [stored["row"]]})()

    monkeypatch.setattr(db, "get_client", lambda: type("C", (), {"table": lambda self, n: Table()})())
    monkeypatch.setattr(oa, "index_resume", lambda *a, **k: None)
    saved = resume_service.confirm_and_persist_profile(
        profile(summary="Backend engineer.", skills=["Python"], experience=[LEGACY_JOB], projects=[PROJECT])
    )
    assert saved.resume is None                                  # internal JSON is not shown to clients
    fetched = db.fetch(db.PROFILES, saved.id)                    # what the review endpoint reads
    assert fetched["experience"] == [LEGACY_JOB] and fetched["projects"] == [PROJECT]
    assert fetched["skills"] == ["Python"] and fetched["summary"] == "Backend engineer."


def test_dots_used_as_bullets_and_dashes_still_parse():
    (doc,) = docs_for("experience", experience=["Dev | Acme\nJan 2022 · Present\n· Built a thing that processes events for the\npayments ledger across services."])
    assert doc.metadata["dates"] == "Jan 2022 · Present"
    assert "• Built a thing that processes events for the payments ledger across services." in doc.page_content
