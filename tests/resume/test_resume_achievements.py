from unittest.mock import patch

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from app.models.profile import ProfileCreate
from app.services import outreach_agent as oa
from app.services import resume_service
from app.services.resume_service import extract_and_analyze_resume, extract_bullet_entries
from tests.conftest import create_sample_pdf

# The exact lines from the bug report, wrapped where the PDF wraps them.
REPORTED_SECTION = """\
• Multi-Hackathon Winner: Winner/Runner-Up titles across 8 national and international hackathons, including 1st
Runner-Up at Build on Aptos (Consumer Apps Track) and 1st Runner-Up at Hack This Fall.
• Global Top-10 Finalist – TON Open Hack (Dec 2025): Built LocalPay (Next.js), an instant Web3 Telegram checkout for
local merchants with auto-stablecoin settlement, simulated TON payments, and AI-driven merchant insights.
• Achieved Global Rank #68 in GirlScript Summer of Code 2023 with 29 merged open-source pull requests.
"""
REPORTED_EXPECTED = [
    "Multi-Hackathon Winner: Winner/Runner-Up titles across 8 national and international hackathons, including 1st "
    "Runner-Up at Build on Aptos (Consumer Apps Track) and 1st Runner-Up at Hack This Fall.",
    "Global Top-10 Finalist – TON Open Hack (Dec 2025): Built LocalPay (Next.js), an instant Web3 Telegram checkout for "
    "local merchants with auto-stablecoin settlement, simulated TON payments, and AI-driven merchant insights.",
    "Achieved Global Rank #68 in GirlScript Summer of Code 2023 with 29 merged open-source pull requests.",
]


# --- the five cases asked for ---------------------------------------------------------------------------------

def test_a_wrapped_achievement_is_one_entry():
    assert extract_bullet_entries("• Won first place at the national\nhackathon in 2024.") == [
        "Won first place at the national hackathon in 2024."
    ]


def test_independent_bullets_stay_separate():
    assert extract_bullet_entries("• Won a hackathon.\n• Ranked #68 in GSSoC 2023.\n• Merged 29 pull requests.") == [
        "Won a hackathon.", "Ranked #68 in GSSoC 2023.", "Merged 29 pull requests.",
    ]


def test_every_listed_marker_starts_a_new_achievement():
    assert extract_bullet_entries("• one\n● two\n▪ three\n- four\n* five") == ["one", "two", "three", "four", "five"]


def test_achievements_without_bullet_markers_are_kept_and_a_wrapped_line_continues_the_first():
    assert extract_bullet_entries("Won a hackathon in 2024") == ["Won a hackathon in 2024"]
    assert extract_bullet_entries("Won first place at the\nnational hackathon in 2024.") == [
        "Won first place at the national hackathon in 2024."
    ]


def test_the_first_line_is_kept_even_without_a_marker_when_bullets_follow():
    assert extract_bullet_entries("Top finisher at a hackathon\n• Ranked #68 in GSSoC 2023.") == [
        "Top finisher at a hackathon", "Ranked #68 in GSSoC 2023.",
    ]


def test_empty_lines_between_wrapped_lines_and_between_bullets_are_ignored():
    text = "• Won first place at the national\n\n   \nhackathon in 2024.\n\n• Ranked #68 in GSSoC 2023."
    assert extract_bullet_entries(text) == ["Won first place at the national hackathon in 2024.", "Ranked #68 in GSSoC 2023."]


def test_wording_punctuation_numbers_and_dates_are_preserved():
    result = extract_bullet_entries(REPORTED_SECTION)
    assert result == REPORTED_EXPECTED
    for kept in ("Multi-Hackathon", "Winner/Runner-Up", "1st", "(Consumer Apps Track)", "Top-10", "–", "(Dec 2025)",
                 "(Next.js)", "auto-stablecoin", "#68", "29 merged"):
        assert any(kept in entry for entry in result), kept


def test_the_reported_five_fragments_become_exactly_three_achievements():
    result = extract_bullet_entries(REPORTED_SECTION)
    assert len(result) == 3
    assert not any(entry.startswith(("Runner-Up at Build", "local merchants")) for entry in result)   # no orphan fragments


# --- the marker rules and edge cases -----------------------------------------------------------------------------

def test_a_wrapped_line_starting_with_a_dash_or_star_but_no_space_is_not_a_new_bullet():
    assert extract_bullet_entries("• Cut latency by\n-5% across services.") == ["Cut latency by -5% across services."]
    assert extract_bullet_entries("• Wrote helpers using\n*args patterns.") == ["Wrote helpers using *args patterns."]


def test_a_filled_bullet_glyph_without_a_space_is_still_a_marker():
    assert extract_bullet_entries("•Won a hackathon\n●Ranked #68") == ["Won a hackathon", "Ranked #68"]


def test_whitespace_is_normalised_when_joining():
    assert extract_bullet_entries("•   Won   first\t place  \n    at the   hackathon.   ") == ["Won first place at the hackathon."]


def test_a_marker_on_its_own_line_takes_the_following_line_as_its_text():
    assert extract_bullet_entries("•\nWon a hackathon.\n•\n• Ranked #68") == ["Won a hackathon.", "Ranked #68"]
    assert extract_bullet_entries("-\nWon a hackathon.\n*\nRanked #68") == ["Won a hackathon.", "Ranked #68"]


@pytest.mark.parametrize("empty", ["", "   ", "\n\n", "•", "• \n  \n-  "])
def test_nothing_in_gives_nothing_out(empty):
    assert extract_bullet_entries(empty) == []


def test_a_parsing_error_is_not_swallowed():
    with pytest.raises(AttributeError):
        extract_bullet_entries(None)


def test_the_input_text_is_not_modified_and_the_result_is_stable():
    first = extract_bullet_entries(REPORTED_SECTION)
    assert extract_bullet_entries(REPORTED_SECTION) == first and REPORTED_SECTION.count("•") == 3


# --- through the real flow: PDF -> ExtractedProfile -> persisted profile -> FAISS documents ------------------------

PDF_TEXT = """\
Alex Rivera
alex@example.com

ACHIEVEMENTS
- Multi-Hackathon Winner: Winner/Runner-Up titles across 8 national and
international hackathons, including 1st Runner-Up at Build on Aptos
(Consumer Apps Track) and 1st Runner-Up at Hack This Fall.
- Global Top-10 Finalist - TON Open Hack (Dec 2025): Built LocalPay (Next.js), an
instant Web3 Telegram checkout for local merchants with auto-stablecoin
settlement, simulated TON payments, and AI-driven merchant insights.
- Achieved Global Rank #68 in GirlScript Summer of Code 2023 with 29 merged
open-source pull requests.

EDUCATION
B.Tech in Computer Science | IIT Madras
"""
PDF_EXPECTED = [
    "Multi-Hackathon Winner: Winner/Runner-Up titles across 8 national and international hackathons, including 1st "
    "Runner-Up at Build on Aptos (Consumer Apps Track) and 1st Runner-Up at Hack This Fall.",
    "Global Top-10 Finalist - TON Open Hack (Dec 2025): Built LocalPay (Next.js), an instant Web3 Telegram checkout for "
    "local merchants with auto-stablecoin settlement, simulated TON payments, and AI-driven merchant insights.",
    "Achieved Global Rank #68 in GirlScript Summer of Code 2023 with 29 merged open-source pull requests.",
]


def test_an_uploaded_pdf_yields_three_complete_achievements_in_the_extracted_profile():
    profile = extract_and_analyze_resume(create_sample_pdf(PDF_TEXT))
    assert profile.achievements == PDF_EXPECTED
    assert profile.education and "IIT Madras" in profile.education[0]               # the next section is untouched


def test_the_achievements_flow_through_persistence_into_the_search_index(tmp_path, monkeypatch):
    monkeypatch.setattr(oa, "INDEX_DIR", tmp_path)
    monkeypatch.setattr(oa, "get_embeddings", lambda: DeterministicFakeEmbedding(size=16))
    oa._embedding_dimension.cache_clear()
    oa.clear_index_cache()

    extracted = extract_and_analyze_resume(create_sample_pdf(PDF_TEXT))
    confirmed = ProfileCreate(**extracted.model_dump(exclude={"raw_resume_text"}))
    with patch("app.db.models.insert", side_effect=lambda table, data: dict(data)):
        saved = resume_service.confirm_and_persist_profile(confirmed)

    assert saved.achievements == PDF_EXPECTED
    indexed = [d.page_content for d in oa._VECTOR_STORE_CACHE[saved.id].docstore._dict.values()
               if d.metadata.get("source_section") == "achievements"]
    assert sorted(indexed) == sorted(f"Quantifiable Achievement: {text}" for text in PDF_EXPECTED)
    oa.clear_index_cache()
    oa._embedding_dimension.cache_clear()
