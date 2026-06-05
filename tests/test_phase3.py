from placement.chunker import PlacementChunker
from placement.models import (
    ConflictRecord,
    EligibilityProfile,
    HiringDistribution,
    InterviewExperience,
    OverallStats,
    PlacementDataset,
    PlacementTrend,
)

DOC_ID = "test_placement"
SOURCE = "Placement_RAG_Dataset_Enhanced.pdf"
BASE_META = {"source": SOURCE, "access_level": "internal"}


def _make_chunker(dedupe_threshold: float = 0.95) -> PlacementChunker:
    return PlacementChunker(
        doc_id=DOC_ID,
        source=SOURCE,
        base_metadata=BASE_META,
        dedupe_threshold=dedupe_threshold,
    )


def _sample_dataset() -> PlacementDataset:
    return PlacementDataset(
        eligibility_profiles=[
            EligibilityProfile(company="TCS", min_cgpa=6.0, max_backlogs=2, package_lpa=4.1, bond_years=0, key_topics="DSA, SQL", tech_focus="Java"),
            EligibilityProfile(company="Amazon", min_cgpa=6.4, max_backlogs=1, package_lpa=28.6, bond_years=2, key_topics="DSA, C++, LLD", tech_focus="C++"),
            EligibilityProfile(company="Google", min_cgpa=7.0, max_backlogs=0, package_lpa=42.0, bond_years=0, key_topics="DSA, System Design", tech_focus="Python"),
        ],
        hiring_distributions=[
            HiringDistribution(company="TCS", sde=120, analyst=80, officer=30, intern=200, total=430),
            HiringDistribution(company="Amazon", sde=60, analyst=20, officer=10, intern=40, total=130),
        ],
        placement_trends=[
            PlacementTrend(company="TCS", package_2021=3.2, package_2022=3.5, package_2023=3.8, package_2024=4.1, absolute_growth_2021_2024=0.9, trend_label="up"),
            PlacementTrend(company="Amazon", package_2021=22.0, package_2022=24.5, package_2023=26.0, package_2024=28.6, absolute_growth_2021_2024=6.6, trend_label="up"),
        ],
        conflict_records=[
            ConflictRecord(company="Amazon", official_cgpa=6.4, portal_cgpa=7.0, official_package_lpa=28.6, portal_package_lpa=28.6, cgpa_conflict=True, package_conflict=False),
        ],
        overall_stats=[
            OverallStats(company="TCS", avg_package=27.3, max_offers=150, min_offers=20, avg_cgpa_cutoff=6.0, bond_free=True),
        ],
        interview_experiences=[
            InterviewExperience(company="Google", technical_focus="DSA, Python", round_number=1, round_title="Technical Round 1", details="Coding problems on arrays and graphs.", tip="Practice graph algorithms."),
            InterviewExperience(company="Google", technical_focus="System Design", round_number=2, round_title="System Design", details="Design a URL shortener.", tip="Focus on scalability."),
        ],
    )


class TestEligibilityChunking:
    def test_one_row_per_chunk(self) -> None:
        dataset = _sample_dataset()
        chunker = _make_chunker()
        chunks = chunker._chunk_eligibility(dataset.eligibility_profiles)
        assert len(chunks) == 3

    def test_chunk_text_format(self) -> None:
        dataset = _sample_dataset()
        chunker = _make_chunker()
        chunks = chunker._chunk_eligibility(dataset.eligibility_profiles)
        tcs = chunks[0]
        assert "Company: TCS" in tcs.text
        assert "CGPA 6.0" in tcs.text
        assert "package 4.1" in tcs.text
        assert "backlogs 2" in tcs.text

    def test_chunk_metadata(self) -> None:
        dataset = _sample_dataset()
        chunker = _make_chunker()
        chunks = chunker._chunk_eligibility(dataset.eligibility_profiles)
        amazon = chunks[1]
        assert amazon.metadata["section"] == "eligibility"
        assert amazon.metadata["company"] == "Amazon"
        assert amazon.metadata["min_cgpa"] == 6.4
        assert amazon.metadata["max_backlogs"] == 1
        assert amazon.metadata["package_lpa"] == 28.6
        assert amazon.metadata["bond_years"] == 2
        assert amazon.metadata["content_type"] == "structured_row"
        assert amazon.metadata["source_type"] == "official"


class TestHiringChunking:
    def test_one_row_per_chunk(self) -> None:
        dataset = _sample_dataset()
        chunker = _make_chunker()
        chunks = chunker._chunk_hiring(dataset.hiring_distributions)
        assert len(chunks) == 2

    def test_chunk_text_format(self) -> None:
        dataset = _sample_dataset()
        chunker = _make_chunker()
        chunks = chunker._chunk_hiring(dataset.hiring_distributions)
        tcs = chunks[0]
        assert "Company: TCS" in tcs.text
        assert "120 SDE" in tcs.text
        assert "200 Intern" in tcs.text

    def test_chunk_metadata(self) -> None:
        dataset = _sample_dataset()
        chunker = _make_chunker()
        chunks = chunker._chunk_hiring(dataset.hiring_distributions)
        amazon = chunks[1]
        assert amazon.metadata["section"] == "hiring"
        assert amazon.metadata["company"] == "Amazon"
        assert amazon.metadata["sde"] == 60
        assert amazon.metadata["intern"] == 40


class TestTrendChunking:
    def test_one_row_per_chunk(self) -> None:
        dataset = _sample_dataset()
        chunker = _make_chunker()
        chunks = chunker._chunk_trends(dataset.placement_trends)
        assert len(chunks) == 2

    def test_chunk_text_format(self) -> None:
        dataset = _sample_dataset()
        chunker = _make_chunker()
        chunks = chunker._chunk_trends(dataset.placement_trends)
        tcs = chunks[0]
        assert "Company: TCS" in tcs.text
        assert "2021: 3.2" in tcs.text
        assert "2024: 4.1" in tcs.text
        assert "growth" in tcs.text

    def test_chunk_metadata(self) -> None:
        dataset = _sample_dataset()
        chunker = _make_chunker()
        chunks = chunker._chunk_trends(dataset.placement_trends)
        tcs = chunks[0]
        assert tcs.metadata["section"] == "trend"
        assert tcs.metadata["company"] == "TCS"
        assert tcs.metadata["package_2021"] == 3.2
        assert tcs.metadata["absolute_growth"] == 0.9
        assert tcs.metadata["trend_label"] == "up"
        assert tcs.metadata["metric"] == "package_trend"


class TestConflictChunking:
    def test_one_record_per_chunk(self) -> None:
        dataset = _sample_dataset()
        chunker = _make_chunker()
        chunks = chunker._chunk_conflicts(dataset.conflict_records)
        assert len(chunks) == 1

    def test_chunk_text_format(self) -> None:
        dataset = _sample_dataset()
        chunker = _make_chunker()
        chunks = chunker._chunk_conflicts(dataset.conflict_records)
        amazon = chunks[0]
        assert "conflicting records" in amazon.text
        assert "6.4 CGPA" in amazon.text
        assert "7.0 CGPA" in amazon.text

    def test_chunk_metadata(self) -> None:
        dataset = _sample_dataset()
        chunker = _make_chunker()
        chunks = chunker._chunk_conflicts(dataset.conflict_records)
        amazon = chunks[0]
        assert amazon.metadata["section"] == "conflict"
        assert amazon.metadata["company"] == "Amazon"
        assert amazon.metadata["conflict"] is True
        assert amazon.metadata["cgpa_conflict"] is True
        assert amazon.metadata["package_conflict"] is False


class TestStatsChunking:
    def test_single_chunk(self) -> None:
        dataset = _sample_dataset()
        chunker = _make_chunker()
        chunks = chunker._chunk_stats(dataset.overall_stats)
        assert len(chunks) == 1

    def test_chunk_text_format(self) -> None:
        dataset = _sample_dataset()
        chunker = _make_chunker()
        chunks = chunker._chunk_stats(dataset.overall_stats)
        assert "Overall placement statistics" in chunks[0].text
        assert "TCS" in chunks[0].text
        assert "27.3" in chunks[0].text

    def test_chunk_metadata(self) -> None:
        dataset = _sample_dataset()
        chunker = _make_chunker()
        chunks = chunker._chunk_stats(dataset.overall_stats)
        assert chunks[0].metadata["section"] == "statistics"
        assert chunks[0].metadata["content_type"] == "full_table"

    def test_empty_stats(self) -> None:
        chunker = _make_chunker()
        chunks = chunker._chunk_stats([])
        assert chunks == []


class TestInterviewChunking:
    def test_one_round_per_chunk(self) -> None:
        dataset = _sample_dataset()
        chunker = _make_chunker()
        chunks = chunker._chunk_interviews(dataset.interview_experiences)
        # Both Google rounds fit under 300 tokens → merged into 1 chunk
        assert len(chunks) == 1
        assert "Round 1" in chunks[0].text and "Round 2" in chunks[0].text

    def test_chunk_metadata(self) -> None:
        dataset = _sample_dataset()
        chunker = _make_chunker()
        chunks = chunker._chunk_interviews(dataset.interview_experiences)
        for c in chunks:
            assert c.metadata["section"] == "interview"
            assert c.metadata["content_type"] == "interview_round"
            assert "company" in c.metadata
        assert chunks[0].metadata["company"] == "Google"

    def test_no_interviews(self) -> None:
        chunker = _make_chunker()
        chunks = chunker._chunk_interviews([])
        assert chunks == []


class TestInterviewDeduplication:
    def test_no_dedup_needed(self) -> None:
        dataset = _sample_dataset()
        chunker = _make_chunker()
        chunks = chunker._chunk_interviews(dataset.interview_experiences)
        deduped = chunker._deduplicate_interviews(chunks)
        # Single chunk with both Google rounds merged (< 300 tokens)
        assert len(deduped) == 1
        assert deduped[0].metadata["dedupe_count"] == 1

    def test_exact_duplicates(self) -> None:
        chunker = _make_chunker()
        chunks = []
        for i in range(3):
            chunks.extend(chunker._chunk_interviews([
                InterviewExperience(company="Google", technical_focus="DSA", round_number=1, round_title="Tech", details="Arrays and Strings.", tip="Practice"),
            ]))
        deduped = chunker._deduplicate_interviews(chunks)
        assert len(deduped) == 1
        assert deduped[0].metadata["dedupe_count"] == 3
        assert "dedupe_key" in deduped[0].metadata

    def test_unique_chunks_kept(self) -> None:
        chunker = _make_chunker()
        chunks = chunker._chunk_interviews([
            InterviewExperience(company="Google", technical_focus="DSA", round_number=1, round_title="Tech", details="Arrays.", tip="Practice"),
            InterviewExperience(company="Amazon", technical_focus="C++", round_number=1, round_title="Tech", details="Trees.", tip="Study"),
        ])
        deduped = chunker._deduplicate_interviews(chunks)
        assert len(deduped) == 2


class TestNormalizeText:
    def test_lowercase(self) -> None:
        chunker = _make_chunker()
        assert chunker._normalize_text("Hello World") == "hello world"

    def test_collapse_whitespace(self) -> None:
        chunker = _make_chunker()
        result = chunker._normalize_text("Company:   Amazon.   Details: Test.")
        assert "  " not in result

    def test_remove_bullets(self) -> None:
        chunker = _make_chunker()
        result = chunker._normalize_text("• Point one\n- Point two")
        assert "•" not in result
        assert "-" not in result


class TestFullDatasetChunking:
    def test_chunk_dataset(self) -> None:
        dataset = _sample_dataset()
        chunker = _make_chunker()
        chunks = chunker.chunk_dataset(dataset)
        assert len(chunks) > 0

    def test_all_sections_present(self) -> None:
        dataset = _sample_dataset()
        chunker = _make_chunker()
        chunks = chunker.chunk_dataset(dataset)
        sections = {c.metadata["section"] for c in chunks}
        assert "eligibility" in sections
        assert "hiring" in sections
        assert "trend" in sections
        assert "conflict" in sections
        assert "statistics" in sections
        assert "interview" in sections

    def test_each_chunk_has_required_metadata(self) -> None:
        dataset = _sample_dataset()
        chunker = _make_chunker()
        chunks = chunker.chunk_dataset(dataset)
        for c in chunks:
            assert "section" in c.metadata
            assert "content_type" in c.metadata
            assert "source" in c.metadata
            assert c.doc_id == DOC_ID

    def test_empty_dataset(self) -> None:
        dataset = PlacementDataset(
            eligibility_profiles=[],
            interview_experiences=[],
            hiring_distributions=[],
            placement_trends=[],
            conflict_records=[],
            overall_stats=[],
        )
        chunker = _make_chunker()
        chunks = chunker.chunk_dataset(dataset)
        assert len(chunks) == 0


class TestEdgeCases:
    def test_duplicate_company_eligibility(self) -> None:
        chunker = _make_chunker()
        profiles = [
            EligibilityProfile(company="TCS", min_cgpa=6.0, max_backlogs=2, package_lpa=4.1, bond_years=0, key_topics="DSA", tech_focus="Java"),
            EligibilityProfile(company="TCS", min_cgpa=6.0, max_backlogs=2, package_lpa=4.1, bond_years=0, key_topics="DSA", tech_focus="Java"),
        ]
        chunks = chunker._chunk_eligibility(profiles)
        assert len(chunks) == 2
        assert chunks[0].text == chunks[1].text

    def test_large_interview_text_chunking(self) -> None:
        chunker = _make_chunker()
        long_details = " ".join(["detail"] * 1000)
        experiences = [
            InterviewExperience(company="Google", technical_focus="DSA", round_number=1, round_title="Long Round", details=long_details, tip="Long tip " * 100),
        ]
        chunks = chunker._chunk_interviews(experiences)
        assert len(chunks) >= 1
        total_tokens = sum(len(chunker._encoder.encode(c.text)) for c in chunks)
        assert total_tokens > 0
