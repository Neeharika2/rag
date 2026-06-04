import json
from typing import Any, Dict

from placement.extractor import (
    _normalize_company,
    _parse_float,
    _parse_int,
    _parse_table_rows,
    _split_sections,
    extract_all,
    extract_conflicts,
    extract_eligibility_profiles,
    extract_hiring_distribution,
    extract_overall_stats,
    extract_trends,
)
from placement.models import PlacementDataset

SAMPLE_ELIGIBILITY_TABLE = """
| Company | Min CGPA | Max Backlogs | Package (LPA) | Bond (years) | Key Topics | Tech Focus |
|---------|----------|-------------|--------------|-------------|-----------|-----------|
| TCS | 6.0 | 2 | 4.1 | 0 | DSA, SQL | Java |
| Amazon | 6.4 | 1 | 28.6 | 2 | DSA, C++, LLD | C++ |
| Google | 7.0 | 0 | 42.0 | 0 | DSA, System Design | Python |
| Infosys | 6.5 | 1 | 9.5 | 0 | DSA, Python | Python |
| Microsoft | 7.0 | 0 | 35.0 | 0 | DSA, OS, DBMS | C++ |
"""

SAMPLE_HIRING_TABLE = """
| Company | SDE | Analyst | Officer | Intern | Total |
|---------|-----|---------|---------|--------|-------|
| TCS | 120 | 80 | 30 | 200 | 430 |
| Amazon | 60 | 20 | 10 | 40 | 130 |
| Google | 40 | 15 | 5 | 25 | 85 |
| Infosys | 90 | 45 | 20 | 150 | 305 |
| Microsoft | 50 | 25 | 8 | 35 | 118 |
"""

SAMPLE_TREND_TABLE = """
| Company | 2021 | 2022 | 2023 | 2024 |
|---------|------|------|------|------|
| TCS | 3.2 | 3.5 | 3.8 | 4.1 |
| Amazon | 22.0 | 24.5 | 26.0 | 28.6 |
| Google | 35.0 | 38.0 | 40.0 | 42.0 |
| Infosys | 7.5 | 8.0 | 8.5 | 9.5 |
| Microsoft | 28.0 | 30.0 | 32.0 | 35.0 |
"""

SAMPLE_CONFLICT_TABLE = """
| Company | Official CGPA | Portal CGPA | Official Package | Portal Package |
|---------|--------------|------------|-----------------|---------------|
| Amazon | 6.4 | 7.0 | 28.6 | 28.6 |
| TCS | 6.0 | 6.0 | 4.1 | 4.1 |
"""

SAMPLE_STATS_TABLE = """
| Company | Avg Package | Max Offers | Min Offers | Avg CGPA Cutoff | Bond Free |
|---------|------------|-----------|-----------|----------------|-----------|
| TCS | 27.3 | 150 | 20 | 6.0 | Yes |
| Amazon | 29.5 | 80 | 10 | 6.4 | No |
| Google | 43.0 | 50 | 5 | 7.0 | Yes |
| Infosys | 42.9 | 120 | 15 | 6.5 | Yes |
"""

SAMPLE_FULL_TEXT = """
# Section 1: Eligibility

| Company | Min CGPA | Max Backlogs | Package (LPA) | Bond (years) | Key Topics | Tech Focus |
|---------|----------|-------------|--------------|-------------|-----------|-----------|
| TCS | 6.0 | 2 | 4.1 | 0 | DSA, SQL | Java |
| Amazon | 6.4 | 1 | 28.6 | 2 | DSA, C++, LLD | C++ |
| Google | 7.0 | 0 | 42.0 | 0 | DSA, System Design | Python |

# Section 2: Interview Experiences

## TCS
Round 1: Technical
DSA and SQL questions.
Tip: Practice SQL queries.

## Amazon
Round 1: Technical
C++ and DSA focused.
Tip: Focus on problem-solving.

# Section 3: Hiring Distribution

| Company | SDE | Analyst | Officer | Intern | Total |
|---------|-----|---------|---------|--------|-------|
| TCS | 120 | 80 | 30 | 200 | 430 |
| Amazon | 60 | 20 | 10 | 40 | 130 |

# Section 5: Temporal Trends

| Company | 2021 | 2022 | 2023 | 2024 |
|---------|------|------|------|------|
| TCS | 3.2 | 3.5 | 3.8 | 4.1 |
| Amazon | 22.0 | 24.5 | 26.0 | 28.6 |

# Section 6: Conflicting Data

| Company | Official CGPA | Portal CGPA | Official Package | Portal Package |
|---------|--------------|------------|-----------------|---------------|
| Amazon | 6.4 | 7.0 | 28.6 | 28.6 |

# Section 7: Overall Statistics

| Company | Avg Package | Max Offers | Min Offers |
|---------|------------|-----------|-----------|
| TCS | 27.3 | 150 | 20 |
| Amazon | 29.5 | 80 | 10 |
"""


class TestCompanyNormalization:
    def test_basic_name(self) -> None:
        assert _normalize_company("TCS") == "TCS"

    def test_samsung_normalization(self) -> None:
        assert _normalize_company("Samsung R&D;") == "Samsung R&D"
        assert _normalize_company("Samsung R&D") == "Samsung R&D"

    def test_trailing_semicolon(self) -> None:
        assert _normalize_company("Intel;") == "Intel"


class TestParseHelpers:
    def test_parse_float_normal(self) -> None:
        assert _parse_float("6.4") == 6.4

    def test_parse_float_with_lpa(self) -> None:
        assert _parse_float("28.6 LPA") == 28.6

    def test_parse_float_empty(self) -> None:
        assert _parse_float("-") == 0.0
        assert _parse_float("") == 0.0
        assert _parse_float("N/A") == 0.0

    def test_parse_int_normal(self) -> None:
        assert _parse_int("2") == 2

    def test_parse_int_float_string(self) -> None:
        assert _parse_int("2.0") == 2

    def test_parse_int_empty(self) -> None:
        assert _parse_int("-") == 0
        assert _parse_int("") == 0


class TestTableParsing:
    def test_parse_markdown_table(self) -> None:
        rows = _parse_table_rows(SAMPLE_ELIGIBILITY_TABLE)
        # Raw parsing includes header row + 5 data rows
        assert len(rows) == 6
        assert rows[0][0] == "Company"
        assert rows[1][0] == "TCS"
        assert rows[2][0] == "Amazon"

    def test_parse_hiring_table(self) -> None:
        rows = _parse_table_rows(SAMPLE_HIRING_TABLE)
        assert len(rows) >= 5
        assert rows[0][0] == "Company"

    def test_empty_text(self) -> None:
        rows = _parse_table_rows("")
        assert rows == []

    def test_no_table(self) -> None:
        rows = _parse_table_rows("Just some text\nwithout any table")
        assert rows == []


class TestSectionSplitting:
    def test_identify_sections(self) -> None:
        sections = _split_sections(SAMPLE_FULL_TEXT)
        assert "eligibility" in sections
        assert "interview" in sections
        assert "hiring" in sections
        assert "trend" in sections
        assert "conflict" in sections
        assert "statistics" in sections

    def test_section_content(self) -> None:
        sections = _split_sections(SAMPLE_FULL_TEXT)
        assert "TCS" in sections["eligibility"]
        assert "Amazon" in sections["eligibility"]
        assert "TCS" in sections["hiring"]

    def test_empty_text(self) -> None:
        sections = _split_sections("")
        assert "unknown" in sections or all(
            v == "" for v in sections.values()
        )

    def test_no_anchors(self) -> None:
        sections = _split_sections("Random text without section headers")
        assert len(sections) > 0


class TestEligibilityExtraction:
    def test_extract_eligibility_profiles(self) -> None:
        profiles = extract_eligibility_profiles(SAMPLE_ELIGIBILITY_TABLE)
        assert len(profiles) == 5

        tcs = profiles[0]
        assert tcs.company == "TCS"
        assert tcs.min_cgpa == 6.0
        assert tcs.max_backlogs == 2
        assert tcs.package_lpa == 4.1
        assert tcs.bond_years == 0
        assert tcs.tech_focus == "Java"

        amazon = profiles[1]
        assert amazon.company == "Amazon"
        assert amazon.min_cgpa == 6.4
        assert amazon.max_backlogs == 1
        assert amazon.package_lpa == 28.6
        assert amazon.bond_years == 2
        assert amazon.tech_focus == "C++"

    def test_empty_text(self) -> None:
        profiles = extract_eligibility_profiles("")
        assert profiles == []


class TestHiringExtraction:
    def test_extract_hiring_distribution(self) -> None:
        hiring = extract_hiring_distribution(SAMPLE_HIRING_TABLE)
        assert len(hiring) >= 4

        tcs = hiring[0]
        assert tcs.company == "TCS"
        assert tcs.sde == 120
        assert tcs.analyst == 80
        assert tcs.intern == 200

    def test_empty_text(self) -> None:
        hiring = extract_hiring_distribution("")
        assert hiring == []


class TestTrendExtraction:
    def test_extract_trends(self) -> None:
        trends = extract_trends(SAMPLE_TREND_TABLE)
        assert len(trends) >= 4

        tcs = trends[0]
        assert tcs.company == "TCS"
        assert tcs.package_2021 == 3.2
        assert tcs.package_2024 == 4.1
        assert tcs.absolute_growth_2021_2024 == 0.9
        assert tcs.trend_label == "up"

    def test_trend_label_down(self) -> None:
        down_data = """
| Company | 2021 | 2022 | 2023 | 2024 |
|---------|------|------|------|------|
| TestCorp | 10.0 | 9.0 | 8.5 | 8.0 |
"""
        trends = extract_trends(down_data)
        assert trends[0].trend_label == "down"

    def test_empty_text(self) -> None:
        trends = extract_trends("")
        assert trends == []


class TestConflictExtraction:
    def test_extract_conflicts(self) -> None:
        conflicts = extract_conflicts(SAMPLE_CONFLICT_TABLE)
        assert len(conflicts) == 2

        amazon = conflicts[0]
        assert amazon.company == "Amazon"
        assert amazon.official_cgpa == 6.4
        assert amazon.portal_cgpa == 7.0
        assert amazon.cgpa_conflict is True
        assert amazon.package_conflict is False

        tcs = conflicts[1]
        assert tcs.company == "TCS"
        assert tcs.cgpa_conflict is False

    def test_empty_text(self) -> None:
        conflicts = extract_conflicts("")
        assert conflicts == []


class TestStatsExtraction:
    def test_extract_overall_stats(self) -> None:
        stats = extract_overall_stats(SAMPLE_STATS_TABLE)
        assert len(stats) >= 4

        tcs = stats[0]
        assert tcs.company == "TCS"
        assert tcs.avg_package == 27.3
        assert tcs.max_offers == 150

    def test_empty_text(self) -> None:
        stats = extract_overall_stats("")
        assert stats == []


class TestFullExtraction:
    def test_extract_all(self) -> None:
        dataset = extract_all(SAMPLE_FULL_TEXT)
        assert isinstance(dataset, PlacementDataset)
        assert len(dataset.eligibility_profiles) == 3
        assert len(dataset.hiring_distributions) == 2
        assert len(dataset.placement_trends) == 2
        assert len(dataset.conflict_records) == 1
        assert len(dataset.overall_stats) == 2

    def test_empty_text(self) -> None:
        dataset = extract_all("")
        assert isinstance(dataset, PlacementDataset)
        assert len(dataset.eligibility_profiles) == 0

    def test_dataset_json_serializable(self) -> None:
        dataset = extract_all(SAMPLE_FULL_TEXT)
        dumped = dataset.model_dump(mode="json")
        assert isinstance(dumped, dict)
        assert "eligibility_profiles" in dumped
        assert len(dumped["eligibility_profiles"]) == 3
        # Verify JSON round-trip
        json_str = json.dumps(dumped)
        restored = PlacementDataset.model_validate(json.loads(json_str))
        assert len(restored.eligibility_profiles) == 3


class TestEdgeCases:
    def test_extract_all_with_none_text(self) -> None:
        dataset = extract_all("   ")
        assert len(dataset.eligibility_profiles) == 0

    def test_parse_int_with_plus(self) -> None:
        assert _parse_int("2+") == 2

    def test_parse_float_with_dollar(self) -> None:
        assert _parse_float("$42.0") == 42.0
