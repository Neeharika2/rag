from typing import List

from placement.fallback import (
    detect_fallback,
    get_fallback_message,
    is_low_cgpa_edge_case,
    low_cgpa_response,
)
from placement.models import (
    ConflictRecord,
    EligibilityProfile,
    HiringDistribution,
    InterviewExperience,
    OverallStats,
    PlacementDataset,
    PlacementTrend,
)
from placement.query_router import (
    ROUTE_CONFLICT,
    ROUTE_GENERIC,
    ROUTE_INTERVIEW,
    ROUTE_OUT_OF_CORPUS,
    ROUTE_STRUCTURED,
    ROUTE_TREND,
    route_query,
)
from placement.reasoner import StructuredReasoner


def _sample_dataset() -> PlacementDataset:
    return PlacementDataset(
        eligibility_profiles=[
            EligibilityProfile(company="TCS", min_cgpa=6.0, max_backlogs=2, package_lpa=4.1, bond_years=0, key_topics="DSA, SQL", tech_focus="Java"),
            EligibilityProfile(company="Amazon", min_cgpa=6.4, max_backlogs=1, package_lpa=28.6, bond_years=2, key_topics="DSA, C++, LLD", tech_focus="C++"),
            EligibilityProfile(company="Google", min_cgpa=7.0, max_backlogs=0, package_lpa=42.0, bond_years=0, key_topics="DSA, System Design", tech_focus="Python"),
            EligibilityProfile(company="Infosys", min_cgpa=6.5, max_backlogs=1, package_lpa=9.5, bond_years=0, key_topics="DSA, Python", tech_focus="Python"),
            EligibilityProfile(company="Microsoft", min_cgpa=7.0, max_backlogs=0, package_lpa=35.0, bond_years=0, key_topics="DSA, OS, DBMS", tech_focus="C++"),
        ],
        hiring_distributions=[
            HiringDistribution(company="TCS", sde=120, analyst=80, officer=30, intern=200, total=430),
            HiringDistribution(company="Amazon", sde=60, analyst=20, officer=10, intern=40, total=130),
            HiringDistribution(company="Google", sde=40, analyst=15, officer=5, intern=25, total=85),
            HiringDistribution(company="Infosys", sde=90, analyst=45, officer=20, intern=150, total=305),
        ],
        placement_trends=[
            PlacementTrend(company="TCS", package_2021=3.2, package_2022=3.5, package_2023=3.8, package_2024=4.1, absolute_growth_2021_2024=0.9, trend_label="up"),
            PlacementTrend(company="Amazon", package_2021=22.0, package_2022=24.5, package_2023=26.0, package_2024=28.6, absolute_growth_2021_2024=6.6, trend_label="up"),
            PlacementTrend(company="Google", package_2021=35.0, package_2022=38.0, package_2023=40.0, package_2024=42.0, absolute_growth_2021_2024=7.0, trend_label="up"),
        ],
        conflict_records=[
            ConflictRecord(company="Amazon", official_cgpa=6.4, portal_cgpa=7.0, official_package_lpa=28.6, portal_package_lpa=28.6, cgpa_conflict=True, package_conflict=False),
        ],
        overall_stats=[
            OverallStats(company="TCS", avg_package=27.3, max_offers=150, min_offers=20, avg_cgpa_cutoff=6.0, bond_free=True),
            OverallStats(company="Amazon", avg_package=29.5, max_offers=80, min_offers=10, avg_cgpa_cutoff=6.4, bond_free=False),
        ],
        interview_experiences=[
            InterviewExperience(company="Google", technical_focus="DSA, Python", round_number=1, round_title="Technical", details="Coding.", tip="Practice."),
        ],
    )


class TestFallbackDetection:
    def test_campus_visit(self) -> None:
        reason, msg = detect_fallback("When is the campus visit date?")
        assert reason == "campus_visit_date"
        assert "campus visit" in msg.lower()

    def test_stock_price(self) -> None:
        reason, msg = detect_fallback("What is Infosys's current stock price?")
        assert reason == "stock_price"
        assert "stock" in msg.lower()

    def test_wfh(self) -> None:
        reason, msg = detect_fallback("What is the work from home policy at Google?")
        assert reason == "wfh_policy"
        assert "work" in msg.lower()

    def test_global_scope(self) -> None:
        reason, msg = detect_fallback("Which company pays the most in the world?")
        assert reason == "global_scope"
        assert "global" in msg.lower() or "dataset" in msg.lower()

    def test_subjective(self) -> None:
        reason, msg = detect_fallback("Which company is better?")
        assert reason == "subjective_choice"
        assert "better" in msg.lower() or "compar" in msg.lower()

    def test_institution_scope(self) -> None:
        reason, msg = detect_fallback("How are placements in other colleges?")
        assert reason == "institution_scope"
        assert "institution" in msg.lower() or "document" in msg.lower()

    def test_no_fallback(self) -> None:
        reason, msg = detect_fallback("What is the CGPA requirement for TCS?")
        assert reason is None
        assert msg is None

    def test_empty_query(self) -> None:
        reason, msg = detect_fallback("")
        assert reason is None

    def test_lowercase_patterns(self) -> None:
        reason, _ = detect_fallback("STOCK PRICE of amazon")
        assert reason == "stock_price"


class TestFallbackMessages:
    def test_get_message(self) -> None:
        msg = get_fallback_message("stock_price")
        assert "stock" in msg.lower()

    def test_unknown_reason(self) -> None:
        msg = get_fallback_message("unknown_reason")
        assert "document" in msg.lower() or "information" in msg.lower()


class TestLowCgpaEdgeCase:
    def test_below_threshold(self) -> None:
        assert is_low_cgpa_edge_case("I have CGPA 5.0, which company can I apply to?")
        assert is_low_cgpa_edge_case("With CGPA 4.5 am I eligible?")

    def test_above_threshold(self) -> None:
        assert not is_low_cgpa_edge_case("I have CGPA 7.0, what is the best company?")

    def test_no_cgpa(self) -> None:
        assert not is_low_cgpa_edge_case("What is the package for TCS?")

    def test_response_text(self) -> None:
        response = low_cgpa_response()
        assert "5" in response or "CGPA" in response


class TestQueryRouting:
    def test_structured_route(self) -> None:
        r = route_query("What is the CGPA requirement for TCS?")
        assert r.route == ROUTE_STRUCTURED
        assert r.detected_company == "TCS"

    def test_interview_route(self) -> None:
        r = route_query("Tell me about Google's interview rounds")
        assert r.route == ROUTE_INTERVIEW
        assert r.detected_company == "Google"

    def test_trend_route(self) -> None:
        r = route_query("Which company's package grew the most from 2021 to 2024?")
        assert r.route == ROUTE_TREND

    def test_conflict_route(self) -> None:
        r = route_query("Is Amazon's CGPA 6.4 or 7.0? Explain the conflict.")
        assert r.route == ROUTE_CONFLICT

    def test_out_of_corpus_route(self) -> None:
        r = route_query("What is Infosys's current stock price?")
        assert r.route == ROUTE_OUT_OF_CORPUS
        assert r.fallback_reason == "stock_price"

    def test_generic_route(self) -> None:
        r = route_query("Tell me something interesting about placements")
        assert r.route == ROUTE_GENERIC

    def test_empty_query(self) -> None:
        r = route_query("")
        assert r.route == ROUTE_GENERIC
        assert r.confidence == 0.0

    def test_hiring_metric(self) -> None:
        r = route_query("Which company hires the most interns?")
        assert r.route == ROUTE_STRUCTURED
        assert r.detected_metric == "hiring"

    def test_package_metric(self) -> None:
        r = route_query("What is the highest package in the dataset?")
        assert r.route == ROUTE_STRUCTURED
        assert r.detected_metric in {"package", "statistics"}


class TestReasonerEligibility:
    def test_specific_company(self) -> None:
        reasoner = StructuredReasoner(_sample_dataset())
        r = reasoner.answer("What is the CGPA requirement for TCS?")
        assert "TCS" in r.answer
        assert "6.0" in r.answer

    def test_max_backlogs(self) -> None:
        reasoner = StructuredReasoner(_sample_dataset())
        r = reasoner.answer("Which company has max backlogs of 2?")
        assert r.route == "structured_query"
        assert r.confidence > 0.5

    def test_bond_free(self) -> None:
        reasoner = StructuredReasoner(_sample_dataset())
        r = reasoner.answer("Which companies are bond free?")
        assert "TCS" in r.answer or "Google" in r.answer

    def test_min_cgpa_filter(self) -> None:
        reasoner = StructuredReasoner(_sample_dataset())
        r = reasoner.answer("Companies with minimum CGPA greater than 8.0?")
        assert r.route == "structured_query"

    def test_it_service_companies(self) -> None:
        reasoner = StructuredReasoner(_sample_dataset())
        r = reasoner.answer("What is the highest package among IT service companies?")
        assert "TCS" in r.answer or "Infosys" in r.answer


class TestReasonerHiring:
    def test_most_interns(self) -> None:
        reasoner = StructuredReasoner(_sample_dataset())
        r = reasoner.answer("Which company hires the most interns?")
        assert "TCS" in r.answer
        assert "200" in r.answer

    def test_most_sde(self) -> None:
        reasoner = StructuredReasoner(_sample_dataset())
        r = reasoner.answer("Which company hires the most SDEs?")
        assert "TCS" in r.answer

    def test_company_specific(self) -> None:
        reasoner = StructuredReasoner(_sample_dataset())
        r = reasoner.answer("What is the hiring distribution of Amazon?")
        assert "Amazon" in r.answer
        assert "60" in r.answer

    def test_tech_focus_intern(self) -> None:
        reasoner = StructuredReasoner(_sample_dataset())
        r = reasoner.answer("Which Python company hires the most interns?")
        assert "TCS" in r.answer or "Google" in r.answer


class TestReasonerTrends:
    def test_highest_growth(self) -> None:
        reasoner = StructuredReasoner(_sample_dataset())
        r = reasoner.answer("Which company's package grew the most from 2021 to 2024?")
        assert "Google" in r.answer or "Amazon" in r.answer


class TestReasonerConflicts:
    def test_amazon_conflict(self) -> None:
        reasoner = StructuredReasoner(_sample_dataset())
        r = reasoner.answer("Is Amazon's CGPA 6.4 or 7.0?")
        assert "conflicting" in r.answer.lower() or "official" in r.answer.lower()
        assert r.warning == "conflict_detected"

    def test_no_conflict_companies(self) -> None:
        reasoner = StructuredReasoner(_sample_dataset())
        r = reasoner.answer("List all companies with conflicting records")
        assert "Amazon" in r.answer


class TestReasonerMultiCondition:
    def test_h1_cgpa_7_0(self) -> None:
        reasoner = StructuredReasoner(_sample_dataset())
        r = reasoner.answer("I have CGPA 7.0 and 1 backlog, no bond. Best package?")
        assert r.route == "structured_query"
        assert r.confidence >= 0.7

    def test_h3_cgpa_8_0(self) -> None:
        reasoner = StructuredReasoner(_sample_dataset())
        r = reasoner.answer("CGPA 8.0, 0 backlogs, best package?")
        assert r.route == "structured_query"

    def test_low_cgpa_5_0(self) -> None:
        reasoner = StructuredReasoner(_sample_dataset())
        r = reasoner.answer("I have CGPA 5.0, which company can I apply to?")
        assert r.warning == "below_minimum_cgpa"
        assert "5" in r.answer


class TestReasonerStats:
    def test_package_to_cgpa_ratio(self) -> None:
        reasoner = StructuredReasoner(_sample_dataset())
        r = reasoner.answer("Which company has the best package-to-CGPA ratio?")
        assert r.route == "structured_query"
        assert r.confidence > 0.5

    def test_bond_free_stats(self) -> None:
        reasoner = StructuredReasoner(_sample_dataset())
        r = reasoner.answer("Which companies are bond free?")
        assert r.route == "structured_query"

    def test_comparison(self) -> None:
        reasoner = StructuredReasoner(_sample_dataset())
        r = reasoner.answer("Compare Amazon and Google SDE hiring")
        assert r.route == "structured_query"


class TestReasonerEdgeCases:
    def test_empty_dataset(self) -> None:
        empty = PlacementDataset(
            eligibility_profiles=[],
            interview_experiences=[],
            hiring_distributions=[],
            placement_trends=[],
            conflict_records=[],
            overall_stats=[],
        )
        reasoner = StructuredReasoner(empty)
        r = reasoner.answer("Anything")
        assert r.warning == "empty_dataset"

    def test_no_company_no_metric(self) -> None:
        reasoner = StructuredReasoner(_sample_dataset())
        r = reasoner.answer("Some random unrelated question")
        assert r.route == "structured_query"
