from typing import Any, Dict, List, Optional

from generation.answerer import Answerer
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
)


class StubRetriever:
    def __init__(self, hits: Optional[List[Dict[str, Any]]] = None) -> None:
        self._hits = hits or []
        self.last_query: Optional[str] = None
        self.last_filters: Optional[Dict[str, Any]] = None
        self.last_top_k: Optional[int] = None
        self.call_count = 0

    def retrieve(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        original_query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        self.call_count += 1
        self.last_query = query
        self.last_filters = filters
        self.last_top_k = top_k
        return self._hits


class StubGenerator:
    def __init__(self, response: str = "Generated answer.") -> None:
        self._response = response
        self.prompts: List[str] = []
        self.call_count = 0

    def generate(self, prompt: str) -> str:
        self.call_count += 1
        self.prompts.append(prompt)
        return self._response


class StubMetadataStore:
    def __init__(self, dataset: Optional[PlacementDataset] = None) -> None:
        self._dataset = dataset
        self.call_count = 0

    def get_latest_placement_dataset(self) -> Optional[PlacementDataset]:
        self.call_count += 1
        return self._dataset


def _sample_dataset() -> PlacementDataset:
    return PlacementDataset(
        eligibility_profiles=[
            EligibilityProfile(company="TCS", min_cgpa=6.0, max_backlogs=2, package_lpa=4.1, bond_years=0, key_topics="DSA, SQL", tech_focus="Java"),
            EligibilityProfile(company="Amazon", min_cgpa=6.4, max_backlogs=1, package_lpa=28.6, bond_years=2, key_topics="DSA, C++", tech_focus="C++"),
            EligibilityProfile(company="Google", min_cgpa=7.0, max_backlogs=0, package_lpa=42.0, bond_years=0, key_topics="DSA, System Design", tech_focus="Python"),
        ],
        hiring_distributions=[
            HiringDistribution(company="TCS", sde=120, analyst=80, officer=30, intern=200, total=430),
            HiringDistribution(company="Amazon", sde=60, analyst=20, officer=10, intern=40, total=130),
        ],
        placement_trends=[
            PlacementTrend(company="Amazon", package_2021=22.0, package_2022=24.5, package_2023=26.0, package_2024=28.6, absolute_growth_2021_2024=6.6, trend_label="up"),
        ],
        conflict_records=[
            ConflictRecord(company="Amazon", official_cgpa=6.4, portal_cgpa=7.0, official_package_lpa=28.6, portal_package_lpa=28.6, cgpa_conflict=True, package_conflict=False),
        ],
        overall_stats=[
            OverallStats(company="TCS", avg_package=27.3, max_offers=150, min_offers=20, avg_cgpa_cutoff=6.0, bond_free=True),
        ],
        interview_experiences=[
            InterviewExperience(company="Google", technical_focus="DSA", round_number=1, round_title="Technical", details="Coding.", tip="Practice."),
        ],
    )


def _hit(chunk_id: str, text: str, section: str = "", company: str = "", score: float = 0.8) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {"section": section, "company": company}
    return {
        "chunk_id": chunk_id,
        "doc_id": "doc_1",
        "score": score,
        "text": text,
        "metadata": metadata,
    }


def _build_answerer(
    retriever: Optional[StubRetriever] = None,
    generator: Optional[StubGenerator] = None,
    metadata_store: Optional[StubMetadataStore] = None,
) -> Answerer:
    return Answerer(
        retriever=retriever or StubRetriever(),
        generator=generator or StubGenerator(),
        metadata_store=metadata_store,
    )


class TestOutOfCorpus:
    def test_stock_price_returns_fallback(self) -> None:
        answerer = _build_answerer(metadata_store=StubMetadataStore(_sample_dataset()))
        r = answerer.answer("What is the current stock price of TCS?")
        assert r.route == ROUTE_OUT_OF_CORPUS
        assert r.fallback_reason == "stock_price"
        assert "stock" in r.answer.lower() or "real-time" in r.answer.lower()
        assert r.confidence >= 0.9
        assert r.citations == []

    def test_subjective_returns_fallback(self) -> None:
        answerer = _build_answerer()
        r = answerer.answer("Which company is better?")
        assert r.route == ROUTE_OUT_OF_CORPUS
        assert r.fallback_reason == "subjective_choice"
        assert "subject" in r.answer.lower() or "compare" in r.answer.lower()

    def test_campus_visit_date_fallback(self) -> None:
        answerer = _build_answerer()
        r = answerer.answer("When is the campus visit date?")
        assert r.route == ROUTE_OUT_OF_CORPUS
        assert r.fallback_reason == "campus_visit_date"

    def test_out_of_corpus_skips_retriever(self) -> None:
        retriever = StubRetriever()
        answerer = _build_answerer(retriever=retriever)
        answerer.answer("What is the stock price?")
        assert retriever.call_count == 0
        assert retriever.last_query is None

    def test_out_of_corpus_preserves_detected_fields(self) -> None:
        answerer = _build_answerer()
        r = answerer.answer("What is the WFH policy at Amazon?")
        assert r.route == ROUTE_OUT_OF_CORPUS
        assert r.detected_company == "Amazon"


class TestStructuredRoute:
    def test_eligibility_query_uses_reasoner(self) -> None:
        retriever = StubRetriever()
        generator = StubGenerator()
        answerer = _build_answerer(
            retriever=retriever,
            generator=generator,
            metadata_store=StubMetadataStore(_sample_dataset()),
        )
        r = answerer.answer("What is the CGPA requirement for TCS?")
        assert r.route == ROUTE_STRUCTURED
        assert "TCS" in r.answer
        assert "6.0" in r.answer
        assert retriever.call_count == 0
        assert generator.call_count == 0

    def test_low_cgpa_edge_case_returns_warning(self) -> None:
        answerer = _build_answerer(metadata_store=StubMetadataStore(_sample_dataset()))
        r = answerer.answer("I have CGPA 5.0, which company can I apply to?")
        assert r.warning == "below_minimum_cgpa"
        assert r.confidence == 1.0

    def test_conflict_query_returns_evidence(self) -> None:
        answerer = _build_answerer(metadata_store=StubMetadataStore(_sample_dataset()))
        r = answerer.answer("Is the Amazon CGPA cutoff 6.4 or 7.0?")
        assert r.route == ROUTE_CONFLICT
        assert "Amazon" in r.answer
        assert "6.4" in r.answer
        assert "7.0" in r.answer
        assert r.evidence is not None
        assert len(r.evidence) >= 1
        assert r.warning == "conflict_detected"

    def test_trend_query_uses_reasoner(self) -> None:
        answerer = _build_answerer(metadata_store=StubMetadataStore(_sample_dataset()))
        r = answerer.answer("Which company's package grew the most from 2021 to 2024?")
        assert r.route in (ROUTE_TREND, ROUTE_STRUCTURED)
        assert r.confidence > 0.5
        assert "Amazon" in r.answer or "growth" in r.answer.lower()

    def test_hiring_query_uses_reasoner(self) -> None:
        answerer = _build_answerer(metadata_store=StubMetadataStore(_sample_dataset()))
        r = answerer.answer("Which company hires the most interns?")
        assert r.route == ROUTE_STRUCTURED
        assert "TCS" in r.answer

    def test_multi_condition_query(self) -> None:
        answerer = _build_answerer(metadata_store=StubMetadataStore(_sample_dataset()))
        r = answerer.answer("I have CGPA 7.0 and 1 backlog. What's the highest package?")
        assert r.route == ROUTE_STRUCTURED
        assert r.confidence > 0.5
        assert "Amazon" in r.answer

    def test_no_dataset_falls_back_to_generic(self) -> None:
        retriever = StubRetriever(hits=[_hit("c1", "TCS CGPA 6.0", "eligibility", "TCS")])
        generator = StubGenerator("Generic response.")
        answerer = _build_answerer(retriever=retriever, generator=generator, metadata_store=StubMetadataStore(None))
        r = answerer.answer("What is the CGPA requirement for TCS?")
        assert r.route == ROUTE_GENERIC
        assert generator.call_count == 1
        assert retriever.call_count == 1


class TestInterviewRoute:
    def test_interview_route_uses_section_filter(self) -> None:
        retriever = StubRetriever(hits=[_hit("c1", "Google interview round 1: coding.", "interview", "Google")])
        generator = StubGenerator("Google interview: coding round.")
        answerer = _build_answerer(retriever=retriever, generator=generator, metadata_store=StubMetadataStore(_sample_dataset()))
        r = answerer.answer("Tell me about the Google interview process.")
        assert r.route == ROUTE_INTERVIEW
        assert retriever.call_count == 1
        assert retriever.last_filters is not None
        assert retriever.last_filters.get("section") == "interview"
        assert generator.call_count == 1
        assert len(r.citations) == 1
        assert r.citations[0].metadata.get("section") == "interview"

    def test_interview_no_data_returns_fallback(self) -> None:
        retriever = StubRetriever(hits=[])
        answerer = _build_answerer(retriever=retriever, metadata_store=StubMetadataStore(_sample_dataset()))
        r = answerer.answer("Tell me about the Microsoft interview process.")
        assert r.route == ROUTE_INTERVIEW
        assert r.fallback_reason is not None
        assert "interview" in r.fallback_reason or "no_" in r.fallback_reason
        assert r.confidence < 0.5

    def test_interview_uses_higher_top_k(self) -> None:
        retriever = StubRetriever()
        answerer = _build_answerer(retriever=retriever, metadata_store=StubMetadataStore(_sample_dataset()))
        answerer.answer("Tell me about the Google interview process.")
        assert retriever.last_top_k == 8


class TestGenericRoute:
    def test_generic_uses_no_section_filter(self) -> None:
        retriever = StubRetriever(hits=[_hit("c1", "Some context.", "eligibility", "TCS")])
        generator = StubGenerator("Answer.")
        answerer = _build_answerer(retriever=retriever, generator=generator)
        r = answerer.answer("What about something general?")
        assert r.route == ROUTE_GENERIC
        assert retriever.call_count == 1
        assert retriever.last_filters is None or "section" not in (retriever.last_filters or {})

    def test_generic_no_hits_returns_fallback(self) -> None:
        retriever = StubRetriever(hits=[])
        answerer = _build_answerer(retriever=retriever)
        r = answerer.answer("What about something general?")
        assert r.route == ROUTE_GENERIC
        assert r.fallback_reason == "no_relevant_chunks"
        assert r.confidence == 0.0

    def test_generic_uses_default_top_k(self) -> None:
        retriever = StubRetriever()
        answerer = _build_answerer(retriever=retriever)
        answerer.answer("What about something general?")
        assert retriever.last_top_k == 5

    def test_generic_explicit_filters_override(self) -> None:
        retriever = StubRetriever()
        answerer = _build_answerer(retriever=retriever)
        answerer.answer("What about something general?", filters={"section": "hiring"})
        assert retriever.last_filters == {"section": "hiring"}


class TestCitations:
    def test_structured_evidence_becomes_citation(self) -> None:
        answerer = _build_answerer(metadata_store=StubMetadataStore(_sample_dataset()))
        r = answerer.answer("What is the CGPA requirement for TCS?")
        assert len(r.citations) >= 1
        c = r.citations[0]
        assert c.doc_id == "placement_dataset"
        assert c.metadata.get("section") == "structured"
        assert c.metadata.get("company") == "TCS"

    def test_hits_become_citations_with_metadata(self) -> None:
        retriever = StubRetriever(hits=[_hit("c1", "Source text.", "interview", "Google", score=0.85)])
        answerer = _build_answerer(retriever=retriever, metadata_store=StubMetadataStore(_sample_dataset()))
        r = answerer.answer("Tell me about the Google interview process.")
        assert len(r.citations) == 1
        c = r.citations[0]
        assert c.chunk_id == "c1"
        assert c.doc_id == "doc_1"
        assert c.score == 0.85
        assert c.metadata.get("section") == "interview"
        assert c.metadata.get("company") == "Google"

    def test_out_of_corpus_has_no_citations(self) -> None:
        answerer = _build_answerer()
        r = answerer.answer("What is the stock price of TCS?")
        assert r.citations == []
        assert r.evidence is None


class TestMetadataStoreIntegration:
    def test_dataset_loaded_once_per_request(self) -> None:
        store = StubMetadataStore(_sample_dataset())
        answerer = _build_answerer(metadata_store=store)
        answerer.answer("What is the CGPA requirement for TCS?")
        answerer.answer("What is the CGPA requirement for Amazon?")
        assert store.call_count == 2

    def test_missing_dataset_uses_vector_fallback(self) -> None:
        retriever = StubRetriever(hits=[_hit("c1", "Some text.", "eligibility", "TCS")])
        generator = StubGenerator("Generic.")
        answerer = _build_answerer(retriever=retriever, generator=generator, metadata_store=StubMetadataStore(None))
        r = answerer.answer("What is the CGPA requirement for TCS?")
        assert r.route == ROUTE_GENERIC
        assert generator.call_count == 1

    def test_metadata_store_failure_falls_back(self) -> None:
        class FailingStore:
            def get_latest_placement_dataset(self):
                raise RuntimeError("db down")

        retriever = StubRetriever(hits=[_hit("c1", "Some text.", "eligibility", "TCS")])
        generator = StubGenerator("Generic.")
        answerer = _build_answerer(retriever=retriever, generator=generator, metadata_store=FailingStore())
        r = answerer.answer("What is the CGPA requirement for TCS?")
        assert r.route == ROUTE_GENERIC


class TestRewriting:
    def test_rewrite_invoked_when_enabled(self) -> None:
        class StubRewriter:
            def __init__(self) -> None:
                self.queries: List[str] = []
            def rewrite(self, q: str) -> str:
                self.queries.append(q)
                return f"rewritten: {q}"

        rewriter = StubRewriter()
        retriever = StubRetriever(hits=[_hit("c1", "x", "interview", "Google")])
        answerer = Answerer(
            retriever=retriever,
            generator=StubGenerator("ans"),
            metadata_store=StubMetadataStore(_sample_dataset()),
            query_rewriter=rewriter,
        )
        r = answerer.answer("Tell me about the Google interview process.", rewrite=True)
        assert rewriter.queries == ["Tell me about the Google interview process."]
        assert r.rewritten_query == "rewritten: Tell me about the Google interview process."

    def test_rewrite_disabled_keeps_original(self) -> None:
        class StubRewriter:
            def rewrite(self, q: str) -> str:
                return f"rewritten: {q}"

        retriever = StubRetriever(hits=[_hit("c1", "x", "interview", "Google")])
        answerer = Answerer(
            retriever=retriever,
            generator=StubGenerator("ans"),
            metadata_store=StubMetadataStore(_sample_dataset()),
            query_rewriter=StubRewriter(),
        )
        r = answerer.answer("Tell me about the Google interview process.", rewrite=False)
        assert r.rewritten_query is None

    def test_rewrite_failure_continues(self) -> None:
        class FailingRewriter:
            def rewrite(self, q: str) -> str:
                raise RuntimeError("llm down")

        retriever = StubRetriever(hits=[_hit("c1", "x", "interview", "Google")])
        answerer = Answerer(
            retriever=retriever,
            generator=StubGenerator("ans"),
            metadata_store=StubMetadataStore(_sample_dataset()),
            query_rewriter=FailingRewriter(),
        )
        r = answerer.answer("Tell me about the Google interview process.", rewrite=True)
        assert r.rewritten_query is None
        assert r.route == ROUTE_INTERVIEW
