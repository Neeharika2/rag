import os
import tempfile

from ingestion.metadata_store import MetadataStore
from placement.models import (
    ConflictRecord,
    EligibilityProfile,
    HiringDistribution,
    InterviewExperience,
    OverallStats,
    PlacementDataset,
    PlacementTrend,
)
from vectorstore.chroma_store import ChromaVectorStore

DOC_ID = "test_persist"
SAMPLE_DATASET = PlacementDataset(
    eligibility_profiles=[
        EligibilityProfile(company="TCS", min_cgpa=6.0, max_backlogs=2, package_lpa=4.1, bond_years=0, key_topics="DSA, SQL", tech_focus="Java"),
        EligibilityProfile(company="Amazon", min_cgpa=6.4, max_backlogs=1, package_lpa=28.6, bond_years=2, key_topics="DSA, C++, LLD", tech_focus="C++"),
    ],
    hiring_distributions=[
        HiringDistribution(company="TCS", sde=120, analyst=80, officer=30, intern=200, total=430),
        HiringDistribution(company="Amazon", sde=60, analyst=20, officer=10, intern=40, total=130),
    ],
    placement_trends=[
        PlacementTrend(company="TCS", package_2021=3.2, package_2022=3.5, package_2023=3.8, package_2024=4.1, absolute_growth_2021_2024=0.9, trend_label="up"),
    ],
    conflict_records=[
        ConflictRecord(company="Amazon", official_cgpa=6.4, portal_cgpa=7.0, official_package_lpa=28.6, portal_package_lpa=28.6, cgpa_conflict=True, package_conflict=False),
    ],
    overall_stats=[
        OverallStats(company="TCS", avg_package=27.3, max_offers=150, min_offers=20, avg_cgpa_cutoff=6.0, bond_free=True),
        OverallStats(company="Amazon", avg_package=29.5, max_offers=80, min_offers=10, avg_cgpa_cutoff=6.4, bond_free=False),
    ],
    interview_experiences=[
        InterviewExperience(company="Amazon", technical_focus="DSA", round_number=1, round_title="Technical", details="Coding problems.", tip="Practice."),
    ],
)


def _make_store(db_path: str) -> MetadataStore:
    store = MetadataStore(f"sqlite:///{db_path}")
    store.init_db()
    store.upsert_document(DOC_ID, "test.pdf", "internal", {"test": True})
    return store


def _dispose_store(store: MetadataStore) -> None:
    if hasattr(store, "_engine"):
        store._engine.dispose()


class TestJSONPersistence:
    def test_upsert_and_get(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            store = _make_store(db_path)
            store.upsert_placement_dataset(DOC_ID, SAMPLE_DATASET)
            retrieved = store.get_placement_dataset(DOC_ID)
            assert retrieved is not None
            assert len(retrieved.eligibility_profiles) == 2
            assert retrieved.eligibility_profiles[0].company == "TCS"
        finally:
            _dispose_store(store)
            os.unlink(db_path)

    def test_get_nonexistent(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            store = MetadataStore(f"sqlite:///{db_path}")
            store.init_db()
            result = store.get_placement_dataset("nonexistent")
            assert result is None
        finally:
            _dispose_store(store)
            os.unlink(db_path)

    def test_get_latest(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            store = _make_store(db_path)
            store.upsert_placement_dataset(DOC_ID, SAMPLE_DATASET)
            latest = store.get_latest_placement_dataset()
            assert latest is not None
            assert len(latest.eligibility_profiles) == 2
        finally:
            _dispose_store(store)
            os.unlink(db_path)


class TestNormalizedTablePersistence:
    def _run(self, fn):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        store = _make_store(db_path)
        try:
            fn(store)
        finally:
            _dispose_store(store)
            os.unlink(db_path)

    def test_persist_and_list_eligibility(self) -> None:
        def _test(store):
            store.persist_placement_tables(DOC_ID, SAMPLE_DATASET)
            records = store.list_eligibility_profiles(DOC_ID)
            assert len(records) == 2
            assert records[0].company == "TCS"
            assert records[0].min_cgpa == 6.0
            assert records[1].company == "Amazon"
        self._run(_test)

    def test_persist_and_list_hiring(self) -> None:
        def _test(store):
            store.persist_placement_tables(DOC_ID, SAMPLE_DATASET)
            records = store.list_hiring_distributions(DOC_ID)
            assert len(records) == 2
            assert records[0].company == "TCS"
            assert records[0].sde == 120
        self._run(_test)

    def test_persist_and_list_trends(self) -> None:
        def _test(store):
            store.persist_placement_tables(DOC_ID, SAMPLE_DATASET)
            records = store.list_trends(DOC_ID)
            assert len(records) == 1
            assert records[0].company == "TCS"
            assert records[0].absolute_growth == 0.9
        self._run(_test)

    def test_persist_and_list_conflicts(self) -> None:
        def _test(store):
            store.persist_placement_tables(DOC_ID, SAMPLE_DATASET)
            records = store.list_conflicts(DOC_ID)
            assert len(records) == 1
            assert records[0].company == "Amazon"
            assert records[0].cgpa_conflict is True
        self._run(_test)

    def test_persist_and_list_stats(self) -> None:
        def _test(store):
            store.persist_placement_tables(DOC_ID, SAMPLE_DATASET)
            records = store.list_overall_stats(DOC_ID)
            assert len(records) == 2
            assert records[0].company == "TCS"
            assert records[0].bond_free is True
            assert records[1].bond_free is False
        self._run(_test)

    def test_persist_and_list_interviews(self) -> None:
        def _test(store):
            store.persist_placement_tables(DOC_ID, SAMPLE_DATASET)
            records = store.list_interviews(DOC_ID)
            assert len(records) == 1
            assert records[0].company == "Amazon"
            assert records[0].round_number == 1
        self._run(_test)

    def test_list_without_doc_id(self) -> None:
        def _test(store):
            store.persist_placement_tables(DOC_ID, SAMPLE_DATASET)
            all_eligibility = store.list_eligibility_profiles()
            assert len(all_eligibility) == 2
        self._run(_test)

    def test_re_persist_replaces(self) -> None:
        def _test(store):
            store.persist_placement_tables(DOC_ID, SAMPLE_DATASET)
            store.persist_placement_tables(DOC_ID, SAMPLE_DATASET)
            records = store.list_eligibility_profiles(DOC_ID)
            assert len(records) == 2
        self._run(_test)


class TestDeletePlacementData:
    def _run(self, fn):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        store = _make_store(db_path)
        try:
            fn(store)
        finally:
            _dispose_store(store)
            os.unlink(db_path)

    def test_delete_clears_json(self) -> None:
        def _test(store):
            store.upsert_placement_dataset(DOC_ID, SAMPLE_DATASET)
            assert store.get_placement_dataset(DOC_ID) is not None
            store.delete_placement_data(DOC_ID)
            assert store.get_placement_dataset(DOC_ID) is None
        self._run(_test)

    def test_delete_clears_tables(self) -> None:
        def _test(store):
            store.persist_placement_tables(DOC_ID, SAMPLE_DATASET)
            assert len(store.list_eligibility_profiles(DOC_ID)) == 2
            store.delete_placement_data(DOC_ID)
            assert len(store.list_eligibility_profiles(DOC_ID)) == 0
            assert len(store.list_hiring_distributions(DOC_ID)) == 0
        self._run(_test)


class TestChromaStore:
    def test_delete_by_doc_id(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = ChromaVectorStore(persist_dir=tmpdir, collection_name="test_collection")
            from chunking.recursive import Chunk
            chunk = Chunk(
                chunk_id="test_chunk_1",
                doc_id="test_doc",
                text="test text",
                page_start=1,
                page_end=1,
                metadata={"doc_id": "test_doc"},
            )
            store.upsert([[0.1, 0.2, 0.3]], [chunk])
            store.delete_by_doc_id("test_doc")

    def test_reset_collection(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = ChromaVectorStore(persist_dir=tmpdir, collection_name="test_collection")
            store.reset_collection()
            results = store.search([0.1, 0.2, 0.3], top_k=5)
            assert len(results) == 0
