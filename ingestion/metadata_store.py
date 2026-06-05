import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from chunking.recursive import Chunk as TextChunk
from ingestion.models import (
    Base,
    Chunk,
    ConflictRecord as ConflictTable,
    Document,
    EligibilityRecord,
    HiringRecord,
    InterviewRecord,
    QueryLog,
    RetrievalHit,
    StatsRecord,
    TrendRecord,
)
from placement.models import PlacementDataset

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MetadataStore:
    def __init__(self, db_url: str) -> None:
        self._engine = create_engine(db_url, future=True)
        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False)

    def init_db(self) -> None:
        Base.metadata.create_all(self._engine)

    def upsert_document(
        self,
        doc_id: str,
        source: str,
        access_level: str,
        metadata: Dict[str, Any],
    ) -> None:
        with self._session_factory() as session:
            document = Document(
                doc_id=doc_id,
                source=source,
                access_level=access_level,
                metadata_json=json.dumps(metadata),
            )
            session.merge(document)
            session.commit()

    def has_document(self, doc_id: str) -> bool:
        with self._session_factory() as session:
            return session.get(Document, doc_id) is not None

    def upsert_chunks(self, chunks: Iterable[TextChunk]) -> None:
        with self._session_factory() as session:
            for chunk in chunks:
                row = Chunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    text=chunk.text,
                    metadata_json=json.dumps(chunk.metadata),
                )
                session.merge(row)
            session.commit()

    def log_query(
        self,
        query_text: str,
        filters: Optional[Dict[str, Any]],
        top_k: int,
    ) -> str:
        query_id = uuid.uuid4().hex
        with self._session_factory() as session:
            row = QueryLog(
                query_id=query_id,
                query_text=query_text,
                filters_json=json.dumps(filters or {}),
                top_k=top_k,
            )
            session.add(row)
            session.commit()
        return query_id

    def log_hits(self, query_id: str, hits: List[Dict[str, Any]]) -> None:
        with self._session_factory() as session:
            for rank, hit in enumerate(hits, start=1):
                payload = hit.get("payload", {})
                chunk_id = payload.get("chunk_id", hit.get("id"))
                row = RetrievalHit(
                    hit_id=uuid.uuid4().hex,
                    query_id=query_id,
                    chunk_id=chunk_id,
                    score=float(hit.get("score", 0.0)),
                    rank=rank,
                )
                session.add(row)
            session.commit()

    def upsert_placement_dataset(self, doc_id: str, dataset: PlacementDataset) -> None:
        with self._session_factory() as session:
            document = session.get(Document, doc_id)
            if document is None:
                logger.warning("Document %s not found; cannot attach placement dataset", doc_id)
                return
            existing = json.loads(document.metadata_json or "{}")
            existing["placement_dataset"] = dataset.model_dump(mode="json")
            document.metadata_json = json.dumps(existing)
            session.commit()
            logger.info("Attached placement dataset to document %s", doc_id)

    def get_latest_placement_dataset(self) -> Optional[PlacementDataset]:
        with self._session_factory() as session:
            document = (
                session.query(Document)
                .filter(Document.metadata_json.like('%placement_dataset%'))
                .order_by(Document.uploaded_at.desc())
                .first()
            )
            if document is None:
                return None
            metadata = json.loads(document.metadata_json or "{}")
            data = metadata.get("placement_dataset")
            if data is None:
                return None
            return PlacementDataset.model_validate(data)

    def get_placement_dataset(self, doc_id: str) -> Optional[PlacementDataset]:
        with self._session_factory() as session:
            document = session.get(Document, doc_id)
            if document is None:
                return None
            metadata = json.loads(document.metadata_json or "{}")
            data = metadata.get("placement_dataset")
            if data is None:
                return None
            return PlacementDataset.model_validate(data)

    def persist_placement_tables(self, doc_id: str, dataset: PlacementDataset) -> None:
        with self._session_factory() as session:
            self._delete_placement_rows(session, doc_id)

            for p in dataset.eligibility_profiles:
                session.add(EligibilityRecord(
                    doc_id=doc_id, company=p.company, min_cgpa=p.min_cgpa,
                    max_backlogs=p.max_backlogs, package_lpa=p.package_lpa,
                    bond_years=p.bond_years, key_topics=p.key_topics,
                    tech_focus=p.tech_focus, source_type=p.source_type,
                    page_number=p.page_number,
                ))
            for h in dataset.hiring_distributions:
                session.add(HiringRecord(
                    doc_id=doc_id, company=h.company, sde=h.sde,
                    analyst=h.analyst, officer=h.officer, intern=h.intern,
                    total=h.total, source_type=h.source_type,
                    page_number=h.page_number,
                ))
            for t in dataset.placement_trends:
                session.add(TrendRecord(
                    doc_id=doc_id, company=t.company,
                    package_2021=t.package_2021, package_2022=t.package_2022,
                    package_2023=t.package_2023, package_2024=t.package_2024,
                    absolute_growth=t.absolute_growth_2021_2024,
                    trend_label=t.trend_label, page_number=t.page_number,
                ))
            for c in dataset.conflict_records:
                session.add(ConflictTable(
                    doc_id=doc_id, company=c.company,
                    official_cgpa=c.official_cgpa, portal_cgpa=c.portal_cgpa,
                    official_package_lpa=c.official_package_lpa,
                    portal_package_lpa=c.portal_package_lpa,
                    cgpa_conflict=c.cgpa_conflict,
                    package_conflict=c.package_conflict,
                    page_number=c.page_number,
                ))
            for s in dataset.overall_stats:
                session.add(StatsRecord(
                    doc_id=doc_id, company=s.company,
                    avg_package=s.avg_package, max_offers=s.max_offers,
                    min_offers=s.min_offers,
                    avg_cgpa_cutoff=s.avg_cgpa_cutoff,
                    bond_free=s.bond_free, page_number=s.page_number,
                ))
            for iv in dataset.interview_experiences:
                session.add(InterviewRecord(
                    doc_id=doc_id, company=iv.company,
                    round_number=iv.round_number, round_title=iv.round_title,
                    technical_focus=iv.technical_focus, details=iv.details,
                    tip=iv.tip, page_number=iv.page_number,
                ))

            session.commit()
            logger.info("Persisted placement tables for doc_id=%s", doc_id)

    def _delete_placement_rows(self, session, doc_id: str) -> None:
        tables = [EligibilityRecord, HiringRecord, TrendRecord, ConflictTable, StatsRecord, InterviewRecord]
        for table in tables:
            session.query(table).filter(table.doc_id == doc_id).delete()

    def delete_placement_data(self, doc_id: str) -> None:
        with self._session_factory() as session:
            self._delete_placement_rows(session, doc_id)
            document = session.get(Document, doc_id)
            if document is not None:
                meta = json.loads(document.metadata_json or "{}")
                meta.pop("placement_dataset", None)
                document.metadata_json = json.dumps(meta)
            session.commit()
            logger.info("Deleted placement data for doc_id=%s", doc_id)

    def list_eligibility_profiles(self, doc_id: Optional[str] = None) -> list:
        with self._session_factory() as session:
            q = session.query(EligibilityRecord)
            if doc_id:
                q = q.filter(EligibilityRecord.doc_id == doc_id)
            return q.all()

    def list_hiring_distributions(self, doc_id: Optional[str] = None) -> list:
        with self._session_factory() as session:
            q = session.query(HiringRecord)
            if doc_id:
                q = q.filter(HiringRecord.doc_id == doc_id)
            return q.all()

    def list_trends(self, doc_id: Optional[str] = None) -> list:
        with self._session_factory() as session:
            q = session.query(TrendRecord)
            if doc_id:
                q = q.filter(TrendRecord.doc_id == doc_id)
            return q.all()

    def list_conflicts(self, doc_id: Optional[str] = None) -> list:
        with self._session_factory() as session:
            q = session.query(ConflictTable)
            if doc_id:
                q = q.filter(ConflictTable.doc_id == doc_id)
            return q.all()

    def list_overall_stats(self, doc_id: Optional[str] = None) -> list:
        with self._session_factory() as session:
            q = session.query(StatsRecord)
            if doc_id:
                q = q.filter(StatsRecord.doc_id == doc_id)
            return q.all()

    def list_interviews(self, doc_id: Optional[str] = None) -> list:
        with self._session_factory() as session:
            q = session.query(InterviewRecord)
            if doc_id:
                q = q.filter(InterviewRecord.doc_id == doc_id)
            return q.all()
