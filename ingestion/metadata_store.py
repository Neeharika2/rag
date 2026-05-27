import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from chunking.recursive import Chunk as TextChunk
from ingestion.models import Base, Chunk, Document, QueryLog, RetrievalHit


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
