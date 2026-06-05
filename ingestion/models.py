from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(String, primary_key=True)
    source: Mapped[str] = mapped_column(String, nullable=False)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
    access_level: Mapped[str] = mapped_column(String, default="internal")
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")


class Chunk(Base):
    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(String, primary_key=True)
    doc_id: Mapped[str] = mapped_column(String, ForeignKey("documents.doc_id"))
    page_start: Mapped[int] = mapped_column(Integer, nullable=True)
    page_end: Mapped[int] = mapped_column(Integer, nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)


class QueryLog(Base):
    __tablename__ = "query_logs"

    query_id: Mapped[str] = mapped_column(String, primary_key=True)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    filters_json: Mapped[str] = mapped_column(Text, default="{}")
    top_k: Mapped[int] = mapped_column(Integer, default=5)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)


class RetrievalHit(Base):
    __tablename__ = "retrieval_hits"

    hit_id: Mapped[str] = mapped_column(String, primary_key=True)
    query_id: Mapped[str] = mapped_column(String, ForeignKey("query_logs.query_id"))
    chunk_id: Mapped[str] = mapped_column(String, ForeignKey("chunks.chunk_id"))
    score: Mapped[float] = mapped_column(Float)
    rank: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)


class EligibilityRecord(Base):
    __tablename__ = "placement_eligibility"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(String, ForeignKey("documents.doc_id"))
    company: Mapped[str] = mapped_column(String)
    min_cgpa: Mapped[float] = mapped_column(Float)
    max_backlogs: Mapped[int] = mapped_column(Integer)
    package_lpa: Mapped[float] = mapped_column(Float)
    bond_years: Mapped[int] = mapped_column(Integer)
    key_topics: Mapped[str] = mapped_column(Text, default="")
    tech_focus: Mapped[str] = mapped_column(Text, default="")
    source_type: Mapped[str] = mapped_column(String, default="official")
    page_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)


class HiringRecord(Base):
    __tablename__ = "placement_hiring"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(String, ForeignKey("documents.doc_id"))
    company: Mapped[str] = mapped_column(String)
    sde: Mapped[int] = mapped_column(Integer)
    analyst: Mapped[int] = mapped_column(Integer)
    officer: Mapped[int] = mapped_column(Integer)
    intern: Mapped[int] = mapped_column(Integer)
    total: Mapped[int] = mapped_column(Integer)
    source_type: Mapped[str] = mapped_column(String, default="table")
    page_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)


class TrendRecord(Base):
    __tablename__ = "placement_trends"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(String, ForeignKey("documents.doc_id"))
    company: Mapped[str] = mapped_column(String)
    package_2021: Mapped[float] = mapped_column(Float)
    package_2022: Mapped[float] = mapped_column(Float)
    package_2023: Mapped[float] = mapped_column(Float)
    package_2024: Mapped[float] = mapped_column(Float)
    absolute_growth: Mapped[float] = mapped_column(Float)
    trend_label: Mapped[str] = mapped_column(String)
    page_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)


class ConflictRecord(Base):
    __tablename__ = "placement_conflicts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(String, ForeignKey("documents.doc_id"))
    company: Mapped[str] = mapped_column(String)
    official_cgpa: Mapped[float] = mapped_column(Float)
    portal_cgpa: Mapped[float] = mapped_column(Float)
    official_package_lpa: Mapped[float] = mapped_column(Float)
    portal_package_lpa: Mapped[float] = mapped_column(Float)
    cgpa_conflict: Mapped[bool] = mapped_column(Boolean)
    package_conflict: Mapped[bool] = mapped_column(Boolean)
    page_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)


class StatsRecord(Base):
    __tablename__ = "placement_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(String, ForeignKey("documents.doc_id"))
    company: Mapped[str] = mapped_column(String)
    avg_package: Mapped[float] = mapped_column(Float)
    max_offers: Mapped[int] = mapped_column(Integer)
    min_offers: Mapped[int] = mapped_column(Integer)
    avg_cgpa_cutoff: Mapped[float] = mapped_column(Float)
    bond_free: Mapped[bool] = mapped_column(Boolean)
    page_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)


class InterviewRecord(Base):
    __tablename__ = "placement_interviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(String, ForeignKey("documents.doc_id"))
    company: Mapped[str] = mapped_column(String)
    round_number: Mapped[int] = mapped_column(Integer)
    round_title: Mapped[str] = mapped_column(Text, default="")
    technical_focus: Mapped[str] = mapped_column(Text, default="")
    details: Mapped[str] = mapped_column(Text, default="")
    tip: Mapped[str] = mapped_column(Text, default="")
    page_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
