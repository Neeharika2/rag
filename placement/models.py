from typing import Any, List, Optional

from pydantic import BaseModel


class EligibilityProfile(BaseModel):
    company: str
    min_cgpa: float
    max_backlogs: int
    package_lpa: float
    bond_years: int
    key_topics: str
    tech_focus: str
    source_type: str = "official"
    page_number: Optional[int] = None


class InterviewExperience(BaseModel):
    company: str
    technical_focus: str
    round_number: int
    round_title: str
    details: str
    tip: str
    text_hash: Optional[str] = None
    page_number: Optional[int] = None


class HiringDistribution(BaseModel):
    company: str
    sde: int
    analyst: int
    officer: int
    intern: int
    total: int
    source_type: str = "table"
    page_number: Optional[int] = None


class PlacementTrend(BaseModel):
    company: str
    package_2021: float
    package_2022: float
    package_2023: float
    package_2024: float
    absolute_growth_2021_2024: float
    trend_label: str
    page_number: Optional[int] = None


class ConflictRecord(BaseModel):
    company: str
    official_cgpa: float
    portal_cgpa: float
    official_package_lpa: float
    portal_package_lpa: float
    cgpa_conflict: bool
    package_conflict: bool
    page_number: Optional[int] = None


class OverallStats(BaseModel):
    company: str
    avg_package: float
    max_offers: int
    min_offers: int
    avg_cgpa_cutoff: float
    bond_free: bool
    page_number: Optional[int] = None


class PlacementDataset(BaseModel):
    eligibility_profiles: List[EligibilityProfile]
    interview_experiences: List[InterviewExperience]
    hiring_distributions: List[HiringDistribution]
    placement_trends: List[PlacementTrend]
    conflict_records: List[ConflictRecord]
    overall_stats: List[OverallStats]


class StructuredEvidence(BaseModel):
    eligibility: Optional[EligibilityProfile] = None
    hiring: Optional[HiringDistribution] = None
    trend: Optional[PlacementTrend] = None
    conflict: Optional[ConflictRecord] = None
    stats: Optional[OverallStats] = None
    source_sections: List[str] = []


class RoutedQuery(BaseModel):
    query: str
    route: str
    confidence: float
    detected_company: Optional[str] = None
    detected_companies: List[str] = []
    detected_metric: Optional[str] = None
    fallback_reason: Optional[str] = None


class ReasonedAnswer(BaseModel):
    answer: str
    route: str
    evidence: Optional[List[Any]] = None
    confidence: float
    warning: Optional[str] = None
