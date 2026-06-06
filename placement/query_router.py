import re
from typing import Any, List, Optional

from placement.fallback import detect_fallback
from placement.models import RoutedQuery

ROUTE_STRUCTURED = "structured_query"
ROUTE_INTERVIEW = "interview_text"
ROUTE_TREND = "trend_query"
ROUTE_CONFLICT = "conflict_check"
ROUTE_OUT_OF_CORPUS = "out_of_corpus"
ROUTE_GENERIC = "generic_vector"

ROUTE_METADATA_KEYS = {
    ROUTE_STRUCTURED: "section=eligibility/hiring/statistics",
    ROUTE_INTERVIEW: "section=interview",
    ROUTE_TREND: "section=trend",
    ROUTE_CONFLICT: "section=conflict",
    ROUTE_OUT_OF_CORPUS: "n/a",
    ROUTE_GENERIC: "any",
}

CONFLICT_KEYWORDS = [
    r"\bconflict(ing)?\b",
    r"\bdiscrepanc(y|ies)\b",
    r"\bwhich\s+is\s+correct\b",
    r"\bwhich\s+value\s+is\s+right\b",
    r"\bofficial\s+(criteria|record)\b",
    r"\bportal\s+(record|value)\b",
    r"\b(is\s+the\s+.+\s+)?(6\.4|7\.0)\s*(or|and)\s*(7\.0|6\.4)\b",
    r"\bconflict\b",
]

TREND_KEYWORDS = [
    r"\btrend\b",
    r"\b(growth|grew)\b",
    r"\byear[- ]?over[- ]?year\b",
    r"\b2021\s*(to|-)\s*2024\b",
    r"\bpackage\s+(over\s+the\s+years|history|trajectory)\b",
    r"\b(grown|growth|increase[d]?)\b",
    r"\b2021\b.*\b2022\b.*\b2023\b.*\b2024\b",
    r"\babsolute\s+growth\b",
]

INTERVIEW_KEYWORDS = [
    r"\binterview\b",
    r"\bround\s*\d+\b",
    r"\btechnical\s+round\b",
    r"\bHR\s+round\b",
    r"\bmanagerial\b",
    r"\btip[s]?\b",
    r"\bhow\s+to\s+prepare\b",
    r"\binterview\s+(experience|process|question)\b",
    r"\b DSA\b",
    r"\bcoding\s+(round|interview)\b",
]

STATISTICS_KEYWORDS = [
    r"\bavg(erage)?\s+package\b",
    r"\boverall\s+statistic\b",
    r"\bmax\s+offers?\b",
    r"\bmin\s+offers?\b",
    r"\bbond[- ]?free\b",
    r"\bcompar(e|ison)\b",
    r"\bbest\s+(package|company|offer)\b",
    r"\bhighest\s+paying\b",
    r"\branking\b",
    r"\bbond\s+free\b",
]

CGPA_KEYWORDS = [
    r"\bCGPA\b",
    r"\bGPA\b",
    r"\bbacklog[s]?\b",
    r"\bbond\b",
    r"\beligib(le|ility)\b",
    r"\bmin(imum)?\s+CGPA\b",
    r"\bmin(imum)?\s+GPA\b",
]

HIRING_KEYWORDS = [
    r"\bhire[ds]?\b",
    r"\bhiring\b",
    r"\bSDE\b",
    r"\banalyst\b",
    r"\bofficer\b",
    r"\bintern[s]?\b",
    r"\b(total|number\s+of)\s+(offers?|hires?)\b",
    r"\bmost\s+interns?\b",
    r"\bmost\s+SDEs?\b",
]

PACKAGE_KEYWORDS = [
    r"\bpackage\b",
    r"\bLPA\b",
    r"\bsalary\b",
    r"\bCTC\b",
    r"\boffer(red)?\b",
    r"\bpay(ing|s)?\b",
]

import logging
logger = logging.getLogger(__name__)

_metadata_store = None

def get_metadata_store():
    global _metadata_store
    if _metadata_store is None:
        from settings import Settings
        from ingestion.metadata_store import MetadataStore
        settings = Settings.from_env()
        _metadata_store = MetadataStore(settings.metadata_db_url)
    return _metadata_store

def get_dynamic_companies() -> List[str]:
    try:
        db = get_metadata_store()
        dataset = db.get_latest_placement_dataset()
        if dataset and dataset.eligibility_profiles:
            return list({p.company for p in dataset.eligibility_profiles})
    except Exception as exc:
        logger.warning("Failed to load companies dynamically from DB: %s", exc)
    
    return [
        "TCS", "Amazon", "Google", "Infosys", "Microsoft", "Intel", "IBM",
        "Accenture", "Wipro", "Cognizant", "Capgemini", "HCL", "Tech Mahindra",
        "Deloitte", "Flipkart", "Samsung", "L&T Infotech", "PayPal", "Oracle",
        "Adobe", "Goldman Sachs", "JP Morgan", "Salesforce",
    ]

def get_dynamic_conflict_companies() -> List[str]:
    try:
        db = get_metadata_store()
        dataset = db.get_latest_placement_dataset()
        if dataset and dataset.conflict_records:
            return list({c.company for c in dataset.conflict_records if c.cgpa_conflict or c.package_conflict})
    except Exception as exc:
        logger.warning("Failed to load conflict companies dynamically from DB: %s", exc)
    
    return ["Amazon", "TCS", "Google", "Infosys", "Microsoft"]

def __getattr__(name: str) -> Any:
    if name == "COMPANIES":
        return get_dynamic_companies()
    if name == "CONFLICT_COMPANIES":
        return get_dynamic_conflict_companies()
    raise AttributeError(f"module {__name__} has no attribute {name}")



def route_query(query: str) -> RoutedQuery:
    if not query or not query.strip():
        return RoutedQuery(
            query=query or "",
            route=ROUTE_GENERIC,
            confidence=0.0,
            fallback_reason="empty_query",
        )

    fallback_reason, _ = detect_fallback(query)
    if fallback_reason:
        return RoutedQuery(
            query=query,
            route=ROUTE_OUT_OF_CORPUS,
            confidence=0.95,
            fallback_reason=fallback_reason,
        )

    detected_companies = _detect_companies(query)
    detected_company = detected_companies[0] if detected_companies else None
    detected_metric = _detect_metric(query)

    if _matches_any(query, CONFLICT_KEYWORDS) or (any(c in get_dynamic_conflict_companies() for c in detected_companies) and _mentions_value_pair(query)):
        return RoutedQuery(
            query=query,
            route=ROUTE_CONFLICT,
            confidence=0.9,
            detected_company=detected_company,
            detected_companies=detected_companies,
            detected_metric="conflict",
        )

    if _matches_any(query, TREND_KEYWORDS):
        return RoutedQuery(
            query=query,
            route=ROUTE_TREND,
            confidence=0.85,
            detected_company=detected_company,
            detected_companies=detected_companies,
            detected_metric="trend",
        )

    if _matches_any(query, INTERVIEW_KEYWORDS):
        return RoutedQuery(
            query=query,
            route=ROUTE_INTERVIEW,
            confidence=0.85,
            detected_company=detected_company,
            detected_companies=detected_companies,
            detected_metric="interview",
        )

    if _matches_any(query, STATISTICS_KEYWORDS) or _matches_any(query, HIRING_KEYWORDS) or _matches_any(query, CGPA_KEYWORDS) or _matches_any(query, PACKAGE_KEYWORDS):
        confidence = 0.8
        return RoutedQuery(
            query=query,
            route=ROUTE_STRUCTURED,
            confidence=confidence,
            detected_company=detected_company,
            detected_companies=detected_companies,
            detected_metric=detected_metric,
        )

    return RoutedQuery(
        query=query,
        route=ROUTE_GENERIC,
        confidence=0.5,
        detected_company=detected_company,
        detected_companies=detected_companies,
    )


def _matches_any(query: str, patterns: List[str]) -> bool:
    q = query.lower()
    for p in patterns:
        if re.search(p, q, re.IGNORECASE):
            return True
    return False


def _detect_companies(query: str) -> List[str]:
    q = query.lower()
    companies = get_dynamic_companies()
    found = []
    for c in sorted(companies, key=len, reverse=True):
        if c == "Samsung R&D":
            pattern = r"samsung\s+r\s*&\s*d"
        elif c == "L&T Infotech":
            pattern = r"l\s*&\s*t\s+infotech"
        else:
            pattern = r"\b" + re.escape(c.lower()) + r"\b"
        if re.search(pattern, q):
            found.append(c)
    return found


def _detect_company(query: str) -> Optional[str]:
    companies = _detect_companies(query)
    return companies[0] if companies else None


def _detect_metric(query: str) -> Optional[str]:
    q = query.lower()
    if _matches_any(query, HIRING_KEYWORDS):
        return "hiring"
    if _matches_any(query, TREND_KEYWORDS):
        return "trend"
    if _matches_any(query, STATISTICS_KEYWORDS):
        return "statistics"
    if _matches_any(query, CGPA_KEYWORDS):
        return "eligibility"
    if _matches_any(query, PACKAGE_KEYWORDS):
        return "package"
    if _matches_any(query, INTERVIEW_KEYWORDS):
        return "interview"
    if _matches_any(query, CONFLICT_KEYWORDS):
        return "conflict"
    return None


def _mentions_value_pair(query: str) -> bool:
    return bool(re.search(r"\b6\.4\b.*\b7\.0\b|\b7\.0\b.*\b6\.4\b", query))
