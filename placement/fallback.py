import re
from typing import Optional, Tuple

OUT_OF_CORPUS_PATTERNS = [
    (r"\bcampus\s+visit\b.*\b(date|time|when)\b", "campus_visit_date"),
    (r"\bcampus\s+visit\s+date\b", "campus_visit_date"),
    (r"\b(when|what\s+date).*\bvisit\b", "campus_visit_date"),

    (r"\b(stock|share)\s+price\b", "stock_price"),
    (r"\bcurrent\s+price\b", "stock_price"),
    (r"\bstock\s+market\b", "stock_price"),
    (r"\bshare\s+trading\b", "stock_price"),
    (r"\bmarket\s+cap\b", "stock_price"),

    (r"\bwork\s*from\s*home\b", "wfh_policy"),
    (r"\bWFH\s+policy\b", "wfh_policy"),
    (r"\bremote\s+work\s+policy\b", "wfh_policy"),
    (r"\bhybrid\s+policy\b", "wfh_policy"),

    (r"\bpays?\s+the\s+most\s+in\s+the\s+world\b", "global_scope"),
    (r"\bbest\s+company\s+in\s+the\s+world\b", "global_scope"),
    (r"\bhighest\s+paying\s+company\s+(in\s+the\s+world|globally)\b", "global_scope"),
    (r"\bglobal\s+(highest|best)\b", "global_scope"),

    (r"\b(other\s+colleges|other\s+institutes|peer\s+institutions|nationally)\b", "institution_scope"),
    (r"\b(in\s+India|in\s+the\s+US|in\s+the\s+world)\s+(placements?|hiring|salary)\b", "institution_scope"),

    (r"\b(should\s+i\s+(join|choose|prefer))\b", "subjective_choice"),
    (r"\bwhich\s+is\s+better\b", "subjective_choice"),
    (r"\bwhat\s+do\s+you\s+(recommend|suggest|advise)\b", "subjective_choice"),
    (r"\bworth\s+it\b", "subjective_choice"),
    (r"\bbetter\s+than\b", "subjective_choice"),
    (r"\bis\s+better\b", "subjective_choice"),
    (r"\bwhich\s+company\s+is\s+better\b", "subjective_choice"),
]

FALLBACK_MESSAGES = {
    "campus_visit_date": (
        "I don't have information about campus visit dates in the provided "
        "placement documents. Please contact the official placement cell for scheduling details."
    ),
    "stock_price": (
        "Current stock price is real-time market data and is not available in the "
        "placement documents. Please check a financial data provider for live stock information."
    ),
    "wfh_policy": (
        "Work-from-home policy is not specified in the provided placement documents. "
        "Policies vary by team, project, and current company policy."
    ),
    "global_scope": (
        "The provided placement documents cover only the companies listed in this dataset. "
        "I can identify the highest-paying company in this dataset, but cannot compare against "
        "global market data."
    ),
    "institution_scope": (
        "Placement data for other institutions is not in the provided documents. "
        "This dataset covers placement information for the specific institution in the source PDF."
    ),
    "subjective_choice": (
        "The documents do not define one company as universally better. I can compare "
        "them on objective factors such as package, bond, eligibility, hiring distribution, "
        "and interview focus to help you make your own decision."
    ),
}

LOW_CGPA_THRESHOLD = 5.0
MIN_CGPA_FLOOR_THRESHOLD = 5.5


def detect_fallback(query: str) -> Tuple[Optional[str], Optional[str]]:
    if not query or not query.strip():
        return None, None
    q = query.lower().strip()
    for pattern, reason in OUT_OF_CORPUS_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            return reason, FALLBACK_MESSAGES.get(reason, _generic_fallback(q))
    return None, None


def get_fallback_message(reason: str) -> str:
    return FALLBACK_MESSAGES.get(reason, _generic_fallback(reason))


def _generic_fallback(query: str) -> str:
    return (
        "I don't have enough information in the provided placement documents to answer that. "
        "Please rephrase your question or ask about specific eligibility, hiring, package, or interview topics."
    )


def is_low_cgpa_edge_case(query: str) -> bool:
    q = query.lower()
    cgpa_match = re.search(r"cgpa\s*[=<>!]+\s*(\d+\.?\d*)", q)
    if cgpa_match:
        try:
            cgpa = float(cgpa_match.group(1))
            return cgpa <= LOW_CGPA_THRESHOLD
        except (ValueError, TypeError):
            return False
    match = re.search(r"\b(\d+\.?\d*)\s*(?:cgpa|gpa)\b", q)
    if match:
        try:
            cgpa = float(match.group(1))
            return cgpa <= LOW_CGPA_THRESHOLD
        except (ValueError, TypeError):
            return False
    match = re.search(r"(?:cgpa|gpa)\s*[=:>]*\s*(\d+\.?\d*)", q)
    if match:
        try:
            cgpa = float(match.group(1))
            return cgpa <= LOW_CGPA_THRESHOLD
        except (ValueError, TypeError):
            return False
    return False


def low_cgpa_response() -> str:
    return (
        f"With a CGPA below {LOW_CGPA_THRESHOLD}, no company in the provided placement dataset "
        "has minimum eligibility requirements met. Companies typically require a minimum "
        f"CGPA of {MIN_CGPA_FLOOR_THRESHOLD} or higher."
    )
