import logging
import re
from typing import Dict, List, Optional, Tuple

from placement.models import (
    ConflictRecord,
    EligibilityProfile,
    HiringDistribution,
    InterviewExperience,
    OverallStats,
    PlacementDataset,
    PlacementTrend,
)

logger = logging.getLogger(__name__)

COMPANY_NORMALIZE: Dict[str, str] = {
    "samsung r&d;": "Samsung R&D",
    "samsung r&d": "Samsung R&D",
    "l&t infotech": "L&T Infotech",
    "l & t infotech": "L&T Infotech",
    "l&t; infotech": "L&T Infotech",
    "tech mahindra": "Tech Mahindra",
}

EXPECTED_COUNTS = {
    "eligibility_profiles": (15, 25),
    "hiring_distributions": (15, 25),
    "placement_trends": (8, 15),
    "conflict_records": (3, 8),
    "overall_stats": (15, 25),
}

SECTION_ANCHORS: List[Tuple[str, re.Pattern]] = [
    ("eligibility", re.compile(r"(?:section\s*1|eligibility(?:\s+table)?)\s*", re.IGNORECASE)),
    ("interview", re.compile(r"(?:section\s*2|interview(?:\s+experience)?)\s*", re.IGNORECASE)),
    ("hiring", re.compile(r"(?:section\s*3|hiring(?:\s+(?:chart|distribution|table))?)\s*", re.IGNORECASE)),
    ("examples", re.compile(r"(?:section\s*4|multi[- ]?hop|worked\s+example|reasoning\s+example)\s*", re.IGNORECASE)),
    ("trend", re.compile(r"(?:section\s*5|trend|temporal|package\s+trend|growth)\s*", re.IGNORECASE)),
    ("conflict", re.compile(r"(?:section\s*6|conflict(?:ing)?|discrepancy)\s*", re.IGNORECASE)),
    ("statistics", re.compile(r"(?:section\s*7|statistics|overall\s+stat|avg\s+package)\s*", re.IGNORECASE)),
]

SECTION_END: re.Pattern = re.compile(r"(?:^|\n)(?=Section\s+\d|##+\s+Section)", re.IGNORECASE | re.MULTILINE)


def extract_all(text: str) -> PlacementDataset:
    if not text or not text.strip():
        logger.warning("extract_all received empty text")
        return PlacementDataset(
            eligibility_profiles=[],
            interview_experiences=[],
            hiring_distributions=[],
            placement_trends=[],
            conflict_records=[],
            overall_stats=[],
        )

    sections = _split_sections(text)

    eligibility = extract_eligibility_profiles(sections.get("eligibility", ""))
    interviews = extract_interview_experiences(sections.get("interview", ""))
    hiring = extract_hiring_distribution(sections.get("hiring", ""))
    trends = extract_trends(sections.get("trend", ""))
    conflicts = extract_conflicts(sections.get("conflict", ""))
    stats = extract_overall_stats(sections.get("statistics", ""))

    dataset = PlacementDataset(
        eligibility_profiles=eligibility,
        interview_experiences=interviews,
        hiring_distributions=hiring,
        placement_trends=trends,
        overall_stats=stats,
        conflict_records=conflicts,
    )

    _validate_counts(dataset)
    return dataset


def _is_header_row(row: List[str]) -> bool:
    first = row[0].strip().lower() if row else ""
    return first in {"company", "sl no", "s.no", "s no"} or "company" in first or "name" in first


def extract_eligibility_profiles(text: str) -> List[EligibilityProfile]:
    rows = _parse_table_rows(text)
    profiles: List[EligibilityProfile] = []
    for row in rows:
        if len(row) < 4:
            continue
        if _is_header_row(row):
            continue
        company = _normalize_company(row[0])
        try:
            profile = EligibilityProfile(
                company=company,
                min_cgpa=_parse_float(row[1]),
                max_backlogs=_parse_int(row[2]),
                package_lpa=_parse_float(row[3]),
                bond_years=_parse_int(row[4]) if len(row) > 4 else 0,
                key_topics=row[5] if len(row) > 5 else "",
                tech_focus=row[6] if len(row) > 6 else "",
                source_type="official",
            )
            profiles.append(profile)
        except (ValueError, TypeError) as exc:
            logger.warning("Skipping eligibility row %s: %s", row, exc)
    logger.info("Extracted %d eligibility profiles", len(profiles))
    return profiles


def extract_interview_experiences(text: str) -> List[InterviewExperience]:
    experiences: List[InterviewExperience] = []
    company_blocks = _split_interview_companies(text)

    for company, block in company_blocks:
        rounds = _parse_interview_rounds(block)
        for rnd in rounds:
            try:
                experience = InterviewExperience(
                    company=company,
                    technical_focus=rnd.get("tech_focus", ""),
                    round_number=rnd.get("round_number", 1),
                    round_title=rnd.get("round_title", ""),
                    details=rnd.get("details", ""),
                    tip=rnd.get("tip", ""),
                )
                experiences.append(experience)
            except (ValueError, TypeError) as exc:
                logger.warning("Skipping interview round for %s: %s", company, exc)

    logger.info("Extracted %d interview experiences", len(experiences))
    return experiences


def extract_hiring_distribution(text: str) -> List[HiringDistribution]:
    rows = _parse_table_rows(text)
    distributions: List[HiringDistribution] = []
    for row in rows:
        if len(row) < 5:
            continue
        if _is_header_row(row):
            continue
        company = _normalize_company(row[0])
        try:
            distribution = HiringDistribution(
                company=company,
                sde=_parse_int(row[1]),
                analyst=_parse_int(row[2]),
                officer=_parse_int(row[3]),
                intern=_parse_int(row[4]),
                total=_parse_int(row[5]) if len(row) > 5 else 0,
                source_type="table",
            )
            distributions.append(distribution)
        except (ValueError, TypeError) as exc:
            logger.warning("Skipping hiring row %s: %s", row, exc)
    logger.info("Extracted %d hiring distributions", len(distributions))
    return distributions


def extract_trends(text: str) -> List[PlacementTrend]:
    rows = _parse_table_rows(text)
    trends: List[PlacementTrend] = []
    for row in rows:
        if len(row) < 5:
            continue
        if _is_header_row(row):
            continue
        company = _normalize_company(row[0])
        try:
            p2021 = _parse_float(row[1])
            p2022 = _parse_float(row[2])
            p2023 = _parse_float(row[3])
            p2024 = _parse_float(row[4])
            growth = round(p2024 - p2021, 2)
            trend_label = "up" if growth > 0 else ("down" if growth < 0 else "stable")
            trend = PlacementTrend(
                company=company,
                package_2021=p2021,
                package_2022=p2022,
                package_2023=p2023,
                package_2024=p2024,
                absolute_growth_2021_2024=growth,
                trend_label=trend_label,
            )
            trends.append(trend)
        except (ValueError, TypeError) as exc:
            logger.warning("Skipping trend row %s: %s", row, exc)
    logger.info("Extracted %d placement trends", len(trends))
    return trends


def extract_conflicts(text: str) -> List[ConflictRecord]:
    rows = _parse_table_rows(text)
    conflicts: List[ConflictRecord] = []
    for row in rows:
        if len(row) < 4:
            continue
        company = _normalize_company(row[0])
        try:
            off_cgpa = _parse_float(row[1])
            port_cgpa = _parse_float(row[2])
            off_pkg = _parse_float(row[3])
            port_pkg = _parse_float(row[4]) if len(row) > 4 else off_pkg
            conflict = ConflictRecord(
                company=company,
                official_cgpa=off_cgpa,
                portal_cgpa=port_cgpa,
                official_package_lpa=off_pkg,
                portal_package_lpa=port_pkg,
                cgpa_conflict=abs(off_cgpa - port_cgpa) > 0.01,
                package_conflict=abs(off_pkg - port_pkg) > 0.01,
            )
            conflicts.append(conflict)
        except (ValueError, TypeError) as exc:
            logger.warning("Skipping conflict row %s: %s", row, exc)
    logger.info("Extracted %d conflict records", len(conflicts))
    return conflicts


def extract_overall_stats(text: str) -> List[OverallStats]:
    rows = _parse_table_rows(text)
    stats_list: List[OverallStats] = []
    for row in rows:
        if len(row) < 4:
            continue
        company = _normalize_company(row[0])
        try:
            stat = OverallStats(
                company=company,
                avg_package=_parse_float(row[1]),
                max_offers=_parse_int(row[2]),
                min_offers=_parse_int(row[3]),
                avg_cgpa_cutoff=_parse_float(row[4]) if len(row) > 4 else 0.0,
                bond_free=(row[5].strip().lower() in {"yes", "y", "true"}) if len(row) > 5 else False,
            )
            stats_list.append(stat)
        except (ValueError, TypeError) as exc:
            logger.warning("Skipping stats row %s: %s", row, exc)
    logger.info("Extracted %d overall stats", len(stats_list))
    return stats_list


def _normalize_company(name: str) -> str:
    cleaned = name.strip().rstrip(";,")
    lower = cleaned.lower()
    if lower in COMPANY_NORMALIZE:
        return COMPANY_NORMALIZE[lower]
    return cleaned


def _parse_float(value: str) -> float:
    cleaned = value.strip().replace(",", "").replace("$", "").replace("LPA", "").replace("lpa", "")
    if not cleaned or cleaned in {"-", "--", "N/A", "na", "NA", ""}:
        return 0.0
    return float(cleaned)


def _parse_int(value: str) -> int:
    cleaned = value.strip().replace(",", "").replace("+", "")
    if not cleaned or cleaned in {"-", "--", "N/A", "na", "NA", ""}:
        return 0
    return int(float(cleaned))


def _split_sections(text: str) -> Dict[str, str]:
    boundaries: List[Tuple[int, str]] = []
    for name, pattern in SECTION_ANCHORS:
        for m in pattern.finditer(text):
            boundaries.append((m.start(), name))

    boundaries.sort(key=lambda x: x[0])

    if not boundaries:
        logger.warning("No section anchors found; returning full text as unknown")
        return {"unknown": text}

    segments: Dict[str, List[str]] = {}
    for i, (pos, name) in enumerate(boundaries):
        next_pos = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        chunk = text[pos:next_pos].strip()
        segments.setdefault(name, []).append(chunk)

    result: Dict[str, str] = {}
    for name, chunks in segments.items():
        result[name] = "\n\n".join(chunks)

    for name in dict(SECTION_ANCHORS):
        if name not in result:
            logger.debug("Section '%s' not found in document", name)

    return result


def _parse_table_rows(text: str) -> List[List[str]]:
    lines = text.split("\n")
    rows: List[List[str]] = []
    in_table = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            in_table = False
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            in_table = True
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            if not cells:
                continue
            if re.match(r"^[\s\-:|+]+\s*$", stripped):
                continue
            rows.append(cells)
        elif in_table and "|" in stripped:
            cells = [c.strip() for c in stripped.split("|")]
            rows.append(cells)

    if not rows:
        rows = _parse_fallback_table(text)

    return rows


def _parse_fallback_table(text: str) -> List[List[str]]:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    potential_rows: List[List[str]] = []
    for line in lines:
        if re.match(r"^[A-Z][a-zA-Z\s&.]+", line):
            parts = re.split(r"\s{2,}|\t", line)
            if len(parts) >= 3:
                potential_rows.append(parts)
    return potential_rows


def _split_interview_companies(text: str) -> List[Tuple[str, str]]:
    company_pattern = re.compile(
        r"(?:(?:^|\n)(?:##+\s*|(?:\d+\.)?\s*)\*{0,2}({companies})\*{0,2})",
        re.IGNORECASE | re.MULTILINE,
    )
    blocks: List[Tuple[str, str]] = []
    last_pos = 0
    last_company: Optional[str] = None

    for m in company_pattern.finditer(text):
        if last_company is not None:
            blocks.append((last_company, text[last_pos : m.start()].strip()))
        last_company = m.group(1).strip()
        last_pos = m.start()

    if last_company is not None:
        blocks.append((last_company, text[last_pos:].strip()))

    return blocks


def _parse_interview_rounds(block: str) -> List[Dict]:
    round_pattern = re.compile(
        r"(?:Round\s*(\d+)|Technical\s*Round|HR\s*Round|Managerial\s*Round)",
        re.IGNORECASE,
    )
    tip_pattern = re.compile(r"(?:Tip|Advice|Suggestion)\s*:?\s*(.*)", re.IGNORECASE)
    lines = block.split("\n")
    rounds: List[Dict] = []
    current_round: Optional[Dict] = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        rm = round_pattern.search(stripped)
        if rm:
            if current_round:
                rounds.append(current_round)
            rnum = 1
            try:
                rnum = int(rm.group(1)) if rm.group(1) else 1
            except (ValueError, IndexError):
                rnum = len(rounds) + 1
            current_round = {
                "round_number": rnum,
                "round_title": stripped,
                "tech_focus": "",
                "details": "",
                "tip": "",
            }
            continue

        tm = tip_pattern.match(stripped)
        if tm and current_round:
            current_round["tip"] = (current_round["tip"] + " " + tm.group(1)).strip()
            continue

        if current_round:
            tech_keywords = ["DSA", "C++", "Python", "Java", "SQL", "OS", "DBMS", "Networking", "System Design"]
            for kw in tech_keywords:
                if kw.lower() in stripped.lower():
                    current_round["tech_focus"] = (
                        (current_round["tech_focus"] + ", " + kw) if current_round["tech_focus"] else kw
                    )
            current_round["details"] = (current_round["details"] + " " + stripped).strip()
        else:
            current_round = {
                "round_number": 1,
                "round_title": "General",
                "tech_focus": "",
                "details": stripped,
                "tip": "",
            }

    if current_round:
        rounds.append(current_round)

    return rounds


def _validate_counts(dataset: PlacementDataset) -> None:
    field_map = {
        "eligibility_profiles": dataset.eligibility_profiles,
        "hiring_distributions": dataset.hiring_distributions,
        "placement_trends": dataset.placement_trends,
        "conflict_records": dataset.conflict_records,
        "overall_stats": dataset.overall_stats,
    }
    for name, records in field_map.items():
        lo, hi = EXPECTED_COUNTS.get(name, (0, 999))
        count = len(records)
        if count < lo:
            logger.warning(
                "Expected %d-%d %s, got %d (low)", lo, hi, name, count,
            )
        elif count > hi:
            logger.warning(
                "Expected %d-%d %s, got %d (high)", lo, hi, name, count,
            )
        else:
            logger.info(
                "Validated %s: %d records (expected %d-%d)", name, count, lo, hi,
            )
