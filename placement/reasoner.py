import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from placement.fallback import is_low_cgpa_edge_case, low_cgpa_response
from placement.models import (
    ConflictRecord,
    EligibilityProfile,
    HiringDistribution,
    OverallStats,
    PlacementDataset,
    PlacementTrend,
    ReasonedAnswer,
)

logger = logging.getLogger(__name__)

IT_SERVICE_COMPANIES = [
    "TCS", "Infosys", "Deloitte", "Accenture", "Wipro", "Cognizant",
    "Capgemini", "HCL", "Tech Mahindra",
]


class StructuredReasoner:
    def __init__(self, dataset: PlacementDataset) -> None:
        self._dataset = dataset
        self._eligibility_index = {p.company: p for p in dataset.eligibility_profiles}
        self._hiring_index = {h.company: h for h in dataset.hiring_distributions}
        self._trend_index = {t.company: t for t in dataset.placement_trends}
        self._stats_index = {s.company: s for s in dataset.overall_stats}
        self._conflict_index = {c.company: c for c in dataset.conflict_records}

    def answer(self, query: str) -> ReasonedAnswer:
        if not self._dataset.eligibility_profiles:
            return ReasonedAnswer(
                answer="No placement data is available.",
                route="structured_query",
                confidence=0.0,
                warning="empty_dataset",
            )

        if is_low_cgpa_edge_case(query):
            return ReasonedAnswer(
                answer=low_cgpa_response(),
                route="structured_query",
                confidence=1.0,
                warning="below_minimum_cgpa",
            )

        q = query.lower()
        companies = self._extract_companies(query)
        cgpa_filter = self._parse_cgpa_filter(query)
        backlog_filter = self._parse_backlog_filter(query)
        bond_filter = self._parse_bond_filter(query)
        tech_focus = self._parse_tech_focus(query)
        is_it_service = self._is_it_service_query(query)
        hiring_roles = self._parse_hiring_roles(query)
        is_ranking, sort_field, direction = self._parse_rank_sort(query)

        is_conflict = bool(re.search(r"\bconflict|\bdiscrepanc|\bofficial.*portal|\b6\.4.*7\.0|\b7\.0.*6\.4", q))
        is_trend = bool(re.search(r"\b(growth|grew|grown|year[- ]over[- ]year|2021.*2024|absolute\s+growth|max(imum)?\s+growth)\b", q))
        is_comparison = bool(re.search(r"\bcompar(e|ison)|versus|vs|contrast\b", q))
        is_ratio = bool(re.search(r"\b(ratio|package[- ]?to[- ]?cgpa)\b", q))
        is_stats = bool(re.search(r"\b(average|avg|offers?|overall|statistics?)\b", q))

        try:
            if is_conflict:
                return self._resolve_conflict(query, companies)
            if is_comparison:
                return self._resolve_comparison(query, companies, hiring_roles, is_stats)
            if is_trend and not hiring_roles:
                return self._resolve_trend(query, companies)
            if is_ratio:
                return self._resolve_ratio(query)
            if is_stats:
                return self._resolve_stats(query, companies)

            return self._resolve_general(
                query=query,
                companies=companies,
                cgpa_filter=cgpa_filter,
                backlog_filter=backlog_filter,
                bond_filter=bond_filter,
                tech_focus=tech_focus,
                is_it_service=is_it_service,
                hiring_roles=hiring_roles,
                sort_field=sort_field,
                direction=direction,
                is_ranking=is_ranking,
            )
        except Exception as exc:
            logger.error("Reasoner handler error: %s", exc)
            return ReasonedAnswer(
                answer="An error occurred while processing the structured query.",
                route="structured_query",
                confidence=0.0,
                warning=f"handler_error:{exc}",
            )

    def _extract_companies(self, query: str) -> List[str]:
        q = query.lower()
        found = []
        companies = list({p.company for p in self._dataset.eligibility_profiles})
        if not companies:
            companies = list({h.company for h in self._dataset.hiring_distributions})
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

    def _parse_cgpa_filter(self, query: str) -> Optional[Tuple[str, float]]:
        q = query.lower()
        m = re.search(r"cgpa\s*(?:above|greater\s+than|higher\s+than|>)\s*(\d+\.?\d*)", q)
        if m:
            return ">", float(m.group(1))
        m = re.search(r"cgpa\s*(?:below|less\s+than|lower\s+than|<)\s*(\d+\.?\d*)", q)
        if m:
            return "<", float(m.group(1))
        m = re.search(r"(?:i\s+have|student\s+with|with)?\s*(?:a\s+)?cgpa\s*(?:of)?\s*(\d+\.?\d*)(?:\+)?", q)
        if m:
            val = float(m.group(1))
            return "<=", val
        return None

    def _parse_backlog_filter(self, query: str) -> Optional[Tuple[str, int]]:
        q = query.lower()
        if re.search(r"\b(zero|no|0)\s+backlog", q):
            return ">=", 0
        m = re.search(r"(?:allow|accept|at\s+least)?\s*(\d+)\s*backlog", q)
        if m:
            return ">=", int(m.group(1))
        m = re.search(r"backlog[s]?\s*(?:of|allow)?\s*(\d+)", q)
        if m:
            return ">=", int(m.group(1))
        return None

    def _parse_bond_filter(self, query: str) -> Optional[int]:
        q = query.lower()
        if re.search(r"\bno\s+bond\b|\bzero\s+bond\b|\b0\s+bond\b|\bbond[- ]?free\b|\bwithout\s+bond\b", q):
            return 0
        m = re.search(r"bond\s*(?:of)?\s*(\d+)\s*year", q)
        if m:
            return int(m.group(1))
        return None

    def _parse_tech_focus(self, query: str) -> Optional[str]:
        q = query.lower()
        techs = ["Python", "C++", "Java", "Cloud", "System Design", "Aptitude", "OOPs", "DBMS", "OS", "Algorithms"]
        for t in techs:
            if t == "C++":
                pattern = r"\bc\+\+"
            else:
                pattern = r"\b" + re.escape(t.lower()) + r"\b"
            if re.search(pattern, q):
                return t
        return None

    def _is_it_service_query(self, query: str) -> bool:
        q = query.lower()
        return bool(re.search(r"\bit\s+service\b|\bservice\s+firm\b|\bservice\s+compan(y|ies)\b", q))

    def _parse_hiring_roles(self, query: str) -> List[str]:
        q = query.lower()
        roles = []
        if re.search(r"\bsde[s]?\b", q):
            roles.append("sde")
        if re.search(r"\banalyst[s]?\b", q):
            roles.append("analyst")
        if re.search(r"\bofficer[s]?\b", q):
            roles.append("officer")
        if re.search(r"\bintern[s]?\b", q):
            roles.append("intern")
        return roles

    def _parse_rank_sort(self, query: str) -> Tuple[bool, Optional[str], Optional[str]]:
        q = query.lower()
        is_ranking = bool(re.search(r"\brank\b|\blist\b|\bcompar(e|ison)\b", q))
        sort_field = None
        direction = "desc"

        if re.search(r"\b(package|pay|salary|ctc|offered)\b", q):
            sort_field = "package"
        elif re.search(r"\b(growth|grew|grown)\b", q):
            sort_field = "growth"
        elif re.search(r"\bratio\b", q):
            sort_field = "ratio"

        if re.search(r"\b(highest|maximum|max|best|most)\b", q):
            is_ranking = True
            direction = "desc"
        elif re.search(r"\b(lowest|minimum|min|least)\b", q):
            is_ranking = True
            direction = "asc"

        return is_ranking, sort_field, direction

    def _resolve_conflict(self, query: str, companies: List[str]) -> ReasonedAnswer:
        if not companies:
            companies = self._extract_companies(query)

        if companies:
            c_name = companies[0]
            c = self._conflict_index.get(c_name)
            if c:
                warning = None
                answer = (
                    f"There are conflicting records for {c_name}. "
                    f"The official criteria states {c.official_cgpa} CGPA, "
                    f"while the portal record lists {c.portal_cgpa} CGPA. "
                    f"Please verify with the official placement cell."
                )
                if c.official_package_lpa != c.portal_package_lpa:
                    answer += f" Official package: {c.official_package_lpa} LPA, Portal package: {c.portal_package_lpa} LPA."
                if c.cgpa_conflict or c.package_conflict:
                    warning = "conflict_detected"
                return ReasonedAnswer(
                    answer=answer,
                    route="conflict_check",
                    evidence=[c.model_dump()],
                    confidence=0.95,
                    warning=warning,
                )

        conflicted = [c for c in self._dataset.conflict_records if c.cgpa_conflict or c.package_conflict]
        if conflicted:
            names = ", ".join(c.company for c in conflicted)
            return ReasonedAnswer(
                answer=f"Companies with conflicting records: {names}.",
                route="conflict_check",
                evidence=[c.model_dump() for c in conflicted],
                confidence=0.95,
            )
        return ReasonedAnswer(
            answer="No conflicting records found in the dataset.",
            route="conflict_check",
            confidence=0.9,
        )

    def _resolve_comparison(
        self,
        query: str,
        companies: List[str],
        hiring_roles: List[str],
        is_stats: bool,
    ) -> ReasonedAnswer:
        if not companies:
            companies = self._extract_companies(query)
        if len(companies) < 2:
            return ReasonedAnswer(
                answer="Please specify at least two companies to compare.",
                route="structured_query",
                confidence=0.5,
            )

        if hiring_roles:
            role = hiring_roles[0]
            evidence = []
            lines = []
            for c in companies:
                h = self._hiring_index.get(c)
                if h:
                    count = getattr(h, role, 0)
                    lines.append(f"{c} hires {count} {role.upper()} roles")
                    evidence.append(h.model_dump())
            return ReasonedAnswer(
                answer=" | ".join(lines) + f" (Total: " + ", ".join(f"{h['company']}: {h['total']}" for h in evidence) + ").",
                route="structured_query",
                evidence=evidence,
                confidence=0.95,
            )

        q = query.lower()
        all_dimensions = any(w in q for w in ["all dimensions", "eligibility, package, hiring, trend", "every dimension", "full comparison"])

        evidence = []
        comparison_blocks = []
        for c in companies:
            c_evidence = {}
            parts = []
            p = self._eligibility_index.get(c)
            if p:
                c_evidence["eligibility"] = p.model_dump()
                parts.append(f"Eligibility: min CGPA {p.min_cgpa}, max backlogs {p.max_backlogs}, offered package {p.package_lpa} LPA, bond {p.bond_years} years, tech focus: {p.tech_focus}")
            h = self._hiring_index.get(c)
            if h:
                c_evidence["hiring"] = h.model_dump()
                if all_dimensions:
                    parts.append(f"Hiring: {h.sde} SDE, {h.analyst} Analyst, {h.officer} Officer, {h.intern} Intern (Total: {h.total})")
            t = self._trend_index.get(c)
            if t:
                c_evidence["trend"] = t.model_dump()
                if all_dimensions:
                    parts.append(f"Package Trend: 2021: {t.package_2021} LPA → 2024: {t.package_2024} LPA (Growth: +{t.absolute_growth_2021_2024} LPA, {t.trend_label})")
            s = self._stats_index.get(c)
            if s:
                c_evidence["stats"] = s.model_dump()
                if all_dimensions:
                    parts.append(f"Stats: avg package {s.avg_package} LPA, max offers {s.max_offers}")

            evidence.append(c_evidence)
            comparison_blocks.append(f"**{c}**:\n" + "\n- ".join([""] + parts))

        summary = "\n\n".join(comparison_blocks)
        title = f"Comparison of {', '.join(companies)}:"
        return ReasonedAnswer(
            answer=f"{title}\n\n{summary}",
            route="structured_query",
            evidence=evidence,
            confidence=0.95,
        )

    def _resolve_trend(self, query: str, companies: List[str]) -> ReasonedAnswer:
        q = query.lower()
        if companies:
            c_name = companies[0]
            t = self._trend_index.get(c_name)
            if t:
                return ReasonedAnswer(
                    answer=(
                        f"Package trend for {c_name}: "
                        f"2021: {t.package_2021} LPA | 2022: {t.package_2022} LPA | "
                        f"2023: {t.package_2023} LPA | 2024: {t.package_2024} LPA. "
                        f"Absolute growth: +{t.absolute_growth_2021_2024} LPA (trend: {t.trend_label})."
                    ),
                    route="structured_query",
                    evidence=[t.model_dump()],
                    confidence=0.95,
                )

        if re.search(r"\b(growth|grew|grown|absolute\s+growth|max(imum)?\s+growth)\b", q):
            sorted_t = sorted(
                self._dataset.placement_trends,
                key=lambda t: t.absolute_growth_2021_2024,
                reverse=True,
            )
            if sorted_t:
                top = sorted_t[0]
                return ReasonedAnswer(
                    answer=(
                        f"Highest package growth from 2021 to 2024: {top.company} with "
                        f"+{top.absolute_growth_2021_2024} LPA increase "
                        f"({top.package_2021} → {top.package_2024} LPA, trend: {top.trend_label})."
                    ),
                    route="structured_query",
                    evidence=[top.model_dump()],
                    confidence=0.95,
                )

        lines = [
            f"{t.company}: {t.package_2021} → {t.package_2024} (+{t.absolute_growth_2021_2024} LPA)"
            for t in self._dataset.placement_trends
        ]
        return ReasonedAnswer(
            answer="Placement trends: " + " | ".join(lines),
            route="structured_query",
            evidence=[t.model_dump() for t in self._dataset.placement_trends],
            confidence=0.9,
        )

    def _resolve_ratio(self, query: str) -> ReasonedAnswer:
        ratios = []
        for p in self._dataset.eligibility_profiles:
            if p.min_cgpa > 0:
                ratio = p.package_lpa / p.min_cgpa
                ratios.append((p, ratio))
        if ratios:
            ratios.sort(key=lambda x: x[1], reverse=True)
            top, ratio = ratios[0]
            return ReasonedAnswer(
                answer=(
                    f"Best package-to-CGPA ratio: {top.company} at "
                    f"{ratio:.2f} LPA per CGPA point "
                    f"({top.package_lpa} LPA / {top.min_cgpa} CGPA)."
                ),
                route="structured_query",
                evidence=[top.model_dump()],
                confidence=0.9,
            )
        return ReasonedAnswer(
            answer="Could not calculate package-to-CGPA ratio.",
            route="structured_query",
            confidence=0.5,
        )

    def _resolve_stats(self, query: str, companies: List[str]) -> ReasonedAnswer:
        q = query.lower()

        if re.search(r"\bbond[- ]?free\b", q):
            bond_free_companies = [s for s in self._dataset.overall_stats if s.bond_free]
            names = ", ".join(s.company for s in bond_free_companies)
            return ReasonedAnswer(
                answer=f"Bond-free companies (overall stats): {names}.",
                route="structured_query",
                evidence=[s.model_dump() for s in bond_free_companies],
                confidence=0.9,
            )

        if companies:
            c_name = companies[0]
            s = self._stats_index.get(c_name)
            if s:
                return ReasonedAnswer(
                    answer=(
                        f"{c_name} stats: average package {s.avg_package} LPA, "
                        f"max offers {s.max_offers}, min offers {s.min_offers}, "
                        f"avg CGPA cutoff {s.avg_cgpa_cutoff}, bond free: {s.bond_free}."
                    ),
                    route="structured_query",
                    evidence=[s.model_dump()],
                    confidence=0.95,
                )

        if re.search(r"\b(average|avg)\s+package\b", q):
            packages = [s.avg_package for s in self._dataset.overall_stats if s.avg_package > 0]
            if packages:
                avg_pkg = sum(packages) / len(packages)
                return ReasonedAnswer(
                    answer=f"The average package across all companies is {avg_pkg:.2f} LPA.",
                    route="structured_query",
                    evidence=[s.model_dump() for s in self._dataset.overall_stats],
                    confidence=0.9,
                )

        return ReasonedAnswer(
            answer="I can provide statistics such as average package, max/min offers, and bond-free status from the overall stats table.",
            route="structured_query",
            confidence=0.5,
        )

    def _resolve_general(
        self,
        query: str,
        companies: List[str],
        cgpa_filter: Optional[Tuple[str, float]],
        backlog_filter: Optional[Tuple[str, int]],
        bond_filter: Optional[int],
        tech_focus: Optional[str],
        is_it_service: bool,
        hiring_roles: List[str],
        sort_field: Optional[str],
        direction: str,
        is_ranking: bool,
    ) -> ReasonedAnswer:
        q = query.lower()
        candidates = list(self._dataset.eligibility_profiles)
        has_other_filters = any([cgpa_filter, backlog_filter, bond_filter is not None, tech_focus, is_it_service])

        if companies and not has_other_filters:
            c_name = companies[0]
            p = self._eligibility_index.get(c_name)
            if p:
                h = self._hiring_index.get(c_name)
                h_text = f" Hiring SDE: {h.sde}, Analyst: {h.analyst}, Intern: {h.intern}." if h else ""
                return ReasonedAnswer(
                    answer=(
                        f"{c_name} eligibility: minimum CGPA {p.min_cgpa}, "
                        f"max {p.max_backlogs} backlogs, package {p.package_lpa} LPA, "
                        f"bond {p.bond_years} years. Tech focus: {p.tech_focus}.{h_text}"
                    ),
                    route="structured_query",
                    evidence=[p.model_dump()],
                    confidence=0.95,
                )

        if companies:
            candidates = [p for p in candidates if p.company in companies]

        if cgpa_filter:
            op, val = cgpa_filter
            if op == ">":
                candidates = [p for p in candidates if p.min_cgpa > val]
            elif op == "<":
                candidates = [p for p in candidates if p.min_cgpa < val]
            elif op == "<=":
                candidates = [p for p in candidates if p.min_cgpa <= val]

        if backlog_filter:
            op, val = backlog_filter
            candidates = [p for p in candidates if p.max_backlogs >= val]

        if bond_filter is not None:
            candidates = [p for p in candidates if p.bond_years == bond_filter]

        if tech_focus:
            candidates = [p for p in candidates if tech_focus.lower() in p.tech_focus.lower()]

        if is_it_service:
            service_set = set(IT_SERVICE_COMPANIES)
            candidates = [p for p in candidates if p.company in service_set]

        candidate_data = []
        for p in candidates:
            c_data = {
                "company": p.company,
                "min_cgpa": p.min_cgpa,
                "max_backlogs": p.max_backlogs,
                "package_lpa": p.package_lpa,
                "bond_years": p.bond_years,
                "tech_focus": p.tech_focus,
                "profile": p.model_dump(),
            }
            h = self._hiring_index.get(p.company)
            if h:
                c_data["hiring"] = h.model_dump()
                c_data["intern"] = h.intern
                c_data["sde"] = h.sde
                c_data["analyst"] = h.analyst
                c_data["officer"] = h.officer
                c_data["total_hired"] = h.total
            else:
                c_data["intern"] = 0
                c_data["sde"] = 0
                c_data["analyst"] = 0
                c_data["officer"] = 0
                c_data["total_hired"] = 0

            t = self._trend_index.get(p.company)
            if t:
                c_data["trend"] = t.model_dump()
                c_data["growth"] = t.absolute_growth_2021_2024
            else:
                c_data["growth"] = 0.0

            c_data["ratio"] = p.package_lpa / p.min_cgpa if p.min_cgpa > 0 else 0.0
            candidate_data.append(c_data)

        sort_key_field = "package_lpa"
        if sort_field == "growth":
            sort_key_field = "growth"
        elif sort_field == "ratio":
            sort_key_field = "ratio"
        elif hiring_roles:
            sort_key_field = hiring_roles[0]

        reverse = (direction == "desc")
        candidate_data.sort(key=lambda x: x.get(sort_key_field, 0), reverse=reverse)

        if not candidate_data:
            if cgpa_filter and cgpa_filter[0] == "<=" and cgpa_filter[1] < 5.5:
                return ReasonedAnswer(
                    answer=low_cgpa_response(),
                    route="structured_query",
                    confidence=1.0,
                    warning="below_minimum_cgpa",
                )
            return ReasonedAnswer(
                answer="No companies match the specified criteria.",
                route="structured_query",
                confidence=0.9,
                evidence=[],
            )

        evidence = [x["profile"] for x in candidate_data]

        if is_ranking and not ("rank" in q or "list" in q):
            top = candidate_data[0]
            if hiring_roles:
                role = hiring_roles[0]
                val = top.get(role, 0)
                ans = f"{top['company']} hires the most {role.upper()}s among matching companies with {val} positions."
            elif sort_field == "growth":
                ans = f"{top['company']} has the highest package growth with +{top['growth']} LPA."
            else:
                ans = f"{top['company']} offers the highest package at {top['package_lpa']} LPA."

            return ReasonedAnswer(
                answer=ans,
                route="structured_query",
                evidence=[top["profile"]],
                confidence=0.95,
            )

        if "rank" in q or "list" in q or is_ranking:
            lines = []
            for idx, item in enumerate(candidate_data, start=1):
                val_str = ""
                if hiring_roles:
                    role = hiring_roles[0]
                    val_str = f" ({item[role]} {role.upper()} hires)"
                elif sort_field == "growth":
                    val_str = f" (+{item['growth']} LPA growth)"
                elif sort_field == "ratio":
                    val_str = f" (ratio: {item['ratio']:.2f})"
                else:
                    val_str = f" ({item['package_lpa']} LPA)"
                lines.append(f"{idx}. {item['company']}{val_str}")

            ans = "Ranked list of matching companies:\n" + "\n".join(lines)
            return ReasonedAnswer(
                answer=ans,
                route="structured_query",
                evidence=evidence,
                confidence=0.95,
            )

        names = ", ".join(item["company"] for item in candidate_data)
        return ReasonedAnswer(
            answer=f"Matching companies: {names}.",
            route="structured_query",
            evidence=evidence,
            confidence=0.9,
        )
