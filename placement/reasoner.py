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

        handler, params = self._classify_intent(query)
        if handler is None:
            return ReasonedAnswer(
                answer=self._generic_structured_answer(query),
                route="structured_query",
                confidence=0.5,
            )

        try:
            return handler(**params)
        except Exception as exc:
            logger.error("Reasoner handler error: %s", exc)
            return ReasonedAnswer(
                answer="An error occurred while processing the structured query.",
                route="structured_query",
                confidence=0.0,
                warning=f"handler_error:{exc}",
            )

    def _classify_intent(self, query: str) -> Tuple[Optional[Any], Dict[str, Any]]:
        q = query.lower()
        company = self._extract_company(query)

        if re.search(r"\bconflict|\bdiscrepanc|\bofficial.*portal|\b6\.4.*7\.0|\b7\.0.*6\.4", q):
            return self._answer_conflict, {"query": query, "company": company}
        if re.search(r"\b(growth|grew|year[- ]over[- ]year|2021.*2024|absolute\s+growth|max(imum)?\s+growth)\b", q) and not re.search(r"\bhire|sde|analyst|officer|intern", q):
            return self._answer_trend_query, {"query": query}
        if re.search(r"\b(hire|hir(ing|e)|sde|analyst|officer|intern|most\s+interns?|max(imum)?\s+interns?|max(imum)?\s+sdes?|most\s+sdes?|most\s+analysts?)\b", q):
            return self._answer_hiring_query, {"query": query, "company": company}
        if re.search(r"\b(ratio|package[- ]?to[- ]?cgpa)\b", q):
            return self._answer_stats_query, {"query": query, "company": company}
        if re.search(r"\b(cgpa|backlogs?|bonds?|eligibility|packages?|lpa|salary|ctc|min(imum)?\s*cgpa|tech[_ ]?focus|python|java|c\+\+)\b", q):
            return self._answer_eligibility_query, {"query": query, "company": company}
        if re.search(r"\b(statistic|average|avg|compar(e|ison)|ranking|best|highest)\b", q):
            return self._answer_stats_query, {"query": query, "company": company}
        return None, {}

    def _extract_company(self, query: str) -> Optional[str]:
        companies = sorted(
            set(self._eligibility_index.keys())
            | set(self._hiring_index.keys())
            | set(self._trend_index.keys())
            | set(self._stats_index.keys())
            | set(self._conflict_index.keys()),
            key=len,
            reverse=True,
        )
        for c in companies:
            if c.lower() in query.lower():
                return c
        return None

    def _answer_eligibility_query(
        self, query: str, company: Optional[str] = None
    ) -> ReasonedAnswer:
        q = query.lower()

        if re.search(r"\bmax(imum)?\s+backlog[s]?\b", q):
            n = self._extract_int(q, r"max(imum)?\s+backlog[s]?\s*(?:of|=|:)?\s*(\d+)")
            if n is not None:
                filtered = [p for p in self._dataset.eligibility_profiles if p.max_backlogs >= n]
                sorted_profiles = sorted(filtered, key=lambda p: p.package_lpa, reverse=True)
                if sorted_profiles:
                    top = sorted_profiles[0]
                    return ReasonedAnswer(
                        answer=(
                            f"Among companies accepting {n}+ backlogs, the highest package is "
                            f"{top.company} at {top.package_lpa} LPA (allows up to {top.max_backlogs} backlogs)."
                        ),
                        route="structured_query",
                        evidence=[top.model_dump()],
                        confidence=0.9,
                    )

        if re.search(r"\bmin(imum)?\s+cgpa\b", q):
            cgpa = self._extract_float(q, r"min(imum)?\s+cgpa\s*[:=]?\s*(\d+\.?\d*)")
            if cgpa is not None:
                filtered = [p for p in self._dataset.eligibility_profiles if p.min_cgpa > cgpa]
                if filtered:
                    sorted_profiles = sorted(filtered, key=lambda p: p.package_lpa, reverse=True)
                    names = ", ".join(f"{p.company} ({p.min_cgpa})" for p in sorted_profiles)
                    return ReasonedAnswer(
                        answer=(
                            f"Companies with minimum CGPA > {cgpa}: {names}. "
                            f"Highest package: {sorted_profiles[0].company} at {sorted_profiles[0].package_lpa} LPA."
                        ),
                        route="structured_query",
                        evidence=[p.model_dump() for p in sorted_profiles],
                        confidence=0.9,
                    )

        if re.search(r"\bbond\s*(free|=\s*0)\b|\bno\s+bond\b|\b0\s+bond\b", q):
            filtered = [p for p in self._dataset.eligibility_profiles if p.bond_years == 0]
            sorted_profiles = sorted(filtered, key=lambda p: p.package_lpa, reverse=True)
            if sorted_profiles:
                names = ", ".join(f"{p.company} ({p.package_lpa} LPA)" for p in sorted_profiles)
                return ReasonedAnswer(
                    answer=f"Bond-free companies (no service bond): {names}.",
                    route="structured_query",
                    evidence=[p.model_dump() for p in sorted_profiles],
                    confidence=0.9,
                )

        if re.search(r"\bit\s+service\b|\bservice\s+compan(y|ies)\b", q):
            service_set = set(IT_SERVICE_COMPANIES)
            filtered = [p for p in self._dataset.eligibility_profiles if p.company in service_set]
            if filtered:
                sorted_profiles = sorted(filtered, key=lambda p: p.package_lpa, reverse=True)
                top = sorted_profiles[0]
                return ReasonedAnswer(
                    answer=(
                        f"Among IT service companies, the highest package is "
                        f"{top.company} at {top.package_lpa} LPA."
                    ),
                    route="structured_query",
                    evidence=[top.model_dump()],
                    confidence=0.9,
                )

        if re.search(r"\btech[_ ]?focus\b|\bpython\b|\bc\+\+\b|\bjava\b", q):
            focus = self._extract_tech_focus(q)
            if focus:
                filtered = [p for p in self._dataset.eligibility_profiles
                            if focus.lower() in p.tech_focus.lower()]
                if filtered:
                    sorted_profiles = sorted(filtered, key=lambda p: p.package_lpa, reverse=True)
                    top = sorted_profiles[0]
                    return ReasonedAnswer(
                        answer=(
                            f"Among {focus}-focused companies, the highest package is "
                            f"{top.company} at {top.package_lpa} LPA."
                        ),
                        route="structured_query",
                        evidence=[top.model_dump()],
                        confidence=0.9,
                    )

        if re.search(r"\b(cgpa|backlog).*\b\b(bond|package)\b", q) or re.search(r"\b(i\s+have|i\s+am).*cgpa", q):
            cgpa = self._extract_float(q, r"(\d+\.?\d*)\s*cgpa") or self._extract_float(q, r"cgpa\s*[:=]?\s*(\d+\.?\d*)")
            backlogs = self._extract_int(q, r"(\d+)\s*backlog")
            bond_zero = bool(re.search(r"\bno\s+bond\b|\bbond\s*=\s*0\b|\bzero\s+bond\b", q))

            if cgpa is not None:
                filtered = []
                for p in self._dataset.eligibility_profiles:
                    if p.min_cgpa > cgpa:
                        continue
                    if backlogs is not None and p.max_backlogs < backlogs:
                        continue
                    if bond_zero and p.bond_years != 0:
                        continue
                    filtered.append(p)
                sorted_profiles = sorted(filtered, key=lambda p: p.package_lpa, reverse=True)
                if sorted_profiles:
                    top = sorted_profiles[0]
                    parts = [f"CGPA <= {cgpa}"]
                    if backlogs is not None:
                        parts.append(f"backlogs <= {backlogs}")
                    if bond_zero:
                        parts.append("no bond")
                    conditions = ", ".join(parts)
                    return ReasonedAnswer(
                        answer=(
                            f"For students with {conditions}, the best package is "
                            f"{top.company} at {top.package_lpa} LPA "
                            f"(requires min CGPA {top.min_cgpa}, max {top.max_backlogs} backlogs, bond {top.bond_years} years)."
                        ),
                        route="structured_query",
                        evidence=[top.model_dump()],
                        confidence=0.95,
                    )
                return ReasonedAnswer(
                    answer=f"No companies match the criteria (CGPA {cgpa}).",
                    route="structured_query",
                    confidence=0.9,
                )

        if company and company in self._eligibility_index:
            p = self._eligibility_index[company]
            return ReasonedAnswer(
                answer=(
                    f"{company} eligibility: minimum CGPA {p.min_cgpa}, "
                    f"max {p.max_backlogs} backlogs, package {p.package_lpa} LPA, "
                    f"bond {p.bond_years} years. Tech focus: {p.tech_focus}."
                ),
                route="structured_query",
                evidence=[p.model_dump()],
                confidence=0.95,
            )

        return ReasonedAnswer(
            answer=self._generic_structured_answer(query),
            route="structured_query",
            confidence=0.6,
        )

    def _answer_hiring_query(
        self, query: str, company: Optional[str] = None
    ) -> ReasonedAnswer:
        q = query.lower()

        if re.search(r"\bmost\s+intern[s]?\b|\bmax(imum)?\s+interns?\b", q):
            sorted_h = sorted(self._dataset.hiring_distributions, key=lambda h: h.intern, reverse=True)
            if sorted_h:
                top = sorted_h[0]
                return ReasonedAnswer(
                    answer=f"{top.company} hires the most interns with {top.intern} intern positions.",
                    route="structured_query",
                    evidence=[top.model_dump()],
                    confidence=0.95,
                )

        if re.search(r"\bmost\s+sdes?\b|\bmax(imum)?\s+sdes?\b", q):
            sorted_h = sorted(self._dataset.hiring_distributions, key=lambda h: h.sde, reverse=True)
            if sorted_h:
                top = sorted_h[0]
                return ReasonedAnswer(
                    answer=f"{top.company} hires the most SDEs with {top.sde} SDE positions.",
                    route="structured_query",
                    evidence=[top.model_dump()],
                    confidence=0.95,
                )

        if re.search(r"\bmost\s+analyst[s]?\b", q):
            sorted_h = sorted(self._dataset.hiring_distributions, key=lambda h: h.analyst, reverse=True)
            if sorted_h:
                top = sorted_h[0]
                return ReasonedAnswer(
                    answer=f"{top.company} hires the most analysts with {top.analyst} positions.",
                    route="structured_query",
                    evidence=[top.model_dump()],
                    confidence=0.95,
                )

        if re.search(r"\b(python|tech[_ ]?focus|technology)\b.*\b(intern|hire)\b", q):
            focus = self._extract_tech_focus(q)
            if focus:
                tech_companies = {p.company for p in self._dataset.eligibility_profiles
                                  if focus.lower() in p.tech_focus.lower()}
                filtered = [h for h in self._dataset.hiring_distributions if h.company in tech_companies]
                if filtered:
                    sorted_h = sorted(filtered, key=lambda h: h.intern, reverse=True)
                    top = sorted_h[0]
                    return ReasonedAnswer(
                        answer=(
                            f"Among {focus}-focused companies, {top.company} hires the most "
                            f"interns with {top.intern} intern positions."
                        ),
                        route="structured_query",
                        evidence=[top.model_dump()],
                        confidence=0.9,
                    )

        if company and company in self._hiring_index:
            h = self._hiring_index[company]
            return ReasonedAnswer(
                answer=(
                    f"{company} hiring distribution: {h.sde} SDE, {h.analyst} Analyst, "
                    f"{h.officer} Officer, {h.intern} Intern. Total: {h.total}."
                ),
                route="structured_query",
                evidence=[h.model_dump()],
                confidence=0.95,
            )

        if re.search(r"\bcompar(e|ison)\b", q) and re.search(r"\band\b", q):
            companies = self._extract_companies_in_query(query, list(self._hiring_index.keys()))
            if len(companies) >= 2:
                comp_data = [self._hiring_index[c] for c in companies if c in self._hiring_index]
                lines = [
                    f"{c.company}: {c.sde} SDE, {c.analyst} Analyst, {c.officer} Officer, {c.intern} Intern"
                    for c in comp_data
                ]
                return ReasonedAnswer(
                    answer="Hiring comparison: " + " | ".join(lines),
                    route="structured_query",
                    evidence=[c.model_dump() for c in comp_data],
                    confidence=0.9,
                )

        return ReasonedAnswer(
            answer=self._generic_structured_answer(query),
            route="structured_query",
            confidence=0.6,
        )

    def _answer_trend_query(self, query: str) -> ReasonedAnswer:
        q = query.lower()

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
                        f"{top.absolute_growth_2021_2024} LPA increase "
                        f"({top.package_2021} → {top.package_2024} LPA, trend: {top.trend_label})."
                    ),
                    route="structured_query",
                    evidence=[top.model_dump()],
                    confidence=0.95,
                )

        if re.search(r"\btrend\b", q):
            sorted_t = sorted(self._dataset.placement_trends, key=lambda t: t.trend_label)
            ups = [t for t in sorted_t if t.trend_label == "up"]
            if ups:
                top = sorted(ups, key=lambda t: t.absolute_growth_2021_2024, reverse=True)[0]
                return ReasonedAnswer(
                    answer=(
                        f"Best growing company: {top.company} (2021: {top.package_2021} LPA → "
                        f"2024: {top.package_2024} LPA, growth: +{top.absolute_growth_2021_2024} LPA)."
                    ),
                    route="structured_query",
                    evidence=[top.model_dump()],
                    confidence=0.9,
                )

        return ReasonedAnswer(
            answer=self._generic_structured_answer(query),
            route="structured_query",
            confidence=0.6,
        )

    def _answer_conflict(self, query: str, company: Optional[str] = None) -> ReasonedAnswer:
        target = company
        if target is None:
            for c in self._conflict_index.keys():
                if c.lower() in query.lower():
                    target = c
                    break
        if target and target in self._conflict_index:
            c = self._conflict_index[target]
            warning = None
            answer = (
                f"There are conflicting records for {target}. "
                f"The official criteria states {c.official_cgpa} CGPA, "
                f"while the portal record lists {c.portal_cgpa} CGPA. "
                f"Please verify with the official placement cell."
            )
            if c.cgpa_conflict:
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

    def _answer_stats_query(
        self, query: str, company: Optional[str] = None
    ) -> ReasonedAnswer:
        q = query.lower()

        if re.search(r"\b(best|highest).*package[- ]?to[- ]?cgpa\b|\bratio\b", q):
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

        if re.search(r"\bcompar(e|ison)\b", q) and re.search(r"\band\b", q):
            companies = self._extract_companies_in_query(query, list(self._eligibility_index.keys()))
            if len(companies) >= 2:
                comp_data = [self._eligibility_index[c] for c in companies if c in self._eligibility_index]
                lines = [
                    f"{c.company}: min CGPA {c.min_cgpa}, max {c.max_backlogs} backlogs, {c.package_lpa} LPA, bond {c.bond_years} years"
                    for c in comp_data
                ]
                return ReasonedAnswer(
                    answer="Eligibility comparison: " + " | ".join(lines),
                    route="structured_query",
                    evidence=[c.model_dump() for c in comp_data],
                    confidence=0.9,
                )

        if re.search(r"\bbond[- ]?free\b", q):
            bond_free_companies = [s for s in self._dataset.overall_stats if s.bond_free]
            names = ", ".join(s.company for s in bond_free_companies)
            return ReasonedAnswer(
                answer=f"Bond-free companies (overall stats): {names}.",
                route="structured_query",
                evidence=[s.model_dump() for s in bond_free_companies],
                confidence=0.9,
            )

        if company and company in self._stats_index:
            s = self._stats_index[company]
            return ReasonedAnswer(
                answer=(
                    f"{company} stats: average package {s.avg_package} LPA, "
                    f"max offers {s.max_offers}, min offers {s.min_offers}, "
                    f"avg CGPA cutoff {s.avg_cgpa_cutoff}, bond free: {s.bond_free}."
                ),
                route="structured_query",
                evidence=[s.model_dump()],
                confidence=0.9,
            )

        return ReasonedAnswer(
            answer=self._generic_structured_answer(query),
            route="structured_query",
            confidence=0.6,
        )

    def _generic_structured_answer(self, query: str) -> str:
        return (
            "I can answer questions about eligibility, hiring distribution, package trends, "
            "conflicts, overall statistics, or interview experiences for the companies in the dataset."
        )

    def _extract_int(self, text: str, pattern: str) -> Optional[int]:
        m = re.search(pattern, text)
        if m:
            for g in reversed(m.groups()):
                if g is None:
                    continue
                try:
                    return int(g)
                except (ValueError, TypeError):
                    continue
        return None

    def _extract_float(self, text: str, pattern: str) -> Optional[float]:
        m = re.search(pattern, text)
        if m:
            for g in reversed(m.groups()):
                if g is None:
                    continue
                try:
                    return float(g)
                except (ValueError, TypeError):
                    continue
        return None

    def _extract_tech_focus(self, text: str) -> Optional[str]:
        techs = ["Python", "C++", "Java", "C#", "Go", "Rust", "JavaScript", "TypeScript", "Ruby", "PHP"]
        for t in techs:
            if t.lower() in text:
                return t
        return None

    def _extract_companies_in_query(self, query: str, candidates: List[str]) -> List[str]:
        found = []
        q = query.lower()
        for c in candidates:
            if c.lower() in q:
                found.append(c)
        return found
