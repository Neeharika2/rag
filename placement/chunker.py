import hashlib
import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from chunking.recursive import Chunk
from placement.models import (
    ConflictRecord,
    EligibilityProfile,
    HiringDistribution,
    InterviewExperience,
    OverallStats,
    PlacementDataset,
    PlacementTrend,
)

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

logger = logging.getLogger(__name__)

INTERVIEW_MAX_TOKENS = 300
INTERVIEW_MIN_TOKENS = 100

# ---------------------------------------------------------------------------
# Chunk text templates
# Each template uses a semantically distinct *opening phrase* so that the
# embedding model places each chunk type in a separate region of vector space.
# This prevents eligibility chunks from being retrieved for hiring queries
# and vice-versa.
# ---------------------------------------------------------------------------

# Sections that are already fully covered by per-row structured chunks and
# should NOT be redundantly indexed as raw markdown text.
_RAW_SECTION_SKIP = {"hiring", "trend", "statistics"}


class PlacementChunker:
    def __init__(
        self,
        doc_id: str,
        source: str,
        base_metadata: Optional[Dict[str, Any]] = None,
        dedupe_threshold: float = 0.95,
    ) -> None:
        self._doc_id = doc_id
        self._source = source
        self._base = dict(base_metadata or {})
        self._dedupe_threshold = dedupe_threshold
        self._chunk_index = 0
        import tiktoken
        self._encoder = tiktoken.get_encoding("cl100k_base")

    def chunk_dataset(self, dataset: PlacementDataset) -> List[Chunk]:
        chunks: List[Chunk] = []

        chunks.extend(self._chunk_eligibility(dataset.eligibility_profiles))
        chunks.extend(self._chunk_hiring(dataset.hiring_distributions))
        chunks.extend(self._chunk_trends(dataset.placement_trends))
        chunks.extend(self._chunk_conflicts(dataset.conflict_records))
        chunks.extend(self._chunk_stats(dataset.overall_stats))

        interview_chunks = self._chunk_interviews(dataset.interview_experiences)
        interview_chunks = self._deduplicate_interviews(interview_chunks)
        chunks.extend(interview_chunks)

        # Chunk other sections — skip sections already fully covered by
        # structured per-row chunks to avoid noisy duplicate retrieval.
        if dataset.raw_sections:
            for sec_name, sec_text in dataset.raw_sections.items():
                if sec_name not in _RAW_SECTION_SKIP:
                    chunks.extend(self._chunk_raw_section(sec_name, sec_text))

        logger.info(
            "PlacementChunker: %d total chunks (%d eligibility, %d hiring, "
            "%d trends, %d conflicts, %d stats, %d interviews after dedup)",
            len(chunks),
            len(dataset.eligibility_profiles),
            len(dataset.hiring_distributions),
            len(dataset.placement_trends),
            len(dataset.conflict_records),
            len(dataset.overall_stats),
            len(interview_chunks),
        )
        return chunks

    def _chunk_raw_section(self, name: str, text: str) -> List[Chunk]:
        paragraphs = text.split("\n\n")
        chunks = []
        current_block = ""
        
        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            if len(self._encoder.encode(current_block + "\n\n" + p)) <= 300:
                current_block = (current_block + "\n\n" + p).strip()
            else:
                if current_block:
                    meta = {
                        "content_type": "raw_section",
                        "section_name": name,
                        "source_type": "text",
                    }
                    chunks.append(self._make_chunk(current_block, name, meta))
                current_block = p
                
        if current_block:
            meta = {
                "content_type": "raw_section",
                "section_name": name,
                "source_type": "text",
            }
            chunks.append(self._make_chunk(current_block, name, meta))
            
        return chunks

    def _make_chunk(
        self,
        text: str,
        section: str,
        metadata: Dict[str, Any],
    ) -> Chunk:
        chunk_id = f"{self._doc_id}_placement_{section}_{self._chunk_index}_{uuid.uuid4().hex[:8]}"
        self._chunk_index += 1

        full_meta = dict(self._base)
        full_meta["section"] = section
        full_meta["content_type"] = metadata.pop("content_type", "structured_row")
        full_meta.update(metadata)

        return Chunk(
            chunk_id=chunk_id,
            doc_id=self._doc_id,
            text=text,
            page_start=metadata.get("page"),
            page_end=metadata.get("page"),
            metadata=full_meta,
        )

    def _chunk_eligibility(self, profiles: List[EligibilityProfile]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for p in profiles:
            # Prefix: "Eligibility criteria for <Company>" — semantically anchors
            # this chunk to CGPA/backlog/bond/package eligibility queries.
            text = (
                f"Eligibility criteria for {p.company}: "
                f"minimum CGPA {p.min_cgpa}, maximum backlogs {p.max_backlogs}, "
                f"package {p.package_lpa} LPA, bond {p.bond_years} years. "
                f"Key topics: {p.key_topics}. Technical focus: {p.tech_focus}."
            )
            meta: Dict[str, Any] = {
                "content_type": "structured_row",
                "company": p.company,
                "min_cgpa": p.min_cgpa,
                "max_backlogs": p.max_backlogs,
                "package_lpa": p.package_lpa,
                "bond_years": p.bond_years,
                "key_topics": p.key_topics,
                "tech_focus": p.tech_focus,
                "source_type": p.source_type,
                "page": p.page_number,
            }
            chunks.append(self._make_chunk(text, "eligibility", meta))
        return chunks

    def _chunk_hiring(self, distributions: List[HiringDistribution]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for d in distributions:
            # Prefix: "Role-wise hiring distribution for <Company>" — semantically
            # distinct from eligibility. Mentions all four roles explicitly so a
            # single-chunk retrieval is sufficient to answer any role breakdown query.
            text = (
                f"Role-wise hiring distribution for {d.company}: "
                f"SDE={d.sde}, Analyst={d.analyst}, Officer={d.officer}, Intern={d.intern}. "
                f"Total hires: {d.total}."
            )
            meta: Dict[str, Any] = {
                "content_type": "structured_row",
                "company": d.company,
                "sde": d.sde,
                "analyst": d.analyst,
                "officer": d.officer,
                "intern": d.intern,
                "total": d.total,
                "source_type": d.source_type,
                "page": d.page_number,
            }
            chunks.append(self._make_chunk(text, "hiring", meta))
        return chunks

    def _chunk_trends(self, trends: List[PlacementTrend]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for t in trends:
            # Prefix: "Year-over-year package trend for <Company>" — semantically
            # anchors this chunk to temporal/trend queries, not eligibility.
            text = (
                f"Year-over-year package trend for {t.company}: "
                f"2021={t.package_2021} LPA, 2022={t.package_2022} LPA, "
                f"2023={t.package_2023} LPA, 2024={t.package_2024} LPA. "
                f"Absolute growth (2021→2024): {t.absolute_growth_2021_2024} LPA ({t.trend_label})."
            )
            meta: Dict[str, Any] = {
                "content_type": "structured_row",
                "company": t.company,
                "package_2021": t.package_2021,
                "package_2022": t.package_2022,
                "package_2023": t.package_2023,
                "package_2024": t.package_2024,
                "absolute_growth": t.absolute_growth_2021_2024,
                "trend_label": t.trend_label,
                "metric": "package_trend",
                "page": t.page_number,
            }
            chunks.append(self._make_chunk(text, "trend", meta))
        return chunks

    def _chunk_conflicts(self, conflicts: List[ConflictRecord]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for c in conflicts:
            # Prefix: "Conflicting placement records for <Company>" — semantically
            # anchors this to hallucination-detection / discrepancy queries.
            text = (
                f"Conflicting placement records for {c.company}: "
                f"official CGPA cutoff {c.official_cgpa}, official package {c.official_package_lpa} LPA; "
                f"portal CGPA {c.portal_cgpa}, portal package {c.portal_package_lpa} LPA. "
                f"CGPA conflict: {c.cgpa_conflict}. Package conflict: {c.package_conflict}."
            )
            meta: Dict[str, Any] = {
                "content_type": "structured_row",
                "company": c.company,
                "conflict": True,
                "cgpa_conflict": c.cgpa_conflict,
                "package_conflict": c.package_conflict,
                "official_cgpa": c.official_cgpa,
                "portal_cgpa": c.portal_cgpa,
                "official_package_lpa": c.official_package_lpa,
                "portal_package_lpa": c.portal_package_lpa,
                "source_type": "derived",
                "page": c.page_number,
            }
            chunks.append(self._make_chunk(text, "conflict", meta))
        return chunks

    def _chunk_stats(self, stats_list: List[OverallStats]) -> List[Chunk]:
        if not stats_list:
            return []

        rows = []
        for s in stats_list:
            bond = "Yes" if s.bond_free else "No"
            rows.append(
                f"Company: {s.company}. Average package: {s.avg_package} LPA. "
                f"Max offers: {s.max_offers}. Min offers: {s.min_offers}. "
                f"Average CGPA cutoff: {s.avg_cgpa_cutoff}. Bond free: {bond}."
            )
        full_text = "Overall placement statistics:\n" + "\n".join(rows)

        meta: Dict[str, Any] = {
            "content_type": "full_table",
            "source_type": "table",
        }
        return [self._make_chunk(full_text, "statistics", meta)]

    def _chunk_interviews(self, experiences: List[InterviewExperience]) -> List[Chunk]:
        if not experiences:
            return []

        chunks: List[Chunk] = []
        current_text = ""
        current_company = ""
        current_rounds: List[int] = []

        for exp in experiences:
            round_text = (
                f"Company: {exp.company}. Round {exp.round_number}: {exp.round_title}. "
                f"Technical focus: {exp.technical_focus}. Details: {exp.details}. "
            )
            if exp.tip:
                round_text += f"Tip: {exp.tip}. "

            if current_company and exp.company != current_company:
                if current_text.strip():
                    chunks.extend(
                        self._split_interview_chunk(current_text.strip(), current_company, current_rounds)
                    )
                current_text = ""
                current_rounds = []

            current_company = exp.company
            current_rounds.append(exp.round_number)
            current_text += round_text + "\n"

        if current_text.strip():
            chunks.extend(
                self._split_interview_chunk(current_text.strip(), current_company, current_rounds)
            )

        return chunks

    def _split_interview_chunk(
        self, text: str, company: str, round_numbers: List[int]
    ) -> List[Chunk]:
        tokens = self._encoder.encode(text, disallowed_special=())
        if len(tokens) <= INTERVIEW_MAX_TOKENS:
            meta: Dict[str, Any] = {
                "content_type": "interview_round",
                "company": company,
                "round_numbers": round_numbers,
                "source_type": "narrative",
            }
            return [self._make_chunk(text, "interview", meta)]

        chunks: List[Chunk] = []
        start = 0
        part_idx = 0
        while start < len(tokens):
            end = min(start + INTERVIEW_MAX_TOKENS, len(tokens))
            part_text = self._encoder.decode(tokens[start:end]).strip()
            if part_text:
                meta: Dict[str, Any] = {
                    "content_type": "interview_round",
                    "company": company,
                    "round_numbers": round_numbers,
                    "source_type": "narrative",
                    "part": part_idx,
                }
                chunks.append(self._make_chunk(part_text, "interview", meta))
                part_idx += 1
            start = end
        return chunks

    def _deduplicate_interviews(self, chunks: List[Chunk]) -> List[Chunk]:
        if not chunks:
            return []

        normalized: List[Tuple[Chunk, str, str]] = []
        for c in chunks:
            norm_text = self._normalize_text(c.text)
            text_hash = hashlib.sha256(norm_text.encode("utf-8")).hexdigest()
            normalized.append((c, norm_text, text_hash))

        seen_hash: Dict[str, List[Tuple[Chunk, str]]] = {}
        for c, norm_text, text_hash in normalized:
            seen_hash.setdefault(text_hash, []).append((c, norm_text))

        deduped: List[Chunk] = []
        for text_hash, group in seen_hash.items():
            if len(group) == 1:
                c, _ = group[0]
                c.metadata["dedupe_key"] = text_hash
                c.metadata["dedupe_count"] = 1
                deduped.append(c)
                continue

            clusters = self._cluster_near_duplicates([n for _, n in group])

            for cluster in clusters:
                canonical_idx = 0
                count = len(cluster)
                c, _ = group[cluster[canonical_idx]]
                c.metadata["dedupe_key"] = text_hash
                c.metadata["dedupe_count"] = count
                deduped.append(c)

        removed = len(chunks) - len(deduped)
        if removed > 0:
            logger.info(
                "Interview deduplication: %d -> %d chunks (%d removed)",
                len(chunks), len(deduped), removed,
            )
        return deduped

    def _normalize_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[•·\-–—\*]+", " ", text)
        text = re.sub(r"[^\w\s]", "", text)
        return text.strip()

    def _cluster_near_duplicates(self, texts: List[str]) -> List[List[int]]:
        if not HAS_RAPIDFUZZ or len(texts) <= 1:
            return [[i] for i in range(len(texts))]

        n = len(texts)
        assigned: Set[int] = set()
        clusters: List[List[int]] = []

        for i in range(n):
            if i in assigned:
                continue
            cluster = [i]
            assigned.add(i)
            for j in range(i + 1, n):
                if j in assigned:
                    continue
                ratio = fuzz.ratio(texts[i], texts[j])
                if ratio / 100.0 >= self._dedupe_threshold:
                    cluster.append(j)
                    assigned.add(j)
            clusters.append(cluster)

        return clusters
