import logging
from typing import Any, Dict, List, Optional

from agents.query_rewriter import QueryRewriter
from generation.answer_result import AnswerResult, Citation
from generation.base import AnswerGenerator
from placement.fallback import get_fallback_message
from placement.models import PlacementDataset
from placement.query_router import (
    ROUTE_CONFLICT,
    ROUTE_GENERIC,
    ROUTE_INTERVIEW,
    ROUTE_OUT_OF_CORPUS,
    ROUTE_STRUCTURED,
    ROUTE_TREND,
    route_query,
)
from placement.reasoner import StructuredReasoner
from retrieval.retriever import Retriever

logger = logging.getLogger(__name__)


_ROUTE_TO_SECTION = {
    ROUTE_INTERVIEW: "interview",
    ROUTE_TREND: "trend",
    ROUTE_CONFLICT: "conflict",
    ROUTE_STRUCTURED: "eligibility",
}


class Answerer:
    def __init__(
        self,
        retriever: Retriever,
        generator: AnswerGenerator,
        metadata_store: Optional[Any] = None,
        query_rewriter: Optional[QueryRewriter] = None,
        narrative_top_k: int = 8,
        generic_top_k: int = 5,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._metadata_store = metadata_store
        self._query_rewriter = query_rewriter
        self._narrative_top_k = narrative_top_k
        self._generic_top_k = generic_top_k

    def answer(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        rewrite: bool = False,
    ) -> AnswerResult:
        rewritten_query = self._maybe_rewrite(query, rewrite)

        routed = route_query(query)

        if routed.route == ROUTE_OUT_OF_CORPUS:
            return self._handle_out_of_corpus(query, routed, rewritten_query)

        dataset = self._load_dataset()
        if routed.route in (ROUTE_STRUCTURED, ROUTE_TREND, ROUTE_CONFLICT) and dataset is not None:
            return self._handle_structured(query, routed, rewritten_query, dataset)

        if routed.route == ROUTE_INTERVIEW:
            return self._handle_narrative(
                query, routed, rewritten_query,
                top_k=top_k or self._narrative_top_k,
                section=_ROUTE_TO_SECTION[routed.route],
                explicit_filters=filters,
            )

        return self._handle_generic(
            query, routed, rewritten_query,
            top_k=top_k or self._generic_top_k,
            explicit_filters=filters,
        )

    def _maybe_rewrite(self, query: str, rewrite: bool) -> Optional[str]:
        if not rewrite or self._query_rewriter is None:
            return None
        try:
            rewritten = self._query_rewriter.rewrite(query)
        except Exception as exc:
            logger.warning("Query rewrite failed: %s", exc)
            return None
        if rewritten and rewritten != query:
            return rewritten
        return None

    def _load_dataset(self) -> Optional[PlacementDataset]:
        if self._metadata_store is None:
            return None
        try:
            return self._metadata_store.get_latest_placement_dataset()
        except Exception as exc:
            logger.warning("Failed to load placement dataset: %s", exc)
            return None

    def _detect_company_in_query(self, query: str) -> Optional[str]:
        from placement.query_router import COMPANIES
        q = query.lower()
        for c in sorted(COMPANIES, key=len, reverse=True):
            if c.lower() in q:
                return c
        return None

    def _handle_out_of_corpus(
        self,
        query: str,
        routed: Any,
        rewritten_query: Optional[str],
    ) -> AnswerResult:
        reason = routed.fallback_reason or "out_of_corpus"
        message = get_fallback_message(reason)
        detected_company = routed.detected_company or self._detect_company_in_query(query)
        return AnswerResult(
            answer=message,
            route=ROUTE_OUT_OF_CORPUS,
            confidence=routed.confidence or 0.95,
            citations=[],
            evidence=None,
            fallback_reason=reason,
            rewritten_query=rewritten_query,
            detected_company=detected_company,
            detected_metric=routed.detected_metric,
        )

    def _handle_structured(
        self,
        query: str,
        routed: Any,
        rewritten_query: Optional[str],
        dataset: PlacementDataset,
    ) -> AnswerResult:
        reasoner = StructuredReasoner(dataset)
        reasoned = reasoner.answer(query)
        return AnswerResult(
            answer=reasoned.answer,
            route=reasoned.route or routed.route,
            confidence=reasoned.confidence,
            citations=self._evidence_to_citations(reasoned.evidence, routed.detected_company),
            evidence=reasoned.evidence,
            fallback_reason=None,
            warning=reasoned.warning,
            rewritten_query=rewritten_query,
            detected_company=routed.detected_company,
            detected_metric=routed.detected_metric,
        )

    def _handle_narrative(
        self,
        query: str,
        routed: Any,
        rewritten_query: Optional[str],
        top_k: int,
        section: str,
        explicit_filters: Optional[Dict[str, Any]],
    ) -> AnswerResult:
        merged = self._merge_filters({"section": section}, routed, explicit_filters)
        search_query = rewritten_query or query
        hits = self._retrieve(search_query, top_k, merged, query)

        if not hits:
            return AnswerResult(
                answer=self._no_section_data_message(section, routed.detected_company),
                route=routed.route,
                confidence=0.4,
                citations=[],
                evidence=None,
                fallback_reason=f"no_{section}_data",
                rewritten_query=rewritten_query,
                detected_company=routed.detected_company,
                detected_metric=routed.detected_metric,
            )

        prompt = self._build_prompt(query, hits)
        try:
            answer = self._generator.generate(prompt)
        except Exception as exc:
            logger.error("Generation failed for %s: %s", routed.route, exc)
            return AnswerResult(
                answer=f"Failed to generate answer: {exc}",
                route=routed.route,
                confidence=0.0,
                citations=self._hits_to_citations(hits),
                rewritten_query=rewritten_query,
                detected_company=routed.detected_company,
                detected_metric=routed.detected_metric,
            )

        return AnswerResult(
            answer=answer,
            route=routed.route,
            confidence=min(0.9, max(0.5, hits[0]["score"])) if hits else 0.5,
            citations=self._hits_to_citations(hits),
            evidence=None,
            rewritten_query=rewritten_query,
            detected_company=routed.detected_company,
            detected_metric=routed.detected_metric,
        )

    def _handle_generic(
        self,
        query: str,
        routed: Any,
        rewritten_query: Optional[str],
        top_k: int,
        explicit_filters: Optional[Dict[str, Any]],
    ) -> AnswerResult:
        merged = self._merge_filters(None, routed, explicit_filters)
        search_query = rewritten_query or query
        hits = self._retrieve(search_query, top_k, merged, query)
        detected_company = routed.detected_company or self._detect_company_in_query(query)

        if not hits:
            return AnswerResult(
                answer=(
                    "I don't have enough information in the provided placement documents to answer that. "
                    "Please rephrase your question or ask about specific eligibility, hiring, package, or interview topics."
                ),
                route=ROUTE_GENERIC,
                confidence=0.0,
                citations=[],
                evidence=None,
                fallback_reason="no_relevant_chunks",
                rewritten_query=rewritten_query,
                detected_company=detected_company,
                detected_metric=routed.detected_metric,
            )

        prompt = self._build_prompt(query, hits)
        try:
            answer = self._generator.generate(prompt)
        except Exception as exc:
            logger.error("Generation failed for %s: %s", routed.route, exc)
            return AnswerResult(
                answer=f"Failed to generate answer: {exc}",
                route=ROUTE_GENERIC,
                confidence=0.0,
                citations=self._hits_to_citations(hits),
                rewritten_query=rewritten_query,
                detected_company=detected_company,
                detected_metric=routed.detected_metric,
            )

        confidence = min(0.9, max(0.5, hits[0]["score"])) if hits else 0.5
        return AnswerResult(
            answer=answer,
            route=ROUTE_GENERIC,
            confidence=confidence,
            citations=self._hits_to_citations(hits),
            evidence=None,
            rewritten_query=rewritten_query,
            detected_company=detected_company,
            detected_metric=routed.detected_metric,
        )

    def _merge_filters(
        self,
        section_filter: Optional[Dict[str, Any]],
        routed: Any,
        explicit_filters: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        merged: Dict[str, Any] = {}
        if section_filter:
            merged.update(section_filter)
        if explicit_filters:
            merged.update(explicit_filters)
        elif routed and routed.detected_company and "company" not in merged:
            merged["company"] = routed.detected_company
        return merged or None

    def _retrieve(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
        original_query: Optional[str],
    ) -> List[Dict[str, Any]]:
        try:
            return self._retriever.retrieve(
                query=query,
                top_k=top_k,
                filters=filters,
                original_query=original_query if original_query != query else None,
            )
        except Exception as exc:
            logger.error("Retrieval failed: %s", exc)
            return []

    def _build_prompt(self, query: str, hits: List[Dict[str, Any]]) -> str:
        if hits:
            sources = []
            for idx, hit in enumerate(hits, start=1):
                source_text = hit.get("text", "")
                chunk_id = hit.get("chunk_id", hit.get("id", ""))
                doc_id = hit.get("doc_id", "")
                section = hit.get("metadata", {}).get("section", "")
                company = hit.get("metadata", {}).get("company", "")
                source_label = f"[{idx}] doc_id={doc_id} chunk_id={chunk_id}"
                if section:
                    source_label += f" section={section}"
                if company:
                    source_label += f" company={company}"
                sources.append(f"{source_label}\n{source_text}")
            sources_block = "\n\n".join(sources)
        else:
            sources_block = "(no sources retrieved)"

        return (
            "You are a helpful assistant. Use only the sources below to answer. "
            "Cite sources inline using [n]. If the sources do not contain the answer, "
            "say you don't have that information. "
            "Do not invent dates, stock prices, work-mode, or institution-specific counts.\n\n"
            f"Question: {query}\n\n"
            f"Sources:\n{sources_block}\n\n"
            "Answer:"
        )

    def _hits_to_citations(self, hits: List[Dict[str, Any]]) -> List[Citation]:
        citations: List[Citation] = []
        for hit in hits:
            metadata = hit.get("metadata", {}) or {}
            citations.append(
                Citation(
                    chunk_id=hit.get("chunk_id", hit.get("id", "")),
                    doc_id=hit.get("doc_id", ""),
                    score=hit.get("score", 0.0),
                    text=hit.get("text", ""),
                    metadata={k: v for k, v in metadata.items() if k != "provenance"},
                )
            )
        return citations

    def _evidence_to_citations(
        self,
        evidence: Optional[List[Dict[str, Any]]],
        company: Optional[str],
    ) -> List[Citation]:
        if not evidence:
            return []
        citations: List[Citation] = []
        for idx, ev in enumerate(evidence, start=1):
            ev_company = ev.get("company") or company or "structured"
            citations.append(
                Citation(
                    chunk_id=f"structured_{idx}",
                    doc_id="placement_dataset",
                    score=1.0,
                    text=str(ev),
                    metadata={"section": "structured", "company": ev_company},
                )
            )
        return citations

    def _no_section_data_message(self, section: str, company: Optional[str]) -> str:
        target = f" for {company}" if company else ""
        return (
            f"No {section} data is available{target} in the provided placement documents. "
            f"Try a broader question or ask about a different company."
        )
