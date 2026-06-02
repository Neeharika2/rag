from typing import Any, Dict, List, Optional, Tuple

from retrieval.retriever import Retriever
from generation.base import AnswerGenerator
from agents.query_rewriter import QueryRewriter


class Answerer:
    def __init__(
        self,
        retriever: Retriever,
        generator: AnswerGenerator,
        query_rewriter: Optional[QueryRewriter] = None,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._query_rewriter = query_rewriter

    def answer(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        rewrite: bool = False,
    ) -> Tuple[str, List[Dict[str, Any]], Optional[str]]:
        rewritten_query = None
        search_query = query
        if rewrite and self._query_rewriter:
            rewritten_query = self._query_rewriter.rewrite(query)
            if rewritten_query != query:
                search_query = rewritten_query
            else:
                rewritten_query = None

        hits = self._retriever.retrieve(
            query=search_query,
            top_k=top_k,
            filters=filters,
            original_query=query if rewrite else None,
        )
        prompt = self._build_prompt(query, hits)
        answer = self._generator.generate(prompt)
        return answer, hits, rewritten_query

    def _build_prompt(self, query: str, hits: List[Dict[str, Any]]) -> str:
        if hits:
            sources = []
            for idx, hit in enumerate(hits, start=1):
                source_text = hit.get("text", "")
                chunk_id = hit.get("chunk_id", hit.get("id", ""))
                doc_id = hit.get("doc_id", "")
                sources.append(
                    f"[{idx}] doc_id={doc_id} chunk_id={chunk_id}\n{source_text}"
                )
            sources_block = "\n\n".join(sources)
        else:
            sources_block = "(no sources retrieved)"

        return (
            "You are a helpful assistant. Use only the sources below to answer. "
            "Cite sources inline using [n]. If sources are insufficient, say you don't know.\n\n"
            f"Question: {query}\n\n"
            f"Sources:\n{sources_block}\n\n"
            "Answer:"
        )
