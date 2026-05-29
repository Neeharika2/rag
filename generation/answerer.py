from typing import Any, Dict, List, Optional, Tuple

from retrieval.retriever import Retriever
from generation.base import AnswerGenerator


class Answerer:
    def __init__(self, retriever: Retriever, generator: AnswerGenerator) -> None:
        self._retriever = retriever
        self._generator = generator

    def answer(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        hits = self._retriever.retrieve(query=query, top_k=top_k, filters=filters)
        prompt = self._build_prompt(query, hits)
        answer = self._generator.generate(prompt)
        return answer, hits

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
