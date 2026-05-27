from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.document_compressors import FlashrankReranker
from langchain_google_genai import ChatGoogleGenerativeAI

from config import DEFAULT_ALPHA, DEFAULT_TOP_K, GEMINI_MODEL, MULTI_QUERY_COUNT, RERANK_MODEL
from retrieval.hybrid import hybrid_search


class HybridSearchRetriever(BaseRetriever):
    top_k: int = DEFAULT_TOP_K
    alpha: float = DEFAULT_ALPHA

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        results = hybrid_search(query, top_k=self.top_k, alpha=self.alpha)
        return [
            Document(
                page_content=r["document"],
                metadata=r.get("metadata") or {},
            )
            for r in results
        ]


_llm_cache = {}


def _get_llm() -> ChatGoogleGenerativeAI:
    if "llm" not in _llm_cache:
        _llm_cache["llm"] = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            convert_system_message_to_human=True,
        )
    return _llm_cache["llm"]


_retriever_cache = {}


def build_retriever(
    top_k: int = DEFAULT_TOP_K,
    alpha: float = DEFAULT_ALPHA,
) -> ContextualCompressionRetriever:
    key = (top_k, alpha)
    if key in _retriever_cache:
        return _retriever_cache[key]

    base = HybridSearchRetriever(top_k=top_k, alpha=alpha)
    llm = _get_llm()

    multi = MultiQueryRetriever.from_llm(
        retriever=base,
        llm=llm,
    )

    compressor = FlashrankReranker(model_name=RERANK_MODEL)
    compression = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=multi,
    )

    _retriever_cache[key] = compression
    return compression