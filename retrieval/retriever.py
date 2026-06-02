from typing import Any, Dict, List, Optional

from embeddings.base import EmbeddingProvider
from evaluation.query_logger import QueryLogger
from vectorstore.chroma_store import ChromaVectorStore


class Retriever:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: ChromaVectorStore,
        query_logger: QueryLogger,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._query_logger = query_logger

    def retrieve(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        original_query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query_vector = self._embedding_provider.embed_query(query)
        hits = self._vector_store.search(query_vector, top_k, filters)

        log_query_text = original_query if original_query is not None else query
        log_filters = filters.copy() if filters else {}
        if original_query is not None and original_query != query:
            log_filters["rewritten_query"] = query

        query_id = self._query_logger.log_query(log_query_text, log_filters, top_k)
        self._query_logger.log_hits(query_id, hits)

        results = []
        for hit in hits:
            payload = hit.get("payload", {})
            results.append(
                {
                    "chunk_id": payload.get("chunk_id", hit["id"]),
                    "doc_id": payload.get("doc_id"),
                    "score": hit["score"],
                    "text": payload.get("text", ""),
                    "metadata": payload,
                }
            )

        return results
