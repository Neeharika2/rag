from typing import Any, Dict, Iterable, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from chunking.recursive import Chunk


class QdrantVectorStore:
    def __init__(
        self,
        url: str,
        collection_name: str,
        vector_size: int,
    ) -> None:
        self._client = QdrantClient(url=url)
        self._collection_name = collection_name
        self._vector_size = vector_size
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        try:
            self._client.get_collection(self._collection_name)
            return
        except Exception:
            pass

        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=qdrant_models.VectorParams(
                size=self._vector_size,
                distance=qdrant_models.Distance.COSINE,
            ),
        )

    def upsert(self, embeddings: List[List[float]], chunks: List[Chunk]) -> None:
        points: List[qdrant_models.PointStruct] = []
        for embedding, chunk in zip(embeddings, chunks):
            payload = dict(chunk.metadata)
            payload.update(
                {
                    "doc_id": chunk.doc_id,
                    "chunk_id": chunk.chunk_id,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "text": chunk.text,
                }
            )
            points.append(
                qdrant_models.PointStruct(
                    id=chunk.chunk_id,
                    vector=embedding,
                    payload=payload,
                )
            )

        if points:
            self._client.upsert(
                collection_name=self._collection_name,
                points=points,
            )

    def search(
        self,
        query_vector: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        filter_query = self._build_filter(filters)
        results = self._client.search(
            collection_name=self._collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=filter_query,
        )
        hits = []
        for hit in results:
            hits.append(
                {
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload or {},
                }
            )
        return hits

    def _build_filter(
        self, filters: Optional[Dict[str, Any]]
    ) -> Optional[qdrant_models.Filter]:
        if not filters:
            return None

        conditions: List[qdrant_models.FieldCondition] = []
        for key, value in filters.items():
            if isinstance(value, list):
                match = qdrant_models.MatchAny(any=value)
            else:
                match = qdrant_models.MatchValue(value=value)
            conditions.append(qdrant_models.FieldCondition(key=key, match=match))

        if not conditions:
            return None

        return qdrant_models.Filter(must=conditions)
