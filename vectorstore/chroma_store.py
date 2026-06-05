import json
import logging
import uuid
from typing import Any, Dict, List, Optional

import chromadb

from chunking.recursive import Chunk

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    _chunk_namespace = uuid.UUID("1a8d1b2e-6a7f-4f8b-9d53-0f6c1f4e2a3b")

    def __init__(self, persist_dir: str, collection_name: str) -> None:
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _normalize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        normalized = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                normalized[key] = value
            elif isinstance(value, (dict, list, tuple, set)):
                normalized[key] = json.dumps(value)
            else:
                normalized[key] = str(value)
        return normalized

    def _denormalize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        denormalized = {}
        for key, value in metadata.items():
            if isinstance(value, str):
                trimmed = value.strip()
                if (trimmed.startswith("{") and trimmed.endswith("}")) or (trimmed.startswith("[") and trimmed.endswith("]")):
                    try:
                        denormalized[key] = json.loads(value)
                        continue
                    except Exception:
                        pass
            denormalized[key] = value
        return denormalized

    def upsert(self, embeddings: List[List[float]], chunks: List[Chunk]) -> None:
        ids: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        documents: List[str] = []

        for embedding, chunk in zip(embeddings, chunks):
            chunk_id = str(uuid.uuid5(self._chunk_namespace, chunk.chunk_id))
            metadata = dict(chunk.metadata)
            metadata.update(
                {
                    "doc_id": chunk.doc_id,
                    "chunk_id": chunk.chunk_id,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                }
            )
            metadata = self._normalize_metadata(metadata)
            ids.append(chunk_id)
            metadatas.append(metadata)
            documents.append(chunk.text)

        if not ids:
            return

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

    def search(
        self,
        query_vector: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        where = self._build_filter(filters)
        result = self._collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where,
            include=["metadatas", "documents", "distances"],
        )
        return self._format_hits(result)

    def _build_filter(self, filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not filters:
            return None

        conditions = []
        for key, value in filters.items():
            if isinstance(value, list):
                conditions.append({key: {"$in": value}})
            else:
                conditions.append({key: value})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def delete_by_doc_id(self, doc_id: str) -> None:
        results = self._collection.get(where={"doc_id": doc_id})
        ids = results.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)
            logger.info("Deleted %d chunks for doc_id=%s", len(ids), doc_id)

    def reset_collection(self) -> None:
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection.name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Reset collection: %s", self._collection.name)

    def _format_hits(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        ids = (result.get("ids") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        hits: List[Dict[str, Any]] = []
        for idx, hit_id in enumerate(ids):
            metadata = metadatas[idx] or {}
            metadata = self._denormalize_metadata(metadata)
            document = documents[idx] if idx < len(documents) else ""
            distance = distances[idx] if idx < len(distances) else None
            score = 1.0 - float(distance) if distance is not None else 0.0

            payload = dict(metadata)
            if document:
                payload["text"] = document

            hits.append(
                {
                    "id": hit_id,
                    "score": score,
                    "payload": payload,
                }
            )

        return hits
