from typing import Any, Dict, List, Optional

from ingestion.metadata_store import MetadataStore


class QueryLogger:
    def __init__(self, metadata_store: MetadataStore) -> None:
        self._metadata_store = metadata_store

    def log_query(
        self,
        query_text: str,
        filters: Optional[Dict[str, Any]],
        top_k: int,
    ) -> str:
        return self._metadata_store.log_query(
            query_text=query_text,
            filters=filters,
            top_k=top_k,
        )

    def log_hits(self, query_id: str, hits: List[Dict[str, Any]]) -> None:
        self._metadata_store.log_hits(query_id, hits)
