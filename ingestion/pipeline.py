import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from chunking.recursive import RecursiveChunker
from embeddings.base import EmbeddingProvider
from ingestion.metadata_store import MetadataStore
from parsing.base import DocumentParser
from parsing.structured_log import ParseEvent, track_parse
from vectorstore.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class IngestionPipeline:
    def __init__(
        self,
        parser: DocumentParser,
        chunker: RecursiveChunker,
        embedding_provider: EmbeddingProvider,
        vector_store: ChromaVectorStore,
        metadata_store: MetadataStore,
        log_dir: Optional[str] = None,
    ) -> None:
        self._parser = parser
        self._chunker = chunker
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._metadata_store = metadata_store
        self._log_dir = log_dir

    def ingest_file(
        self,
        file_path: str,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        skip_if_exists: bool = True,
    ) -> Dict[str, Any]:
        doc_id = doc_id or os.path.splitext(os.path.basename(file_path))[0]
        metadata = metadata or {}

        if skip_if_exists and self._metadata_store.has_document(doc_id):
            logger.info("Skipping %s (already exists)", doc_id)
            return {
                "doc_id": doc_id,
                "chunks_ingested": 0,
                "source": os.path.basename(file_path),
                "skipped": True,
            }

        base_metadata = {
            "doc_id": doc_id,
            "source": os.path.basename(file_path),
            "uploaded_at": metadata.get("uploaded_at", utc_now()),
            "access_level": metadata.get("access_level", "internal"),
        }
        base_metadata.update(metadata)

        with track_parse(doc_id, file_path, "ingestion_pipeline"):
            parsed = self._parser.parse(file_path, doc_id)
            chunks = self._chunker.chunk_pages(doc_id, parsed.pages, base_metadata)

        if not chunks:
            self._metadata_store.upsert_document(
                doc_id=doc_id,
                source=os.path.basename(file_path),
                access_level=base_metadata["access_level"],
                metadata=base_metadata,
            )
            self._log_ingestion(doc_id, parsed.raw_markdown, chunks)
            logger.warning("No chunks generated for %s", doc_id)
            return {
                "doc_id": doc_id,
                "chunks_ingested": 0,
                "source": os.path.basename(file_path),
            }

        embeddings = self._embedding_provider.embed_texts(
            [chunk.text for chunk in chunks]
        )

        self._vector_store.upsert(embeddings, chunks)
        self._metadata_store.upsert_document(
            doc_id=doc_id,
            source=os.path.basename(file_path),
            access_level=base_metadata["access_level"],
            metadata=base_metadata,
        )
        self._metadata_store.upsert_chunks(chunks)

        self._log_ingestion(doc_id, parsed.raw_markdown, chunks)

        logger.info(
            "Ingested %s: %d chunks, %d pages",
            doc_id, len(chunks), len(parsed.pages),
        )

        return {
            "doc_id": doc_id,
            "chunks_ingested": len(chunks),
            "source": os.path.basename(file_path),
        }

    def _log_ingestion(self, doc_id: str, markdown: str, chunks) -> None:
        if not self._log_dir:
            return

        parsed_dir = os.path.join(self._log_dir, "parsed")
        chunk_dir = os.path.join(self._log_dir, "chunks")
        os.makedirs(parsed_dir, exist_ok=True)
        os.makedirs(chunk_dir, exist_ok=True)

        parsed_path = os.path.join(parsed_dir, f"{doc_id}.md")
        with open(parsed_path, "w", encoding="utf-8") as f:
            f.write(markdown)

        chunk_path = os.path.join(chunk_dir, f"{doc_id}.jsonl")
        with open(chunk_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                record = {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                }
                f.write(json.dumps(record) + "\n")
