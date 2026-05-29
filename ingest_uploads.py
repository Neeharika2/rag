import os
from typing import Dict, List

from chunking.recursive import RecursiveChunker
from embeddings.local import LocalEmbeddingProvider
from ingestion.metadata_store import MetadataStore
from ingestion.pipeline import IngestionPipeline
from parsing.docling_parser import DoclingParser
from settings import Settings
from vectorstore.chroma_store import ChromaVectorStore


def _list_pdfs(upload_dir: str) -> List[str]:
    paths: List[str] = []
    for name in os.listdir(upload_dir):
        if name.lower().endswith(".pdf"):
            paths.append(os.path.join(upload_dir, name))
    return sorted(paths)


def main() -> None:
    settings = Settings.from_env()
    settings.ensure_dirs()

    metadata_store = MetadataStore(settings.metadata_db_url)
    metadata_store.init_db()

    embedding_provider = LocalEmbeddingProvider(settings.embedding_model)
    vector_store = ChromaVectorStore(
        persist_dir=settings.chroma_path,
        collection_name=settings.chroma_collection,
    )

    parser = DoclingParser()
    chunker = RecursiveChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    pipeline = IngestionPipeline(
        parser=parser,
        chunker=chunker,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        metadata_store=metadata_store,
        log_dir=settings.log_dir,
    )

    pdfs = _list_pdfs(settings.upload_dir)
    if not pdfs:
        print(f"No PDFs found in {settings.upload_dir}")
        return

    summary: Dict[str, int] = {"ingested": 0, "skipped": 0}
    for pdf_path in pdfs:
        result = pipeline.ingest_file(pdf_path, skip_if_exists=True)
        if result.get("skipped"):
            summary["skipped"] += 1
            print(f"Skipped: {os.path.basename(pdf_path)}")
        else:
            summary["ingested"] += 1
            print(
                f"Ingested: {os.path.basename(pdf_path)} "
                f"({result['chunks_ingested']} chunks)"
            )

    print(
        f"Done. Ingested {summary['ingested']} files, "
        f"skipped {summary['skipped']} files."
    )


if __name__ == "__main__":
    main()
