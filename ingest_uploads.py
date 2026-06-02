import os
from typing import Dict, List

from chunking.recursive import RecursiveChunker
from embeddings.gemini import GeminiEmbeddingProvider
from ingestion.metadata_store import MetadataStore
from ingestion.pipeline import IngestionPipeline
from parsing.multimodal_parser import MultiModalParser
from settings import Settings
from vectorstore.chroma_store import ChromaVectorStore


def _list_assets(upload_dir: str) -> List[str]:
    allowed = {
        ".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff",
        ".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".mp4"
    }
    paths: List[str] = []
    for name in os.listdir(upload_dir):
        if os.path.splitext(name)[1].lower() in allowed:
            paths.append(os.path.join(upload_dir, name))
    return sorted(paths)


def main() -> None:
    settings = Settings.from_env()
    settings.ensure_dirs()

    metadata_store = MetadataStore(settings.metadata_db_url)
    metadata_store.init_db()

    embedding_provider = GeminiEmbeddingProvider(
        api_key=settings.gemini_api_key,
        model_name=settings.embedding_model,
        dimension=3072,
    )
    vector_store = ChromaVectorStore(
        persist_dir=settings.chroma_path,
        collection_name=settings.chroma_collection,
    )

    parser = MultiModalParser(
        ocr_enabled=settings.ocr_enabled,
        tesseract_cmd=settings.tesseract_cmd,
    )
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

    assets = _list_assets(settings.upload_dir)
    if not assets:
        print(f"No supported files found in {settings.upload_dir}")
        return

    summary: Dict[str, int] = {"ingested": 0, "skipped": 0}
    for asset_path in assets:
        result = pipeline.ingest_file(asset_path, skip_if_exists=True)
        if result.get("skipped"):
            summary["skipped"] += 1
            print(f"Skipped: {os.path.basename(asset_path)}")
        else:
            summary["ingested"] += 1
            print(
                f"Ingested: {os.path.basename(asset_path)} "
                f"({result['chunks_ingested']} chunks)"
            )

    print(
        f"Done. Ingested {summary['ingested']} files, "
        f"skipped {summary['skipped']} files."
    )


if __name__ == "__main__":
    main()
