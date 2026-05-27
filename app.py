import os
import shutil
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from chunking.recursive import RecursiveChunker
from embeddings.local import LocalEmbeddingProvider
from evaluation.query_logger import QueryLogger
from ingestion.metadata_store import MetadataStore
from ingestion.pipeline import IngestionPipeline
from parsing.docling_parser import DoclingParser
from retrieval.retriever import Retriever
from settings import Settings
from vectorstore.qdrant_store import QdrantVectorStore

settings = Settings.from_env()
settings.ensure_dirs()

metadata_store = MetadataStore(settings.metadata_db_url)
metadata_store.init_db()

embedding_provider = LocalEmbeddingProvider(settings.embedding_model)
vector_store = QdrantVectorStore(
    url=settings.qdrant_url,
    collection_name=settings.qdrant_collection,
    vector_size=embedding_provider.dimension,
)

parser = DoclingParser()
chunker = RecursiveChunker(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
)

query_logger = QueryLogger(metadata_store)
retriever = Retriever(
    embedding_provider=embedding_provider,
    vector_store=vector_store,
    query_logger=query_logger,
)

pipeline = IngestionPipeline(
    parser=parser,
    chunker=chunker,
    embedding_provider=embedding_provider,
    vector_store=vector_store,
    metadata_store=metadata_store,
    log_dir=settings.log_dir,
)

app = FastAPI(title="AI Core RAG Foundation")


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = Field(default=settings.top_k, ge=1, le=50)
    filters: Optional[Dict[str, Any]] = None


class RetrievalHit(BaseModel):
    chunk_id: str
    doc_id: str
    score: float
    text: str
    metadata: Dict[str, Any]


class RetrieveResponse(BaseModel):
    results: List[RetrievalHit]


class IngestResponse(BaseModel):
    doc_id: str
    chunks_ingested: int
    source: str


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)) -> IngestResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    safe_name = os.path.basename(file.filename)
    target_path = os.path.join(settings.upload_dir, safe_name)

    with open(target_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = pipeline.ingest_file(target_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return IngestResponse(
        doc_id=result["doc_id"],
        chunks_ingested=result["chunks_ingested"],
        source=result["source"],
    )


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(request: RetrieveRequest) -> RetrieveResponse:
    results = retriever.retrieve(
        query=request.query,
        top_k=request.top_k,
        filters=request.filters,
    )
    return RetrieveResponse(results=results)
