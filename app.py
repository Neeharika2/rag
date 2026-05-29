import os
import shutil
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from chunking.recursive import RecursiveChunker
from embeddings.local import LocalEmbeddingProvider
from evaluation.query_logger import QueryLogger
from generation.answerer import Answerer
from generation.gemini import GeminiGenerator
from ingestion.metadata_store import MetadataStore
from ingestion.pipeline import IngestionPipeline
from parsing.multimodal_parser import MultiModalParser
from retrieval.retriever import Retriever
from settings import Settings
from vectorstore.chroma_store import ChromaVectorStore

settings = Settings.from_env()
settings.ensure_dirs()

metadata_store = MetadataStore(settings.metadata_db_url)
metadata_store.init_db()

embedding_provider = LocalEmbeddingProvider(settings.embedding_model)
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

query_logger = QueryLogger(metadata_store)
retriever = Retriever(
    embedding_provider=embedding_provider,
    vector_store=vector_store,
    query_logger=query_logger,
)
generator = GeminiGenerator(
    api_key=settings.gemini_api_key,
    model_name=settings.gemini_model,
)
answerer = Answerer(retriever=retriever, generator=generator)

pipeline = IngestionPipeline(
    parser=parser,
    chunker=chunker,
    embedding_provider=embedding_provider,
    vector_store=vector_store,
    metadata_store=metadata_store,
    log_dir=settings.log_dir,
)

app = FastAPI(title="AI Core RAG Foundation")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
        return """
        <!doctype html>
        <html lang="en">
        <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <title>RAG Q&A</title>
            <style>
                :root { color-scheme: light; }
                body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Arial; margin: 2rem; }
                .container { max-width: 760px; margin: 0 auto; }
                textarea { width: 100%; min-height: 120px; }
                input, textarea, button { font-size: 1rem; padding: 0.5rem; }
                .row { display: flex; gap: 0.75rem; align-items: center; }
                .row input { width: 6rem; }
                .card { border: 1px solid #ddd; border-radius: 8px; padding: 1rem; margin-top: 1rem; }
                .muted { color: #555; }
                pre { white-space: pre-wrap; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Ask a question</h1>
                <p class="muted">Sends your question to the /answer endpoint and shows citations.</p>
                <label for="query">Question</label>
                <textarea id="query" placeholder="Ask something about your documents..."></textarea>
                <div class="row" style="margin-top: 0.75rem;">
                    <label for="top_k">Top K</label>
                    <input id="top_k" type="number" min="1" max="50" value="5" />
                    <button id="ask">Ask</button>
                </div>
                <div id="error" class="card" style="display:none; border-color: #f99;"></div>
                <div id="answer" class="card" style="display:none;"></div>
                <div id="citations" class="card" style="display:none;"></div>
            </div>
            <script>
                const askBtn = document.getElementById("ask");
                const queryEl = document.getElementById("query");
                const topKEl = document.getElementById("top_k");
                const answerEl = document.getElementById("answer");
                const citationsEl = document.getElementById("citations");
                const errorEl = document.getElementById("error");

                function show(el, html) {
                    el.style.display = "block";
                    el.innerHTML = html;
                }

                function hide(el) {
                    el.style.display = "none";
                    el.innerHTML = "";
                }

                askBtn.addEventListener("click", async () => {
                    hide(errorEl);
                    hide(answerEl);
                    hide(citationsEl);

                    const query = queryEl.value.trim();
                    const topK = Number(topKEl.value || 5);
                    if (!query) {
                        show(errorEl, "Please enter a question.");
                        return;
                    }

                    askBtn.disabled = true;
                    askBtn.textContent = "Asking...";

                    try {
                        const response = await fetch("/answer", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ query, top_k: topK, filters: null })
                        });

                        if (!response.ok) {
                            const text = await response.text();
                            show(errorEl, `Request failed: ${response.status} ${text}`);
                            return;
                        }

                        const data = await response.json();
                        show(answerEl, `<h3>Answer</h3><pre>${data.answer || ""}</pre>`);

                        const citations = (data.citations || []).map((c, idx) => {
                            const header = `[${idx + 1}] ${c.doc_id || ""} / ${c.chunk_id || ""}`;
                            return `<div style="margin-bottom: 0.75rem;"><div><strong>${header}</strong></div><div class="muted">Score: ${c.score}</div><pre>${c.text || ""}</pre></div>`;
                        }).join("");

                        show(citationsEl, `<h3>Citations</h3>${citations || "No citations."}`);
                    } catch (err) {
                        show(errorEl, `Error: ${err}`);
                    } finally {
                        askBtn.disabled = false;
                        askBtn.textContent = "Ask";
                    }
                });
            </script>
        </body>
        </html>
        """


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


class AnswerRequest(BaseModel):
    query: str
    top_k: int = Field(default=settings.top_k, ge=1, le=50)
    filters: Optional[Dict[str, Any]] = None


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    score: float
    text: str


class AnswerResponse(BaseModel):
    answer: str
    citations: List[Citation]


class IngestResponse(BaseModel):
    doc_id: str
    chunks_ingested: int
    source: str


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)) -> IngestResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    ext = os.path.splitext(file.filename)[1].lower()
    allowed_extensions = {
        ".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff",
        ".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".mp4"
    }
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Unsupported file type")

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


@app.post("/answer", response_model=AnswerResponse)
def answer(request: AnswerRequest) -> AnswerResponse:
    answer_text, hits = answerer.answer(
        query=request.query,
        top_k=request.top_k,
        filters=request.filters,
    )
    citations: List[Citation] = []
    for hit in hits:
        citations.append(
            Citation(
                chunk_id=hit.get("chunk_id", hit.get("id", "")),
                doc_id=hit.get("doc_id", ""),
                score=hit.get("score", 0.0),
                text=hit.get("text", ""),
            )
        )
    return AnswerResponse(answer=answer_text, citations=citations)
