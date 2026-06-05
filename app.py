import logging
import os
import shutil
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from agents.query_rewriter import QueryRewriter
from chunking.recursive import RecursiveChunker
from embeddings.gemini import GeminiEmbeddingProvider
from evaluation.query_logger import QueryLogger
from generation.answerer import Answerer
from generation.gemini import GeminiGenerator
from ingestion.metadata_store import MetadataStore
from ingestion.pipeline import IngestionPipeline
from parsing.multimodal_parser import MultiModalParser
from parsing.structured_log import configure_parse_logging, get_parse_events
from placement.models import PlacementDataset
from retrieval.retriever import Retriever
from settings import Settings
from vectorstore.chroma_store import ChromaVectorStore

settings = Settings.from_env()
settings.ensure_dirs()

configure_parse_logging(log_dir=os.path.join(settings.log_dir, "parsing"), level=settings.log_level)
logger = logging.getLogger(__name__)

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

query_logger = QueryLogger(metadata_store)
generator = GeminiGenerator(
    api_key=settings.gemini_api_key,
    model_name=settings.gemini_model,
)
query_rewriter = QueryRewriter(generator)

retriever = Retriever(
    embedding_provider=embedding_provider,
    vector_store=vector_store,
    query_logger=query_logger,
    min_score=settings.retrieval_min_score,
)
answerer = Answerer(
    retriever=retriever,
    generator=generator,
    metadata_store=metadata_store,
    query_rewriter=query_rewriter,
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


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    html_content = """
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
            .badge { background-color: #e2e8f0; color: #4a5568; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.875rem; margin-bottom: 0.75rem; display: inline-block; }
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
                <label style="display: flex; align-items: center; gap: 0.25rem; cursor: pointer; margin-left: 1rem;">
                    <input id="rewrite" type="checkbox" /> Rewrite Query (Gemini Agent)
                </label>
                <button id="ask" style="margin-left: auto;">Ask</button>
            </div>
            <div id="error" class="card" style="display:none; border-color: #f99;"></div>
            <div id="answer" class="card" style="display:none;"></div>
            <div id="citations" class="card" style="display:none;"></div>
        </div>
        <script>
            const askBtn = document.getElementById("ask");
            const queryEl = document.getElementById("query");
            const topKEl = document.getElementById("top_k");
            const rewriteEl = document.getElementById("rewrite");
            const answerEl = document.getElementById("answer");
            const citationsEl = document.getElementById("citations");
            const errorEl = document.getElementById("error");
            rewriteEl.checked = {default_rewrite};
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
                const rewrite = rewriteEl.checked;
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
                        body: JSON.stringify({ query, top_k: topK, filters: null, rewrite })
                    });
                    if (!response.ok) {
                        const text = await response.text();
                        show(errorEl, `Request failed: ${response.status} ${text}`);
                        return;
                    }
                    const data = await response.json();
                    const rewriteBadge = data.rewritten_query ? `<div class="badge">Rewritten to: "<em>${data.rewritten_query}</em>"</div>` : "";
                    show(answerEl, `<h3>Answer</h3>${rewriteBadge}<pre>${data.answer || ""}</pre>`);
                    const citations = (data.citations || []).map((c, idx) => {
                        const header = `[${idx + 1}] ${c.doc_id || ""} / ${c.chunk_id || ""}`;
                        let detail = `<div><strong>${header}</strong></div><div class="muted">Score: ${c.score}</div>`;
                        if (c.provenance) {
                            const p = c.provenance;
                            detail += `<div class="muted">Page: ${p.page_number || "?"}`;
                            if (p.bbox) detail += ` | BBox: (${p.bbox.left}, ${p.bbox.top}, ${p.bbox.right}, ${p.bbox.bottom})`;
                            if (p.tables && p.tables.length) detail += ` | Tables: ${p.tables.length}`;
                            if (p.images && p.images.length) detail += ` | Images: ${p.images.length}`;
                            detail += "</div>";
                        }
                        detail += `<pre>${c.text || ""}</pre>`;
                        return `<div style="margin-bottom: 0.75rem;">${detail}</div>`;
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
    return html_content.replace(
        "{default_rewrite}",
        "true" if settings.rewrite_query_by_default else "false",
    )


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = Field(default=settings.top_k, ge=1, le=50)
    filters: Optional[Dict[str, Any]] = None
    rewrite: Optional[bool] = None


class RetrievalHit(BaseModel):
    chunk_id: str
    doc_id: str
    score: float
    text: str
    metadata: Dict[str, Any]


class RetrieveResponse(BaseModel):
    results: List[RetrievalHit]
    rewritten_query: Optional[str] = None


class AnswerRequest(BaseModel):
    query: str
    top_k: int = Field(default=settings.top_k, ge=1, le=50)
    filters: Optional[Dict[str, Any]] = None
    rewrite: Optional[bool] = None


class ProvenanceDetail(BaseModel):
    page_number: Optional[int] = None
    bbox: Optional[Dict[str, float]] = None
    tables: Optional[List[Dict[str, Any]]] = None
    images: Optional[List[Dict[str, Any]]] = None


class Citation(BaseModel):
    chunk_id: str
    doc_id: Optional[str] = None
    score: float = 0.0
    text: str = ""
    metadata: Dict[str, Any] = {}
    provenance: Optional[ProvenanceDetail] = None


class AnswerResponse(BaseModel):
    answer: str
    citations: List[Citation]
    route: str
    confidence: float
    fallback_reason: Optional[str] = None
    evidence: Optional[List[Dict[str, Any]]] = None
    warning: Optional[str] = None
    detected_company: Optional[str] = None
    detected_metric: Optional[str] = None
    rewritten_query: Optional[str] = None


class EvaluateRequest(BaseModel):
    query: str
    top_k: int = Field(default=8, ge=1, le=50)
    filters: Optional[Dict[str, Any]] = None
    rewrite: Optional[bool] = None


class EvaluateResponse(BaseModel):
    query: str
    route: str
    confidence: float
    detected_company: Optional[str] = None
    detected_metric: Optional[str] = None
    fallback_reason: Optional[str] = None
    answer: str
    evidence: Optional[List[Dict[str, Any]]] = None
    citations: List[Citation]
    warning: Optional[str] = None
    rewritten_query: Optional[str] = None


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
        ".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".mp4",
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
    should_rewrite = (
        request.rewrite
        if request.rewrite is not None
        else settings.rewrite_query_by_default
    )

    rewritten_query = None
    search_query = request.query
    if should_rewrite and query_rewriter:
        rewritten_query = query_rewriter.rewrite(request.query)
        if rewritten_query != request.query:
            search_query = rewritten_query
        else:
            rewritten_query = None

    results = retriever.retrieve(
        query=search_query,
        top_k=request.top_k,
        filters=request.filters,
        original_query=request.query if should_rewrite else None,
    )
    hits: List[RetrievalHit] = [
        RetrievalHit(
            chunk_id=hit.get("chunk_id", hit.get("id", "")),
            doc_id=hit.get("doc_id", ""),
            score=hit.get("score", 0.0),
            text=hit.get("text", ""),
            metadata=hit.get("metadata", {}),
        )
        for hit in results
    ]
    return RetrieveResponse(results=hits, rewritten_query=rewritten_query)


def _extract_provenance_from_metadata(metadata: Dict[str, Any]) -> Optional[ProvenanceDetail]:
    prov = metadata.get("provenance")
    if not prov:
        return None
    return ProvenanceDetail(
        page_number=prov.get("page_number"),
        bbox=prov.get("bbox"),
        tables=prov.get("tables"),
        images=prov.get("images"),
    )


@app.post("/answer", response_model=AnswerResponse)
def answer(request: AnswerRequest) -> AnswerResponse:
    should_rewrite = (
        request.rewrite
        if request.rewrite is not None
        else settings.rewrite_query_by_default
    )

    result = answerer.answer(
        query=request.query,
        top_k=request.top_k,
        filters=request.filters,
        rewrite=should_rewrite,
    )
    citations: List[Citation] = []
    for c in result.citations:
        metadata = c.metadata or {}
        citations.append(
            Citation(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                score=c.score,
                text=c.text,
                metadata=metadata,
                provenance=_extract_provenance_from_metadata(metadata),
            )
        )
    return AnswerResponse(
        answer=result.answer,
        citations=citations,
        route=result.route,
        confidence=result.confidence,
        fallback_reason=result.fallback_reason,
        evidence=result.evidence,
        warning=result.warning,
        detected_company=result.detected_company,
        detected_metric=result.detected_metric,
        rewritten_query=result.rewritten_query,
    )


@app.post("/placement/evaluate", response_model=EvaluateResponse)
def placement_evaluate(request: EvaluateRequest) -> EvaluateResponse:
    should_rewrite = (
        request.rewrite
        if request.rewrite is not None
        else settings.rewrite_query_by_default
    )
    result = answerer.answer(
        query=request.query,
        top_k=request.top_k,
        filters=request.filters,
        rewrite=should_rewrite,
    )
    citations: List[Citation] = []
    for c in result.citations:
        metadata = c.metadata or {}
        citations.append(
            Citation(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                score=c.score,
                text=c.text,
                metadata=metadata,
                provenance=_extract_provenance_from_metadata(metadata),
            )
        )
    return EvaluateResponse(
        query=request.query,
        route=result.route,
        confidence=result.confidence,
        detected_company=result.detected_company,
        detected_metric=result.detected_metric,
        fallback_reason=result.fallback_reason,
        answer=result.answer,
        evidence=result.evidence,
        citations=citations,
        warning=result.warning,
        rewritten_query=result.rewritten_query,
    )


@app.get("/ops/parse-events")
def list_parse_events():
    return get_parse_events()


@app.get("/placement/facts")
def get_placement_facts(doc_id: Optional[str] = None):
    if doc_id:
        dataset = metadata_store.get_placement_dataset(doc_id)
    else:
        dataset = metadata_store.get_latest_placement_dataset()
    if dataset is None:
        raise HTTPException(status_code=404, detail="No placement dataset found")
    return dataset.model_dump(mode="json")
