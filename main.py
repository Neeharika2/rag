import os
from fastapi import FastAPI, File, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from generation import generate_answer
from injestion.graph import run_ingestion_graph
from retrieval import hybrid_search

app = FastAPI()

UPLOAD_DIR = "uploads"


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    alpha: float = 0.6


@app.get("/", response_class=HTMLResponse)
def upload_page() -> str:
    return """
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>RAG Assistant</title>
        <style>
          body { font-family: Arial, sans-serif; margin: 2rem; background: #f5f5f5; }
          .container { max-width: 720px; margin: 0 auto; }
          .card { background: #fff; padding: 1.5rem; border: 1px solid #ddd; border-radius: 8px; margin-bottom: 1.5rem; }
          input[type="text"], input[type="file"] { width: 100%; padding: 0.5rem; margin-top: 0.5rem; box-sizing: border-box; }
          button { margin-top: 1rem; padding: 0.5rem 1rem; cursor: pointer; }
          #answer { margin-top: 1rem; padding: 1rem; background: #f9f9f9; border-left: 4px solid #007bff; display: none; }
          #sources { margin-top: 0.5rem; font-size: 0.9rem; color: #555; }
          .loading { color: #666; font-style: italic; }
          h2 { margin-top: 0; }
        </style>
      </head>
      <body>
        <div class="container">
          <div class="card">
            <h2>Upload a PDF</h2>
            <form id="uploadForm" enctype="multipart/form-data">
              <input type="file" name="file" accept="application/pdf" required />
              <div>
                <button type="submit">Upload</button>
              </div>
            </form>
            <div id="uploadStatus"></div>
          </div>

          <div class="card">
            <h2>Ask a Question</h2>
            <form id="queryForm">
              <input type="text" name="query" placeholder="Type your question here..." required />
              <div>
                <button type="submit">Ask</button>
              </div>
            </form>
            <div id="answer"></div>
            <div id="sources"></div>
          </div>
        </div>

        <script>
          document.getElementById('uploadForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const status = document.getElementById('uploadStatus');
            status.textContent = 'Uploading...';
            const formData = new FormData(e.target);
            try {
              const res = await fetch('/upload', { method: 'POST', body: formData });
              const data = await res.json();
              status.textContent = data.ok
                ? 'Uploaded and indexed ' + data.embedded + ' chunks.'
                : 'Error: ' + (data.error || 'Unknown error');
            } catch (err) {
              status.textContent = 'Error: ' + err.message;
            }
          });

          document.getElementById('queryForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const answerBox = document.getElementById('answer');
            const sourcesBox = document.getElementById('sources');
            answerBox.style.display = 'block';
            answerBox.textContent = 'Thinking...';
            sourcesBox.textContent = '';
            const formData = new FormData(e.target);
            const query = formData.get('query');
            try {
              const res = await fetch('/rag', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query, top_k: 5, alpha: 0.6 })
              });
              const data = await res.json();
              if (data.ok) {
                answerBox.textContent = data.answer;
                sourcesBox.innerHTML = '<strong>Sources:</strong> ' +
                  data.sources.map(s => '#' + s.id + ' (score: ' + s.score.toFixed(3) + ')').join(', ');
              } else {
                answerBox.textContent = 'Error: ' + (data.error || 'Unknown error');
              }
            } catch (err) {
              answerBox.textContent = 'Error: ' + err.message;
            }
          });
        </script>
      </body>
    </html>
    """


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)) -> dict:
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    filename = os.path.basename(file.filename or "uploaded.pdf")
    if not filename.lower().endswith(".pdf"):
        return {"ok": False, "error": "Only PDF files are allowed."}

    save_path = os.path.join(UPLOAD_DIR, filename)

    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    result = await run_in_threadpool(run_ingestion_graph, save_path)

    return {
        "ok": True,
        "file": filename,
        "path": save_path,
        "parsed": result["parsed_file"],
        "chunks": result["chunks_file"],
        "embedded": result["embedded_count"],
    }


@app.post("/query")
async def query_docs(payload: QueryRequest) -> dict:
    results = await run_in_threadpool(
        hybrid_search,
        payload.query,
        payload.top_k,
        payload.alpha,
    )

    return {"ok": True, "query": payload.query, "results": results}


@app.post("/rag")
async def rag_query(payload: QueryRequest) -> dict:
    results = await run_in_threadpool(
        hybrid_search,
        payload.query,
        payload.top_k,
        payload.alpha,
    )

    if not results:
        return {"ok": True, "query": payload.query, "answer": "No relevant documents found."}

    docs = [str(r["document"]) for r in results]
    answer = await run_in_threadpool(generate_answer, payload.query, docs)

    return {
        "ok": True,
        "query": payload.query,
        "answer": answer,
        "sources": [{"id": r["id"], "score": r["score"]} for r in results],
    }
