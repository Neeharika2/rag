import os
from fastapi import FastAPI, File, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse

from injestion.graph import run_ingestion_graph

app = FastAPI()

UPLOAD_DIR = "uploads"


@app.get("/", response_class=HTMLResponse)
def upload_page() -> str:
    return """
    <!doctype html>
    <html>
      <head>
        <meta charset=\"utf-8\" />
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
        <title>PDF Upload</title>
        <style>
          body { font-family: Arial, sans-serif; margin: 2rem; }
          .card { max-width: 520px; padding: 1.5rem; border: 1px solid #ddd; border-radius: 8px; }
          button { margin-top: 1rem; }
        </style>
      </head>
      <body>
        <div class=\"card\">
          <h2>Upload a PDF</h2>
          <form action=\"/upload\" method=\"post\" enctype=\"multipart/form-data\">
            <input type=\"file\" name=\"file\" accept=\"application/pdf\" required />
            <div>
              <button type=\"submit\">Upload</button>
            </div>
          </form>
        </div>
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
