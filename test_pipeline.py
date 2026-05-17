#!/usr/bin/env python3
"""Test script to validate the RAG pipeline end-to-end."""

import os
import sys

# Ensure we're in the repo root
os.chdir(r"C:\Users\neeha\Documents\rag")

print("=" * 60)
print("RAG PIPELINE ANALYSIS & TEST")
print("=" * 60)

# ── 1. Environment / Dependencies ──────────────────────────────
print("\n[1] Dependency Check")

try:
    from sentence_transformers import SentenceTransformer
    print("  sentence-transformers  OK")
except Exception as e:
    print(f"  sentence-transformers  FAIL: {e}")

try:
    import chromadb
    print(f"  chromadb               OK (v{chromadb.__version__})")
except Exception as e:
    print(f"  chromadb               FAIL: {e}")

try:
    import google.generativeai as genai
    print("  google-generativeai    OK")
except Exception as e:
    print(f"  google-generativeai    FAIL: {e}")

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    print("  langchain-text-splitters  OK")
except Exception as e:
    print(f"  langchain-text-splitters  FAIL: {e}")

try:
    from llama_parse import LlamaParse
    print("  llama-parse            OK")
except Exception as e:
    print(f"  llama-parse            FAIL: {e}")

# ── 2. Test Embeddings (Sentence Transformers) ─────────────────
print("\n[2] Embedding Generation Test")
try:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    test_text = "This is a test sentence for embedding generation."
    embedding = model.encode(test_text, normalize_embeddings=True)
    print(f"  Model loaded: all-MiniLM-L6-v2")
    print(f"  Embedding dim: {len(embedding)}")
    print(f"  Embedding sample: [{embedding[0]:.4f}, {embedding[1]:.4f}, ...]")
    print("  Embedding generation   OK")
except Exception as e:
    print(f"  Embedding generation   FAIL: {e}")

# ── 3. Test ChromaDB (fresh in-memory) ─────────────────────────
print("\n[3] Vector DB Test")
try:
    client = chromadb.Client()
    coll = client.create_collection("test_coll")
    coll.add(ids=["1"], embeddings=[[0.1]*384], documents=["test doc"])
    result = coll.query(query_embeddings=[[0.1]*384], n_results=1)
    print(f"  In-memory ChromaDB     OK")
    print(f"  Query returned {len(result['ids'][0])} result(s)")
except Exception as e:
    print(f"  Vector DB test         FAIL: {e}")

# ── 4. Test Existing Persistent DB ─────────────────────────────
print("\n[4] Persistent DB Check (db/chroma)")
if os.path.exists("db/chroma/chroma.sqlite3"):
    print("  DB file exists")
    try:
        client = chromadb.PersistentClient(path="db/chroma")
        coll = client.get_or_create_collection("rag_collection")
        count = coll.count()
        print(f"  Collection 'rag_collection' exists with {count} document(s)")
    except Exception as e:
        print(f"  ERROR reading existing DB: {e}")
        print("  >>> The DB may be corrupted or from an incompatible ChromaDB version.")
else:
    print("  No existing DB found at db/chroma")

# ── 5. Test Chunking ───────────────────────────────────────────
print("\n[5] Chunking Test")
try:
    from injestion.chunk import chunk_file
    if os.path.exists("parsed_output/resume_neeha.md"):
        out = chunk_file("parsed_output/resume_neeha.md", output_dir="test_chunks")
        with open(out, "r", encoding="utf-8") as f:
            data = f.read()
        chunks = [c for c in data.split("--- Chunk ") if c.strip()]
        print(f"  Chunking               OK")
        print(f"  Generated {len(chunks)} chunk(s)")
    else:
        print("  No parsed file to chunk")
except Exception as e:
    print(f"  Chunking               FAIL: {e}")

# ── 6. Test Retrieval (Hybrid Search) ──────────────────────────
print("\n[6] Hybrid Search Test")
try:
    from retrieval.hybrid import hybrid_search
    # Use a fresh temp DB to avoid corruption issues
    import tempfile, shutil
    tmpdir = tempfile.mkdtemp()
    test_db = os.path.join(tmpdir, "test_chroma")
    os.makedirs(test_db, exist_ok=True)
    client = chromadb.PersistentClient(path=test_db)
    coll = client.get_or_create_collection("rag_collection")
    docs = [
        "Python is a high-level programming language.",
        "Machine learning uses algorithms to learn patterns.",
        "ChromaDB is a vector database for embeddings.",
    ]
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    for i, d in enumerate(docs):
        emb = model.encode(d, normalize_embeddings=True).tolist()
        coll.add(ids=[str(i)], embeddings=[emb], documents=[d])
    results = hybrid_search("What is Python?", top_k=2, db_path=test_db)
    print(f"  Hybrid search          OK")
    print(f"  Top result: {results[0]['document'][:50]}...")
    shutil.rmtree(tmpdir)
except Exception as e:
    print(f"  Hybrid search          FAIL: {e}")

# ── 7. Test Gemini Generation ──────────────────────────────────
print("\n[7] Gemini Generation Test")
try:
    import google.generativeai as genai
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("  GOOGLE_API_KEY not found in .env")
    else:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content("Say 'Hello from Gemini' in 5 words.")
        print(f"  Gemini generation      OK")
        print(f"  Response: {response.text.strip()}")
except Exception as e:
    print(f"  Gemini generation      FAIL: {e}")

# ── 8. Check for Generation Module in Codebase ─────────────────
print("\n[8] Codebase Generation Module Check")
gen_files = []
for root, dirs, files in os.walk("."):
    # skip venv and git
    dirs[:] = [d for d in dirs if d not in {".venv", ".git", "__pycache__"}]
    for f in files:
        if f.endswith(".py"):
            path = os.path.join(root, f)
            with open(path, "r", encoding="utf-8") as fh:
                content = fh.read()
            if "gemini" in content.lower() or "generate" in content.lower() or "generative" in content.lower():
                gen_files.append(path)
if gen_files:
    print(f"  Files mentioning generation/Gemini:")
    for gf in gen_files:
        print(f"    - {gf}")
else:
    print("  No generation/Gemini references found in Python files")

# ── 9. Check main.py for generation endpoint ───────────────────
print("\n[9] API Endpoint Check")
with open("main.py", "r", encoding="utf-8") as f:
    main_code = f.read()
if "generate" in main_code.lower():
    print("  main.py contains generation logic")
else:
    print("  main.py does NOT contain generation logic")
    print("  Current /query endpoint only returns retrieval results")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
