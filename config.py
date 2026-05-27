import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
PARSED_DIR = os.path.join(BASE_DIR, "parsed_output")
CHUNKS_DIR = os.path.join(BASE_DIR, "chunks")
DB_PATH = os.path.join(BASE_DIR, "db", "chroma")

COLLECTION_NAME = "rag_collection"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash"
RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 5
DEFAULT_ALPHA = 0.6
MULTI_QUERY_COUNT = 3