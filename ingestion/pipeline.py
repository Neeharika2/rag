from typing import List, Tuple

from ingestion.chunk import chunk_file
from ingestion.embedding import embed_chunks
from ingestion.parser import parse_pdf


def run_pipeline(pdf_path: str) -> dict:
    parsed_file = parse_pdf(pdf_path)
    chunks_file, chunks = chunk_file(parsed_file)
    embedded_count = embed_chunks(chunks=chunks, source_file=pdf_path)

    return {
        "pdf_path": pdf_path,
        "parsed_file": parsed_file,
        "chunks_file": chunks_file,
        "chunks": chunks,
        "embedded_count": embedded_count,
    }