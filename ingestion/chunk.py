import os
from typing import List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE, CHUNKS_DIR


def chunk_file(input_file: str) -> Tuple[str, List[str]]:
    os.makedirs(CHUNKS_DIR, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(CHUNKS_DIR, f"{base_name}_chunks.txt")

    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            data = f.read()
        chunks = [c.strip() for c in data.split("--- Chunk ") if c.strip()]
        if chunks:
            print(f"  Skipping chunk (already exists): {output_file}")
            return output_file, chunks

    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = text_splitter.split_text(text)

    with open(output_file, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"\n\n--- Chunk {i + 1} ---\n\n")
            f.write(chunk)

    return output_file, chunks