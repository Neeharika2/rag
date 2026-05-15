import os
from dotenv import load_dotenv

import chromadb
import google.generativeai as genai


def embed_chunks(
    chunks_file: str,
    db_path: str = "db/chroma",
    collection_name: str = "rag_collection",
) -> int:
    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file")

    genai.configure(api_key=api_key)

    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=collection_name)

    with open(chunks_file, "r", encoding="utf-8") as f:
        data = f.read()

    chunks = data.split("--- Chunk ")
    documents = []

    for chunk in chunks:
        chunk = chunk.strip()
        if len(chunk) > 0:
            documents.append(chunk)

    for i, doc in enumerate(documents):
        response = genai.embed_content(
            model="models/text-embedding-004",
            content=doc,
        )

        embedding = response["embedding"]

        collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[doc],
        )

    return len(documents)


if __name__ == "__main__":
    total = embed_chunks("chunks/chunks.txt")
    print(f"All embeddings stored successfully. Total: {total}")