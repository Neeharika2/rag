import os
from dotenv import load_dotenv

import chromadb
from google import genai


def embed_chunks(
    chunks_file: str,
    db_path: str = "db/chroma",
    collection_name: str = "rag_collection",
) -> int:
    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file")

    client = genai.Client(api_key=api_key)

    os.makedirs(db_path, exist_ok=True)

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
        response = client.models.embed_content(
            model="text-embedding-004",
            contents=doc,
        )

        embedding = _extract_embedding(response)

        collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[doc],
            metadatas=[{"chunk_index": i}],
        )

    return len(documents)


def _extract_embedding(response: object) -> list[float]:
    if hasattr(response, "embeddings"):
        embeddings = getattr(response, "embeddings")
        if embeddings and hasattr(embeddings[0], "values"):
            return list(embeddings[0].values)

    if isinstance(response, dict):
        if "embedding" in response:
            return list(response["embedding"])
        embeddings = response.get("embeddings") or []
        if embeddings and isinstance(embeddings[0], dict) and "values" in embeddings[0]:
            return list(embeddings[0]["values"])

    raise ValueError("Unexpected embedding response format")


if __name__ == "__main__":
    total = embed_chunks("chunks/chunks.txt")
    print(f"All embeddings stored successfully. Total: {total}")