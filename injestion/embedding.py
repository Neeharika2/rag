import os

import chromadb
from sentence_transformers import SentenceTransformer


def embed_chunks(
    chunks_file: str,
    db_path: str = "db/chroma",
    collection_name: str = "rag_collection",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> int:
    model = SentenceTransformer(model_name)

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
        embedding = model.encode(doc, normalize_embeddings=True).tolist()

        collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[doc],
            metadatas=[{"chunk_index": i}],
        )

    return len(documents)


if __name__ == "__main__":
    total = embed_chunks("chunks/chunks.txt")
    print(f"All embeddings stored successfully. Total: {total}")