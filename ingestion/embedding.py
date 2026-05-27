import os
import uuid
from typing import List

import chromadb
from sentence_transformers import SentenceTransformer

from config import COLLECTION_NAME, DB_PATH, EMBEDDING_MODEL


def embed_chunks(chunks: List[str], source_file: str = "") -> int:
    if not chunks:
        return 0

    model = SentenceTransformer(EMBEDDING_MODEL)

    os.makedirs(DB_PATH, exist_ok=True)

    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    source_name = os.path.basename(source_file) if source_file else ""

    if source_name:
        existing = collection.get(include=["metadatas"])
        ids_list = existing.get("ids") or []
        metas = existing.get("metadatas") or []
        already_embedded = [
            eid for eid, meta in zip(ids_list, metas)
            if meta and meta.get("source") == source_name
        ]
        if already_embedded and len(already_embedded) >= len(chunks):
            print(f"  Skipping embed (already embedded): {source_name}")
            return len(already_embedded)
        if already_embedded:
            collection.delete(ids=already_embedded)

    existing_count = collection.count()

    documents = []
    embeddings = []
    ids = []
    metadatas = []

    for i, doc in enumerate(chunks):
        chunk_text = doc.strip()
        if not chunk_text:
            continue

        embedding = model.encode(chunk_text, normalize_embeddings=True).tolist()
        chunk_id = str(uuid.uuid4())

        metadata: dict = {"chunk_index": str(existing_count + i)}
        if source_name:
            metadata["source"] = source_name

        documents.append(chunk_text)
        embeddings.append(embedding)
        ids.append(chunk_id)
        metadatas.append(metadata)

    if documents:
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    return len(documents)