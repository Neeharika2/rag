import math
import os
import re
from collections import Counter
from typing import Dict, List, Optional

import chromadb
from google import genai
from dotenv import load_dotenv

DEFAULT_DB_PATH = "db/chroma"
DEFAULT_COLLECTION = "rag_collection"


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _build_keyword_index(documents: List[str]) -> Dict[str, Dict[str, int]]:
    term_counts: Dict[str, Dict[str, int]] = {}
    for idx, doc in enumerate(documents):
        tokens = _tokenize(doc)
        counts = Counter(tokens)
        term_counts[str(idx)] = dict(counts)
    return term_counts


def _compute_idf(term_counts: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    df: Counter = Counter()
    total_docs = len(term_counts)
    for counts in term_counts.values():
        df.update(counts.keys())
    idf: Dict[str, float] = {}
    for term, freq in df.items():
        idf[term] = math.log((total_docs + 1) / (freq + 1)) + 1.0
    return idf


def _keyword_scores(
    query: str,
    documents: List[str],
) -> Dict[str, float]:
    term_counts = _build_keyword_index(documents)
    idf = _compute_idf(term_counts)
    query_terms = _tokenize(query)

    scores: Dict[str, float] = {}
    for idx, counts in term_counts.items():
        score = 0.0
        for term in query_terms:
            score += counts.get(term, 0) * idf.get(term, 0.0)
        scores[idx] = score

    max_score = max(scores.values(), default=0.0)
    if max_score <= 0.0:
        return {k: 0.0 for k in scores}

    return {k: v / max_score for k, v in scores.items()}


def _semantic_scores(
    query: str,
    collection: chromadb.Collection,
    top_k: int,
) -> Dict[str, float]:
    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file")

    client = genai.Client(api_key=api_key)
    response = client.models.embed_content(
        model="text-embedding-004",
        contents=query,
    )
    query_embedding = _extract_embedding(response)

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["distances", "ids"],
    )

    ids = result.get("ids", [[]])[0]
    distances = result.get("distances", [[]])[0]

    scores: Dict[str, float] = {}
    for doc_id, distance in zip(ids, distances):
        score = 1.0 / (1.0 + float(distance))
        scores[doc_id] = score

    max_score = max(scores.values(), default=0.0)
    if max_score <= 0.0:
        return {k: 0.0 for k in scores}

    return {k: v / max_score for k, v in scores.items()}


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


def hybrid_search(
    query: str,
    top_k: int = 5,
    alpha: float = 0.6,
    db_path: str = DEFAULT_DB_PATH,
    collection_name: str = DEFAULT_COLLECTION,
) -> List[Dict[str, object]]:
    alpha = max(0.0, min(1.0, alpha))
    os.makedirs(db_path, exist_ok=True)

    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=collection_name)

    all_data = collection.get(include=["documents", "metadatas", "ids"])
    documents = all_data.get("documents") or []
    ids = all_data.get("ids") or []
    metadatas = all_data.get("metadatas") or [None] * len(ids)

    if not documents or not ids:
        return []

    keyword_scores = _keyword_scores(query, documents)
    semantic_scores = _semantic_scores(query, collection, top_k=max(top_k, 10))

    results: List[Dict[str, object]] = []
    for idx, doc_id in enumerate(ids):
        semantic = semantic_scores.get(doc_id, 0.0)
        keyword = keyword_scores.get(str(idx), 0.0)
        hybrid = (alpha * semantic) + ((1.0 - alpha) * keyword)
        if hybrid <= 0.0:
            continue
        results.append(
            {
                "id": doc_id,
                "document": documents[idx],
                "metadata": metadatas[idx],
                "score": hybrid,
                "semantic_score": semantic,
                "keyword_score": keyword,
            }
        )

    results.sort(key=lambda item: float(item["score"]), reverse=True)
    return results[:top_k]
