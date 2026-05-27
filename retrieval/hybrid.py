import math
import os
import re
from collections import Counter
from typing import Dict, List

import chromadb
from sentence_transformers import SentenceTransformer

from config import COLLECTION_NAME, DB_PATH, EMBEDDING_MODEL

_MODEL_CACHE: dict[str, SentenceTransformer] = {}


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
    model = _get_model()
    query_embedding = model.encode(query, normalize_embeddings=True).tolist()

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["distances", "documents", "metadatas"],
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


def hybrid_search(
    query: str,
    top_k: int = 5,
    alpha: float = 0.6,
) -> List[Dict[str, object]]:
    alpha = max(0.0, min(1.0, alpha))

    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    all_data = collection.get(include=["documents", "metadatas"])
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


def _get_model() -> SentenceTransformer:
    model_name = EMBEDDING_MODEL
    cached = _MODEL_CACHE.get(model_name)
    if cached is None:
        cached = SentenceTransformer(model_name)
        _MODEL_CACHE[model_name] = cached
    return cached