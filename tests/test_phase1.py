import os
import tempfile
import json
from unittest.mock import MagicMock

from settings import Settings
from vectorstore.chroma_store import ChromaVectorStore
from chunking.recursive import Chunk
from retrieval.retriever import Retriever


def test_settings_default_and_env():
    # Test default values
    settings = Settings.from_env()
    assert settings.retrieval_min_score == 0.25
    assert settings.dedupe_similarity_threshold == 0.95

    # Test loading from env
    os.environ["RETRIEVAL_MIN_SCORE"] = "0.45"
    os.environ["DEDUPE_SIMILARITY_THRESHOLD"] = "0.85"
    try:
        settings_env = Settings.from_env()
        assert settings_env.retrieval_min_score == 0.45
        assert settings_env.dedupe_similarity_threshold == 0.85
    finally:
        del os.environ["RETRIEVAL_MIN_SCORE"]
        del os.environ["DEDUPE_SIMILARITY_THRESHOLD"]


class DummyChromaVectorStore(ChromaVectorStore):
    def __init__(self):
        pass


def test_metadata_normalization():
    store = DummyChromaVectorStore()

    raw_metadata = {
        "company": "Amazon",
        "score_val": 42.0,
        "is_valid": True,
        "nested_dict": {"key": "val", "nested": [1, 2, 3]},
        "nested_list": ["a", "b", "c"],
        "none_val": None,
    }

    normalized = store._normalize_metadata(raw_metadata)

    # Check normalization
    assert normalized["company"] == "Amazon"
    assert normalized["score_val"] == 42.0
    assert normalized["is_valid"] is True
    assert "none_val" not in normalized
    assert json.loads(normalized["nested_dict"]) == {"key": "val", "nested": [1, 2, 3]}
    assert json.loads(normalized["nested_list"]) == ["a", "b", "c"]

    # Check denormalization
    denormalized = store._denormalize_metadata(normalized)
    assert denormalized["company"] == "Amazon"
    assert denormalized["score_val"] == 42.0
    assert denormalized["is_valid"] is True
    assert denormalized["nested_dict"] == {"key": "val", "nested": [1, 2, 3]}
    assert denormalized["nested_list"] == ["a", "b", "c"]


def test_retriever_min_score_filtering():
    embedding_provider = MagicMock()
    embedding_provider.embed_query.return_value = [0.1, 0.2]

    vector_store = MagicMock()
    # Mocking vector store search output
    mock_hits = [
        {"id": "hit1", "score": 0.8, "payload": {"chunk_id": "c1", "doc_id": "d1", "text": "High score text"}},
        {"id": "hit2", "score": 0.3, "payload": {"chunk_id": "c2", "doc_id": "d1", "text": "Medium score text"}},
        {"id": "hit3", "score": 0.1, "payload": {"chunk_id": "c3", "doc_id": "d1", "text": "Low score text"}},
    ]
    vector_store.search.return_value = mock_hits

    query_logger = MagicMock()

    # Case 1: min_score = 0.0 (no filtering)
    retriever_all = Retriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        query_logger=query_logger,
        min_score=0.0,
    )
    res_all = retriever_all.retrieve("test query", top_k=3)
    assert len(res_all) == 3
    assert [r["chunk_id"] for r in res_all] == ["c1", "c2", "c3"]

    # Case 2: min_score = 0.25 (should filter out hit3)
    retriever_mid = Retriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        query_logger=query_logger,
        min_score=0.25,
    )
    res_mid = retriever_mid.retrieve("test query", top_k=3)
    assert len(res_mid) == 2
    assert [r["chunk_id"] for r in res_mid] == ["c1", "c2"]

    # Case 3: min_score = 0.5 (should filter out hit2 and hit3)
    retriever_high = Retriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        query_logger=query_logger,
        min_score=0.5,
    )
    res_high = retriever_high.retrieve("test query", top_k=3)
    assert len(res_high) == 1
    assert [r["chunk_id"] for r in res_high] == ["c1"]
