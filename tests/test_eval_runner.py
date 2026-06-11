"""
tests/test_eval_runner.py
--------------------------
Integration tests for PipelineRunner and EvalRecord.

These tests use a stub Answerer (no live LLM / ChromaDB / embeddings)
so they run quickly in CI without network access.

The stub verifies that PipelineRunner correctly:
  - Calls answerer.answer() for each sample
  - Extracts contexts from citations (vector path)
  - Extracts contexts from evidence dicts (structured path)
  - Skips out-of-corpus samples by default
  - Handles Answerer exceptions gracefully (logs and continues)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from evaluation.pipeline_runner import EvalRecord, PipelineRunner
from evaluation.test_dataset import (
    CONFLICT_SAMPLES,
    GENERIC_SAMPLES,
    INTERVIEW_SAMPLES,
    OUT_OF_CORPUS_SAMPLES,
    STRUCTURED_SAMPLES,
    EvalSample,
)
from generation.answer_result import AnswerResult, Citation


# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------

def _make_citation(text: str) -> Citation:
    return Citation(chunk_id="c1", doc_id="doc1", score=0.9, text=text)


def _make_vector_result(answer: str, contexts: List[str]) -> AnswerResult:
    return AnswerResult(
        answer=answer,
        route="generic_vector",
        confidence=0.85,
        citations=[_make_citation(c) for c in contexts],
    )


def _make_structured_result(
    answer: str,
    evidence: Optional[List[Dict[str, Any]]] = None,
) -> AnswerResult:
    return AnswerResult(
        answer=answer,
        route="structured_query",
        confidence=0.95,
        citations=[],
        evidence=evidence or [],
    )


class _StubAnswerer:
    """Returns pre-canned AnswerResults without any real components."""

    def __init__(self, result: AnswerResult) -> None:
        self._result = result

    def answer(self, query: str, rewrite: bool = False, **kwargs) -> AnswerResult:  # noqa: D401
        return self._result


class _RaisingAnswerer:
    """Always raises on answer()."""

    def answer(self, query: str, **kwargs) -> AnswerResult:
        raise RuntimeError("Simulated Answerer failure")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPipelineRunnerVectorPath:
    def test_produces_one_record_per_sample(self) -> None:
        stub = _StubAnswerer(
            _make_vector_result("Paris is the capital.", ["Paris is the capital of France."])
        )
        runner = PipelineRunner(stub)
        records = runner.run(GENERIC_SAMPLES[:2])
        assert len(records) == 2

    def test_record_fields_are_populated(self) -> None:
        ctx_text = "TCS has a package of 4.1 LPA."
        stub = _StubAnswerer(_make_vector_result("4.1 LPA.", [ctx_text]))
        runner = PipelineRunner(stub)
        records = runner.run([GENERIC_SAMPLES[0]])

        r = records[0]
        assert r.question == GENERIC_SAMPLES[0].question
        assert r.answer == "4.1 LPA."
        assert ctx_text in r.contexts
        assert r.ground_truth == GENERIC_SAMPLES[0].ground_truth
        assert r.route == "generic_vector"
        assert r.confidence == 0.85

    def test_contexts_come_from_citation_text(self) -> None:
        texts = ["chunk A", "chunk B"]
        stub = _StubAnswerer(_make_vector_result("answer", texts))
        runner = PipelineRunner(stub)
        records = runner.run([INTERVIEW_SAMPLES[0]])
        assert records[0].contexts == texts


class TestPipelineRunnerStructuredPath:
    def test_contexts_come_from_evidence_dicts(self) -> None:
        evidence = [{"company": "TCS", "min_cgpa": 6.0, "package_lpa": 4.1}]
        stub = _StubAnswerer(_make_structured_result("TCS: 6.0 CGPA.", evidence))
        runner = PipelineRunner(stub)
        records = runner.run([STRUCTURED_SAMPLES[0]])

        r = records[0]
        assert len(r.contexts) == 1
        assert "company=TCS" in r.contexts[0]
        assert "min_cgpa=6.0" in r.contexts[0]

    def test_empty_evidence_falls_back_to_answer(self) -> None:
        stub = _StubAnswerer(_make_structured_result("Some answer.", evidence=[]))
        runner = PipelineRunner(stub)
        records = runner.run([STRUCTURED_SAMPLES[0]])
        assert records[0].contexts == ["Some answer."]


class TestPipelineRunnerSkipOutOfCorpus:
    def test_out_of_corpus_samples_are_skipped_by_default(self) -> None:
        stub = _StubAnswerer(_make_vector_result("n/a", []))
        runner = PipelineRunner(stub)
        records = runner.run(OUT_OF_CORPUS_SAMPLES)
        assert len(records) == 0

    def test_out_of_corpus_samples_included_when_flag_false(self) -> None:
        stub = _StubAnswerer(_make_vector_result("n/a", ["ctx"]))
        runner = PipelineRunner(stub, skip_out_of_corpus=False)
        records = runner.run(OUT_OF_CORPUS_SAMPLES)
        assert len(records) == len(OUT_OF_CORPUS_SAMPLES)


class TestPipelineRunnerErrorHandling:
    def test_exception_is_logged_and_skipped(self, caplog) -> None:
        import logging
        runner = PipelineRunner(_RaisingAnswerer())
        with caplog.at_level(logging.ERROR):
            records = runner.run(STRUCTURED_SAMPLES[:3])
        assert len(records) == 0
        assert any("Pipeline failed" in m for m in caplog.messages)

    def test_partial_failure_continues(self) -> None:
        """First sample raises, second succeeds — we should get 1 record."""
        call_count = [0]
        good_result = _make_vector_result("ok", ["context"])

        class _PartialAnswerer:
            def answer(self, query: str, **kwargs) -> AnswerResult:
                call_count[0] += 1
                if call_count[0] == 1:
                    raise RuntimeError("first fails")
                return good_result

        runner = PipelineRunner(_PartialAnswerer())
        records = runner.run(GENERIC_SAMPLES[:2])
        assert len(records) == 1


class TestEvalRecordDataclass:
    def test_eval_record_default_tags(self) -> None:
        r = EvalRecord(
            question="q",
            answer="a",
            contexts=["c"],
            ground_truth="gt",
            route="generic_vector",
            confidence=0.9,
        )
        assert r.tags == []
        assert r.warning is None

    def test_eval_record_with_warning(self) -> None:
        r = EvalRecord(
            question="q",
            answer="a",
            contexts=["c"],
            ground_truth="gt",
            route="conflict_check",
            confidence=0.95,
            warning="conflict_detected",
            tags=["conflict", "amazon"],
        )
        assert r.warning == "conflict_detected"
        assert "conflict" in r.tags
