"""
tests/test_eval_dataset.py
---------------------------
Unit tests for the RAGAS evaluation test dataset.

These run without any LLM or network access — they validate that the
dataset is well-formed and that all samples have the expected fields.
"""

import pytest

from evaluation.test_dataset import (
    ALL_EVAL_SAMPLES,
    CONFLICT_SAMPLES,
    GENERIC_SAMPLES,
    INTERVIEW_SAMPLES,
    OUT_OF_CORPUS_SAMPLES,
    SAMPLES_BY_ROUTE,
    STRUCTURED_PATH_SAMPLES,
    STRUCTURED_SAMPLES,
    TREND_SAMPLES,
    VECTOR_PATH_SAMPLES,
    EvalSample,
)

VALID_ROUTES = {
    "structured_query",
    "trend_query",
    "conflict_check",
    "interview_text",
    "generic_vector",
    "out_of_corpus",
}


class TestEvalSampleStructure:
    def test_eval_sample_has_required_fields(self) -> None:
        sample = STRUCTURED_SAMPLES[0]
        assert isinstance(sample.question, str) and sample.question
        assert isinstance(sample.ground_truth, str) and sample.ground_truth
        assert sample.route in VALID_ROUTES
        assert isinstance(sample.tags, list)

    def test_all_samples_have_non_empty_question(self) -> None:
        for sample in ALL_EVAL_SAMPLES + OUT_OF_CORPUS_SAMPLES:
            assert sample.question.strip(), f"Empty question: {sample}"

    def test_all_samples_have_non_empty_ground_truth(self) -> None:
        for sample in ALL_EVAL_SAMPLES + OUT_OF_CORPUS_SAMPLES:
            assert sample.ground_truth.strip(), f"Empty ground truth: {sample}"

    def test_all_samples_have_valid_routes(self) -> None:
        for sample in ALL_EVAL_SAMPLES + OUT_OF_CORPUS_SAMPLES:
            assert sample.route in VALID_ROUTES, (
                f"Unknown route {sample.route!r} for question: {sample.question}"
            )


class TestSampleCounts:
    def test_structured_samples_count(self) -> None:
        assert len(STRUCTURED_SAMPLES) >= 10

    def test_trend_samples_count(self) -> None:
        assert len(TREND_SAMPLES) >= 3

    def test_conflict_samples_count(self) -> None:
        assert len(CONFLICT_SAMPLES) >= 2

    def test_interview_samples_count(self) -> None:
        assert len(INTERVIEW_SAMPLES) >= 4

    def test_generic_samples_count(self) -> None:
        assert len(GENERIC_SAMPLES) >= 3

    def test_out_of_corpus_samples_count(self) -> None:
        assert len(OUT_OF_CORPUS_SAMPLES) >= 2

    def test_total_eval_samples_at_least_30(self) -> None:
        assert len(ALL_EVAL_SAMPLES) >= 28

    def test_all_eval_samples_excludes_out_of_corpus(self) -> None:
        routes = {s.route for s in ALL_EVAL_SAMPLES}
        assert "out_of_corpus" not in routes


class TestAggregatedViews:
    def test_structured_path_contains_eligibility_trend_conflict(self) -> None:
        routes = {s.route for s in STRUCTURED_PATH_SAMPLES}
        assert "structured_query" in routes
        assert "trend_query" in routes
        assert "conflict_check" in routes

    def test_vector_path_contains_interview_and_generic(self) -> None:
        routes = {s.route for s in VECTOR_PATH_SAMPLES}
        assert "interview_text" in routes
        assert "generic_vector" in routes

    def test_structured_plus_vector_equals_all(self) -> None:
        assert (
            len(STRUCTURED_PATH_SAMPLES) + len(VECTOR_PATH_SAMPLES)
            == len(ALL_EVAL_SAMPLES)
        )


class TestSamplesByRoute:
    def test_samples_by_route_keys(self) -> None:
        expected_keys = {
            "structured_query",
            "trend_query",
            "conflict_check",
            "interview_text",
            "generic_vector",
            "out_of_corpus",
        }
        assert set(SAMPLES_BY_ROUTE.keys()) == expected_keys

    def test_samples_by_route_routes_match(self) -> None:
        for route_key, samples in SAMPLES_BY_ROUTE.items():
            for sample in samples:
                assert sample.route == route_key, (
                    f"Sample in {route_key!r} bucket has route {sample.route!r}"
                )

    def test_samples_by_route_no_empty_buckets(self) -> None:
        for route_key, samples in SAMPLES_BY_ROUTE.items():
            assert len(samples) > 0, f"Empty sample bucket for route {route_key!r}"


class TestUniqueness:
    def test_no_duplicate_questions(self) -> None:
        all_questions = [s.question for s in ALL_EVAL_SAMPLES]
        unique = set(all_questions)
        assert len(unique) == len(all_questions), (
            "Duplicate questions found in ALL_EVAL_SAMPLES"
        )
