"""
evaluation/run_eval.py
-----------------------
CLI entrypoint for the RAGAS evaluation suite.

Run all routes:
    python -m evaluation.run_eval

Run a single route:
    python -m evaluation.run_eval --route structured_query
    python -m evaluation.run_eval --route interview_text
    python -m evaluation.run_eval --route generic_vector
    python -m evaluation.run_eval --route trend_query
    python -m evaluation.run_eval --route conflict_check

Options:
    --route ROUTE       Evaluate only samples from this route (default: all).
    --no-db             Skip persisting results to the SQLite metadata DB.
    --no-json           Skip writing JSON report to disk.
    --results-dir DIR   Directory for JSON reports (default: evaluation/results).
    --rewrite           Enable query rewriting during pipeline runs.
    --dry-run           Build the pipeline and print sample count, then exit.
"""

from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger(__name__)


def _build_argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m evaluation.run_eval",
        description="Run RAGAS evaluation over the placement RAG pipeline.",
    )
    p.add_argument(
        "--route",
        choices=[
            "structured_query",
            "trend_query",
            "conflict_check",
            "interview_text",
            "generic_vector",
        ],
        default=None,
        help="Evaluate only this route (default: all routes).",
    )
    p.add_argument(
        "--results-dir",
        default="evaluation/results",
        help="Directory for JSON report files (default: evaluation/results).",
    )
    p.add_argument(
        "--no-db",
        action="store_true",
        help="Skip persisting results to the metadata SQLite DB.",
    )
    p.add_argument(
        "--no-json",
        action="store_true",
        help="Skip writing the JSON report to disk.",
    )
    p.add_argument(
        "--rewrite",
        action="store_true",
        help="Enable query rewriting during pipeline runs.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the pipeline, print sample count, then exit without evaluation.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of samples to evaluate (default: all).",
    )
    return p


def _build_answerer(settings):
    """Construct a live Answerer from Settings."""
    from agents.query_rewriter import QueryRewriter
    from embeddings.gemini import GeminiEmbeddingProvider
    from evaluation.query_logger import QueryLogger
    from generation.gemini import GeminiGenerator
    from generation.answerer import Answerer
    from ingestion.metadata_store import MetadataStore
    from retrieval.retriever import Retriever
    from vectorstore.chroma_store import ChromaVectorStore

    metadata_store = MetadataStore(settings.metadata_db_url)
    embedding_provider = GeminiEmbeddingProvider(
        api_key=settings.gemini_api_key,
        model_name=settings.embedding_model,
    )
    vector_store = ChromaVectorStore(
        persist_dir=settings.chroma_path,
        collection_name=settings.chroma_collection,
    )
    query_logger = QueryLogger(metadata_store)
    retriever = Retriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        query_logger=query_logger,
        min_score=settings.retrieval_min_score,
    )
    generator = GeminiGenerator(
        api_key=settings.gemini_api_key,
        model_name=settings.gemini_model,
    )
    query_rewriter = QueryRewriter(generator=generator)
    return Answerer(
        retriever=retriever,
        generator=generator,
        metadata_store=metadata_store,
        query_rewriter=query_rewriter,
    )


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    # ----- Load settings -----
    from settings import Settings
    settings = Settings.from_env()

    if not settings.gemini_api_key:
        logger.error("GEMINI_API_KEY is not set. Cannot run RAGAS evaluation.")
        return 1

    # ----- Select samples -----
    from evaluation.test_dataset import ALL_EVAL_SAMPLES, SAMPLES_BY_ROUTE

    if args.route:
        samples = SAMPLES_BY_ROUTE.get(args.route, [])
        if not samples:
            logger.error("No samples found for route %r", args.route)
            return 1
        logger.info("Route filter: %s (%d samples)", args.route, len(samples))
    else:
        samples = ALL_EVAL_SAMPLES
        logger.info("Evaluating all routes (%d samples)", len(samples))
 
    if args.limit:
        samples = samples[:args.limit]
        logger.info("Limiting evaluation to first %d samples", args.limit)
 
    if args.dry_run:
        print(f"Dry run: {len(samples)} samples selected, pipeline ready.")
        return 0

    # ----- Build pipeline -----
    logger.info("Building Answerer pipeline...")
    try:
        answerer = _build_answerer(settings)
    except Exception as exc:
        logger.error("Failed to build Answerer: %s", exc)
        return 1

    # ----- Run pipeline -----
    from evaluation.pipeline_runner import PipelineRunner

    runner = PipelineRunner(answerer, rewrite=args.rewrite)
    records = runner.run(samples)

    if not records:
        logger.error("No EvalRecords collected — aborting evaluation.")
        return 1

    # ----- RAGAS evaluation -----
    from evaluation.ragas_evaluator import RagasEvaluator

    evaluator = RagasEvaluator(
        api_key=settings.gemini_api_key,
        model_name=settings.gemini_model,
    )
    logger.info("Starting RAGAS evaluation...")
    report = evaluator.evaluate_all(records)

    # ----- Print summary -----
    print("\n" + "=" * 60)
    print(report.summary())
    print("=" * 60 + "\n")

    # ----- Persist results -----
    from evaluation.result_store import ResultStore

    store = ResultStore(
        results_dir=args.results_dir,
        db_url=settings.metadata_db_url,
    )
    json_path = store.save(
        report,
        save_json=not args.no_json,
        save_db=not args.no_db,
    )
    if json_path:
        print(f"Report saved to: {json_path}")

    # Return non-zero if there were errors
    return 1 if report.errors else 0


if __name__ == "__main__":
    sys.exit(main())
