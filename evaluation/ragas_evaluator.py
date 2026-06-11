"""
evaluation/ragas_evaluator.py
------------------------------
Converts EvalRecords into a HuggingFace Dataset and runs RAGAS evaluation
using Gemini as the judge LLM.

Metric strategy
---------------
- Vector path (interview_text, generic_vector):
    faithfulness, answer_relevancy, context_precision, context_recall

- Structured path (structured_query, trend_query, conflict_check):
    answer_relevancy, answer_correctness
    (faithfulness/context_precision are skipped — structured answers have no
    retrieved text chunks; scoring serialised evidence dicts as "context"
    would produce misleading scores)

Usage
-----
    from evaluation.ragas_evaluator import RagasEvaluator
    evaluator = RagasEvaluator(api_key="...")
    report = evaluator.evaluate_all(records)
    print(report.summary())
"""

from __future__ import annotations
 
import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional
 
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
 
logger = logging.getLogger(__name__)
 
 
class RateLimiter:
    _last_call: float = 0.0
    _async_lock = asyncio.Lock()
    _sync_lock = threading.Lock()
 
    @classmethod
    async def wait_async(cls, delay: float = 5.0):
        async with cls._async_lock:
            now = time.time()
            elapsed = now - cls._last_call
            wait_time = delay - elapsed
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            cls._last_call = time.time()
 
    @classmethod
    def wait_sync(cls, delay: float = 5.0):
        with cls._sync_lock:
            now = time.time()
            elapsed = now - cls._last_call
            wait_time = delay - elapsed
            if wait_time > 0:
                time.sleep(wait_time)
            cls._last_call = time.time()
 
 
def clean_markdown_json(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


class RateLimitedChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    async def _agenerate(self, *args, **kwargs):
        for attempt in range(5):
            try:
                await RateLimiter.wait_async(5.0)
                res = await super()._agenerate(*args, **kwargs)
                if res and hasattr(res, "generations"):
                    for gen in res.generations:
                        if hasattr(gen, "text") and gen.text:
                            gen.text = clean_markdown_json(gen.text)
                        if hasattr(gen, "message") and gen.message and hasattr(gen.message, "content"):
                            if isinstance(gen.message.content, str):
                                gen.message.content = clean_markdown_json(gen.message.content)
                return res
            except Exception as exc:
                if "RESOURCE_EXHAUSTED" in str(exc) or "429" in str(exc):
                    logger.warning("Gemini API Rate Limit hit. Waiting 70 seconds before retrying... (Attempt %d/5)", attempt + 1)
                    await asyncio.sleep(70.0)
                else:
                    raise exc
        res = await super()._agenerate(*args, **kwargs)
        if res and hasattr(res, "generations"):
            for gen in res.generations:
                if hasattr(gen, "text") and gen.text:
                    gen.text = clean_markdown_json(gen.text)
                if hasattr(gen, "message") and gen.message and hasattr(gen.message, "content"):
                    if isinstance(gen.message.content, str):
                        gen.message.content = clean_markdown_json(gen.message.content)
        return res

    def _generate(self, *args, **kwargs):
        for attempt in range(5):
            try:
                RateLimiter.wait_sync(5.0)
                res = super()._generate(*args, **kwargs)
                if res and hasattr(res, "generations"):
                    for gen in res.generations:
                        if hasattr(gen, "text") and gen.text:
                            gen.text = clean_markdown_json(gen.text)
                        if hasattr(gen, "message") and gen.message and hasattr(gen.message, "content"):
                            if isinstance(gen.message.content, str):
                                gen.message.content = clean_markdown_json(gen.message.content)
                return res
            except Exception as exc:
                if "RESOURCE_EXHAUSTED" in str(exc) or "429" in str(exc):
                    logger.warning("Gemini API Rate Limit hit. Waiting 70 seconds before retrying... (Attempt %d/5)", attempt + 1)
                    time.sleep(70.0)
                else:
                    raise exc
        res = super()._generate(*args, **kwargs)
        if res and hasattr(res, "generations"):
            for gen in res.generations:
                if hasattr(gen, "text") and gen.text:
                    gen.text = clean_markdown_json(gen.text)
                if hasattr(gen, "message") and gen.message and hasattr(gen.message, "content"):
                    if isinstance(gen.message.content, str):
                        gen.message.content = clean_markdown_json(gen.message.content)
        return res



class RateLimitedGoogleGenerativeAIEmbeddings(GoogleGenerativeAIEmbeddings):
    async def aembed_documents(self, texts, *args, **kwargs):
        for attempt in range(5):
            try:
                await RateLimiter.wait_async(5.0)
                return await super().aembed_documents(texts, *args, **kwargs)
            except Exception as exc:
                if "RESOURCE_EXHAUSTED" in str(exc) or "429" in str(exc):
                    logger.warning("Gemini API Rate Limit hit. Waiting 70 seconds before retrying... (Attempt %d/5)", attempt + 1)
                    await asyncio.sleep(70.0)
                else:
                    raise exc
        return await super().aembed_documents(texts, *args, **kwargs)

    async def aembed_query(self, text, *args, **kwargs):
        for attempt in range(5):
            try:
                await RateLimiter.wait_async(5.0)
                return await super().aembed_query(text, *args, **kwargs)
            except Exception as exc:
                if "RESOURCE_EXHAUSTED" in str(exc) or "429" in str(exc):
                    logger.warning("Gemini API Rate Limit hit. Waiting 70 seconds before retrying... (Attempt %d/5)", attempt + 1)
                    await asyncio.sleep(70.0)
                else:
                    raise exc
        return await super().aembed_query(text, *args, **kwargs)

    def embed_documents(self, texts, *args, **kwargs):
        for attempt in range(5):
            try:
                RateLimiter.wait_sync(5.0)
                return super().embed_documents(texts, *args, **kwargs)
            except Exception as exc:
                if "RESOURCE_EXHAUSTED" in str(exc) or "429" in str(exc):
                    logger.warning("Gemini API Rate Limit hit. Waiting 70 seconds before retrying... (Attempt %d/5)", attempt + 1)
                    time.sleep(70.0)
                else:
                    raise exc
        return super().embed_documents(texts, *args, **kwargs)

    def embed_query(self, text, *args, **kwargs):
        for attempt in range(5):
            try:
                RateLimiter.wait_sync(5.0)
                return super().embed_query(text, *args, **kwargs)
            except Exception as exc:
                if "RESOURCE_EXHAUSTED" in str(exc) or "429" in str(exc):
                    logger.warning("Gemini API Rate Limit hit. Waiting 70 seconds before retrying... (Attempt %d/5)", attempt + 1)
                    time.sleep(70.0)
                else:
                    raise exc
        return super().embed_query(text, *args, **kwargs)

 
 
# Routes that use the vector retrieval path
_VECTOR_ROUTES = {"interview_text", "generic_vector"}
 
# Routes that use structured in-memory reasoning
_STRUCTURED_ROUTES = {"structured_query", "trend_query", "conflict_check", "hiring_query"}


@dataclass
class EvalReport:
    """
    Holds RAGAS scores for a single evaluation run.

    Attributes
    ----------
    vector_scores:
        Per-metric scores for the vector-path subset.
    structured_scores:
        Per-metric scores for the structured-path subset.
    run_at:
        ISO-8601 UTC timestamp of the evaluation run.
    num_vector_samples:
        Number of records evaluated on the vector path.
    num_structured_samples:
        Number of records evaluated on the structured path.
    model_name:
        Name of the judge LLM used for evaluation.
    errors:
        Any non-fatal errors encountered during the run.
    """

    vector_scores: Dict[str, float] = field(default_factory=dict)
    structured_scores: Dict[str, float] = field(default_factory=dict)
    run_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    num_vector_samples: int = 0
    num_structured_samples: int = 0
    model_name: str = "gemini-2.5-flash"
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable summary table."""
        lines = [
            f"RAGAS Evaluation Report  [{self.run_at}]",
            f"Judge model: {self.model_name}",
            "",
            f"Vector path  ({self.num_vector_samples} samples)",
            "  " + ("-" * 45),
        ]
        for metric, score in sorted(self.vector_scores.items()):
            lines.append(f"  {metric:<30} {score:.4f}")

        lines += [
            "",
            f"Structured path  ({self.num_structured_samples} samples)",
            "  " + ("-" * 45),
        ]
        for metric, score in sorted(self.structured_scores.items()):
            lines.append(f"  {metric:<30} {score:.4f}")

        if self.errors:
            lines += ["", "Errors:"] + [f"  - {e}" for e in self.errors]

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "run_at": self.run_at,
            "model_name": self.model_name,
            "num_vector_samples": self.num_vector_samples,
            "num_structured_samples": self.num_structured_samples,
            "vector_scores": self.vector_scores,
            "structured_scores": self.structured_scores,
            "errors": self.errors,
        }


class RagasEvaluator:
    """
    Runs RAGAS evaluation over a list of EvalRecords.

    Parameters
    ----------
    api_key:
        Gemini API key.  Used both for the judge LLM and for answer
        relevancy's reverse-question generation.
    model_name:
        Gemini model to use as the RAGAS judge.  Defaults to
        ``"gemini-2.5-flash"``.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash",
    ) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for RagasEvaluator")
        self._api_key = api_key
        self._model_name = model_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_all(self, records: List) -> EvalReport:
        """
        Partition *records* by path type and evaluate each partition.

        Parameters
        ----------
        records:
            List of :class:`~evaluation.pipeline_runner.EvalRecord` objects.

        Returns
        -------
        :class:`EvalReport` with scores for both paths.
        """
        vector_records = [r for r in records if r.route in _VECTOR_ROUTES]
        structured_records = [r for r in records if r.route in _STRUCTURED_ROUTES]

        report = EvalReport(
            model_name=self._model_name,
            num_vector_samples=len(vector_records),
            num_structured_samples=len(structured_records),
        )

        if vector_records:
            try:
                report.vector_scores = self._evaluate_vector_path(vector_records)
            except Exception as exc:
                msg = f"Vector path evaluation failed: {exc}"
                logger.error(msg)
                report.errors.append(msg)
        else:
            logger.info("No vector-path records to evaluate")

        if structured_records:
            try:
                report.structured_scores = self._evaluate_structured_path(
                    structured_records
                )
            except Exception as exc:
                msg = f"Structured path evaluation failed: {exc}"
                logger.error(msg)
                report.errors.append(msg)
        else:
            logger.info("No structured-path records to evaluate")

        return report

    def evaluate_vector_path(self, records: List) -> Dict[str, float]:
        """Evaluate only the vector-path records."""
        return self._evaluate_vector_path(records)

    def evaluate_structured_path(self, records: List) -> Dict[str, float]:
        """Evaluate only the structured-path records."""
        return self._evaluate_structured_path(records)

    # ------------------------------------------------------------------
    # Internal evaluation helpers
    # ------------------------------------------------------------------

    def _build_llm(self):
        """Build a LangChain ChatGoogleGenerativeAI instance for RAGAS."""
        return RateLimitedChatGoogleGenerativeAI(
            model=self._model_name,
            google_api_key=self._api_key,
            temperature=0,
        )
 
    def _build_embeddings(self):
        """Build a LangChain GoogleGenerativeAIEmbeddings for RAGAS."""
        from ragas.embeddings import LangchainEmbeddingsWrapper
        embeddings_model = RateLimitedGoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=self._api_key,
        )
        return LangchainEmbeddingsWrapper(embeddings_model)

    def _records_to_hf_dataset(self, records: List, include_ground_truth: bool = True):
        """Convert EvalRecords to a HuggingFace Dataset."""
        try:
            from datasets import Dataset
        except ImportError as exc:
            raise ImportError(
                "datasets is required for RAGAS evaluation. "
                "Install it with: pip install datasets"
            ) from exc

        data: dict = {
            "question": [r.question for r in records],
            "answer": [r.answer for r in records],
            "contexts": [r.contexts for r in records],
        }
        if include_ground_truth:
            data["ground_truth"] = [r.ground_truth for r in records]

        return Dataset.from_dict(data)

    def _scores_from_result(self, result) -> Dict[str, float]:
        """Extract per-metric float scores from a RAGAS result object."""
        scores: Dict[str, float] = {}
        # ragas.evaluate() returns an object whose .scores is a list of dicts
        # or a pandas DataFrame; handle both gracefully.
        try:
            import pandas as pd  # noqa: F401
            df = result.to_pandas()
            for col in df.columns:
                if col not in ("question", "answer", "contexts", "ground_truth"):
                    val = df[col].mean()
                    if not pd.isna(val):
                        scores[col] = float(val)
        except Exception:
            # Fallback: try dict-style access
            try:
                for metric_name, score in result.items():
                    scores[metric_name] = float(score)
            except Exception:
                logger.warning("Could not parse RAGAS result scores")

        if not scores:
            raise ValueError(
                "Ragas returned no valid scores. This usually happens when the judge LLM "
                "API calls fail (e.g. rate limits / exhausted quota)."
            )
        return scores

    def _evaluate_vector_path(self, records: List) -> Dict[str, float]:
        """
        Run faithfulness, answer_relevancy, context_precision, context_recall
        over vector-path records.
        """
        from ragas import evaluate
        from ragas.run_config import RunConfig
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        llm = self._build_llm()
        embeddings = self._build_embeddings()

        # Wire Gemini into each metric
        for metric in (faithfulness, answer_relevancy, context_precision, context_recall):
            metric.llm = llm
            if hasattr(metric, "embeddings"):
                metric.embeddings = embeddings

        dataset = self._records_to_hf_dataset(records, include_ground_truth=True)

        logger.info(
            "Running RAGAS vector-path evaluation on %d samples "
            "[faithfulness, answer_relevancy, context_precision, context_recall]",
            len(records),
        )
        run_config = RunConfig(
            max_workers=1,
            timeout=900,
            max_retries=10,
            max_wait=60,
        )
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            run_config=run_config,
        )
        return self._scores_from_result(result)

    def _evaluate_structured_path(self, records: List) -> Dict[str, float]:
        """
        Run answer_relevancy and answer_correctness over structured-path records.
        """
        from ragas import evaluate
        from ragas.run_config import RunConfig
        from ragas.metrics import answer_correctness, answer_relevancy

        llm = self._build_llm()
        embeddings = self._build_embeddings()

        for metric in (answer_relevancy, answer_correctness):
            metric.llm = llm
            if hasattr(metric, "embeddings"):
                metric.embeddings = embeddings

        dataset = self._records_to_hf_dataset(records, include_ground_truth=True)

        logger.info(
            "Running RAGAS structured-path evaluation on %d samples "
            "[answer_relevancy, answer_correctness]",
            len(records),
        )
        run_config = RunConfig(
            max_workers=1,
            timeout=900,
            max_retries=10,
            max_wait=60,
        )
        result = evaluate(
            dataset,
            metrics=[answer_relevancy, answer_correctness],
            run_config=run_config,
        )
        return self._scores_from_result(result)

