"""
evaluation/pipeline_runner.py
------------------------------
Runs EvalSample questions through the live Answerer pipeline and collects
the structured EvalRecord output needed by RagasEvaluator.

Usage
-----
    from evaluation.pipeline_runner import PipelineRunner
    from evaluation.test_dataset import ALL_EVAL_SAMPLES

    runner = PipelineRunner(answerer)
    records = runner.run(ALL_EVAL_SAMPLES)
"""

from __future__ import annotations
 
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional
 
if TYPE_CHECKING:
    from generation.answerer import Answerer
 
from evaluation.test_dataset import EvalSample
 
logger = logging.getLogger(__name__)
 
 
@dataclass
class EvalRecord:
    """
    One row of evaluation data, ready for RAGAS.
 
    Attributes
    ----------
    question:
        The original user question.
    answer:
        The answer produced by the pipeline.
    contexts:
        List of text strings that were used to produce the answer.
        For vector-path answers these are the retrieved chunk texts.
        For structured-path answers these are the serialised evidence dicts.
    ground_truth:
        The expected correct answer (from EvalSample).
    route:
        The query route that was taken (mirrors query_router constants).
    confidence:
        Confidence score returned by the pipeline.
    warning:
        Optional warning string from the pipeline (e.g. "conflict_detected").
    """
 
    question: str
    answer: str
    contexts: List[str]
    ground_truth: str
    route: str
    confidence: float
    warning: Optional[str] = None
    tags: List[str] = field(default_factory=list)
 
 
class PipelineRunner:
    """
    Drives the Answerer over a list of EvalSamples and collects EvalRecords.
 
    Parameters
    ----------
    answerer:
        A fully initialised :class:`~generation.answerer.Answerer` instance.
    rewrite:
        Whether to enable query rewriting during evaluation.
        Defaults to ``False`` so that scores reflect the raw pipeline.
    skip_out_of_corpus:
        If ``True``, samples with route ``"out_of_corpus"`` are skipped
        because RAGAS metrics do not make sense for refusal responses.
    """
 
    def __init__(
        self,
        answerer: "Answerer",
        rewrite: bool = False,
        skip_out_of_corpus: bool = True,
    ) -> None:
        self._answerer = answerer
        self._rewrite = rewrite
        self._skip_out_of_corpus = skip_out_of_corpus
 
    def run(self, samples: List[EvalSample]) -> List[EvalRecord]:
        """
        Run all *samples* through the pipeline.
 
        Parameters
        ----------
        samples:
            List of :class:`~evaluation.test_dataset.EvalSample` objects.
 
        Returns
        -------
        List of :class:`EvalRecord` objects — one per successfully processed
        sample.  Samples that raise an exception are logged and skipped.
        """
        records: List[EvalRecord] = []
 
        for idx, sample in enumerate(samples, start=1):
            if self._skip_out_of_corpus and sample.route == "out_of_corpus":
                logger.debug("Skipping out-of-corpus sample: %s", sample.question)
                continue
 
            logger.info(
                "[%d/%d] Running: %s (route=%s)",
                idx, len(samples), sample.question[:60], sample.route,
            )
 
            try:
                result = self._answerer.answer(
                    sample.question,
                    rewrite=self._rewrite,
                )
            except Exception as exc:
                logger.error(
                    "Pipeline failed for question %r: %s", sample.question, exc
                )
                continue
            finally:
                if idx < len(samples):
                    time.sleep(4.0)
 
            contexts = self._extract_contexts(result)
 
            records.append(
                EvalRecord(
                    question=sample.question,
                    answer=result.answer,
                    contexts=contexts,
                    ground_truth=sample.ground_truth,
                    route=result.route,
                    confidence=result.confidence,
                    warning=result.warning,
                    tags=list(sample.tags),
                )
            )

        logger.info(
            "PipelineRunner complete: %d/%d samples collected",
            len(records), len(samples),
        )
        return records

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_contexts(self, result) -> List[str]:
        """
        Return a list of context strings from an AnswerResult.

        For the vector path, ``citations`` carry the retrieved chunk text.
        For the structured path, ``evidence`` holds Pydantic model dicts —
        we serialise each dict as a compact key=value string so RAGAS can
        reason over it as plain text.
        """
        # Vector path: use citation text directly
        if result.citations:
            texts = [c.text for c in result.citations if c.text]
            if texts:
                return texts

        # Structured path: serialise evidence dicts
        if result.evidence:
            serialised = []
            for ev in result.evidence:
                if isinstance(ev, dict):
                    serialised.append(
                        " | ".join(f"{k}={v}" for k, v in ev.items())
                    )
                else:
                    serialised.append(str(ev))
            if serialised:
                return serialised

        # Fallback: return the answer itself so RAGAS has *something*
        return [result.answer]
