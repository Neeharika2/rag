"""
evaluation/result_store.py
---------------------------
Persists EvalReport objects to:
  1. A timestamped JSON file under ``evaluation/results/``
  2. An ``eval_runs`` summary table in the existing SQLite metadata DB

Both sinks are optional — pass ``save_json=False`` or ``save_db=False`` to
disable either one.

Usage
-----
    from evaluation.result_store import ResultStore
    store = ResultStore(results_dir="evaluation/results", db_url="sqlite:///./metadata.db")
    path = store.save(report)
    print(f"Report saved to {path}")
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ResultStore:
    """
    Saves :class:`~evaluation.ragas_evaluator.EvalReport` objects.

    Parameters
    ----------
    results_dir:
        Directory where JSON reports are written.
        Created automatically if it does not exist.
    db_url:
        SQLAlchemy connection URL for the metadata database.
        Pass ``None`` to skip DB persistence.
    """

    def __init__(
        self,
        results_dir: str = "evaluation/results",
        db_url: Optional[str] = None,
    ) -> None:
        self._results_dir = Path(results_dir)
        self._db_url = db_url

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        report,
        save_json: bool = True,
        save_db: bool = True,
    ) -> Optional[Path]:
        """
        Persist *report* to disk and/or database.

        Parameters
        ----------
        report:
            :class:`~evaluation.ragas_evaluator.EvalReport` to save.
        save_json:
            Write a timestamped JSON file to ``results_dir``.
        save_db:
            Append a summary row to the ``eval_runs`` SQLite table.

        Returns
        -------
        Path to the JSON file if ``save_json=True``, else ``None``.
        """
        json_path: Optional[Path] = None

        if save_json:
            json_path = self._write_json(report)

        if save_db and self._db_url:
            try:
                self._write_db(report)
            except Exception as exc:
                logger.warning("Failed to write eval_runs to DB: %s", exc)

        return json_path

    # ------------------------------------------------------------------
    # JSON persistence
    # ------------------------------------------------------------------

    def _write_json(self, report) -> Path:
        self._results_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = f"{ts}_ragas_report.json"
        path = self._results_dir / filename

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2)

        logger.info("RAGAS report saved to %s", path)
        return path

    # ------------------------------------------------------------------
    # SQLite persistence
    # ------------------------------------------------------------------

    def _write_db(self, report) -> None:
        """
        Append one row per metric to an ``eval_runs`` table.
        The table is created automatically if it does not exist.
        """
        try:
            from sqlalchemy import (
                Column,
                Float,
                Integer,
                MetaData,
                String,
                Table,
                create_engine,
                insert,
            )
        except ImportError as exc:
            raise ImportError("sqlalchemy is required for DB persistence.") from exc

        engine = create_engine(self._db_url)
        meta = MetaData()

        eval_runs = Table(
            "eval_runs",
            meta,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("run_at", String, nullable=False),
            Column("model_name", String, nullable=False),
            Column("path", String, nullable=False),   # "vector" or "structured"
            Column("metric", String, nullable=False),
            Column("score", Float, nullable=False),
            Column("num_samples", Integer, nullable=False),
        )
        meta.create_all(engine)

        rows = []
        for metric, score in report.vector_scores.items():
            rows.append({
                "run_at": report.run_at,
                "model_name": report.model_name,
                "path": "vector",
                "metric": metric,
                "score": score,
                "num_samples": report.num_vector_samples,
            })
        for metric, score in report.structured_scores.items():
            rows.append({
                "run_at": report.run_at,
                "model_name": report.model_name,
                "path": "structured",
                "metric": metric,
                "score": score,
                "num_samples": report.num_structured_samples,
            })

        if rows:
            with engine.begin() as conn:
                conn.execute(insert(eval_runs), rows)
            logger.info(
                "Wrote %d eval_runs rows to %s", len(rows), self._db_url
            )

    # ------------------------------------------------------------------
    # Loading historical results
    # ------------------------------------------------------------------

    def list_reports(self):
        """Return a list of all saved JSON report paths, newest first."""
        if not self._results_dir.exists():
            return []
        paths = sorted(
            self._results_dir.glob("*_ragas_report.json"),
            reverse=True,
        )
        return paths

    def load_report(self, path) -> dict:
        """Load and return a previously saved JSON report as a dict."""
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
