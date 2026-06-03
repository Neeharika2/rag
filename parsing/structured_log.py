import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_PARSE_EVENTS: List[Dict[str, Any]] = []


def configure_parse_logging(log_dir: str, level: str = "INFO") -> None:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "parse_events.jsonl")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    file_handler.setFormatter(logging.Formatter("%(message)s"))

    logger = logging.getLogger("rag.parse")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    logger.addHandler(stream_handler)


@dataclass
class ParseEvent:
    event: str = "parse"
    doc_id: str = ""
    file_path: str = ""
    parser: str = ""
    duration_ms: float = 0.0
    status: str = "success"
    mime_type: Optional[str] = None
    file_size_mb: Optional[float] = None
    ocr_enabled: Optional[bool] = None
    ocr_used: Optional[bool] = None
    pages: Optional[int] = None
    chunks: Optional[int] = None
    error: Optional[str] = None
    fallback_used: Optional[str] = None
    strategy_reason: Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    def log(self) -> None:
        record = self.to_dict()
        logger = logging.getLogger("rag.parse")
        logger.info(json.dumps(record, default=str))
        _PARSE_EVENTS.append(record)


@contextmanager
def track_parse(
    doc_id: str,
    file_path: str,
    parser_name: str,
    **extra: Any,
):
    start = time.perf_counter()
    status = "success"
    error_msg: Optional[str] = None
    try:
        yield
    except Exception as e:
        status = "error"
        error_msg = f"{type(e).__name__}: {e}"
        raise
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        event = ParseEvent(
            event="parse",
            doc_id=doc_id,
            file_path=file_path,
            parser=parser_name,
            duration_ms=round(duration_ms, 2),
            status=status,
            error=error_msg,
            **extra,
        )
        event.log()


def get_parse_events() -> List[Dict[str, Any]]:
    return list(_PARSE_EVENTS)
