from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class Citation(BaseModel):
    chunk_id: str
    doc_id: Optional[str] = None
    score: float = 0.0
    text: str = ""
    metadata: Dict[str, Any] = {}


class AnswerResult(BaseModel):
    answer: str
    route: str
    confidence: float
    citations: List[Citation] = []
    evidence: Optional[List[Dict[str, Any]]] = None
    fallback_reason: Optional[str] = None
    warning: Optional[str] = None
    rewritten_query: Optional[str] = None
    detected_company: Optional[str] = None
    detected_metric: Optional[str] = None
