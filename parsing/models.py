from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BBox:
    left: float
    top: float
    right: float
    bottom: float


@dataclass
class TableMarker:
    page_number: int
    bbox: BBox
    caption: Optional[str] = None


@dataclass
class ImageMarker:
    page_number: int
    bbox: BBox
    caption: Optional[str] = None


@dataclass
class PageProvenance:
    page_number: int
    bbox: Optional[BBox] = None
    tables: List[TableMarker] = field(default_factory=list)
    images: List[ImageMarker] = field(default_factory=list)


@dataclass
class PageContent:
    page_number: Optional[int]
    text: str
    provenance: Optional[PageProvenance] = None


@dataclass
class ParsedDocument:
    doc_id: str
    source_path: str
    source_name: str
    pages: List[PageContent]
    raw_markdown: str
    provenance: Optional[Dict[int, PageProvenance]] = None
