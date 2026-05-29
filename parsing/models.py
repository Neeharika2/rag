from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PageContent:
    page_number: Optional[int]
    text: str


@dataclass
class ParsedDocument:
    doc_id: str
    source_path: str
    source_name: str
    pages: List[PageContent]
    raw_markdown: str
