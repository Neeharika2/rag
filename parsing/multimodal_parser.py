import os
from typing import Optional

from parsing.base import DocumentParser
from parsing.docling_parser import DoclingParser
from parsing.models import ParsedDocument


class MultiModalParser(DocumentParser):
    def __init__(
        self,
        ocr_enabled: bool = True,
        tesseract_cmd: Optional[str] = None,
    ) -> None:
        self._docling = DoclingParser(ocr_enabled=ocr_enabled)

    def parse(self, file_path: str, doc_id: str) -> ParsedDocument:
        ext = os.path.splitext(file_path)[1].lower()
        docling_supported = {
            ".pdf",
            ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff",
            ".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".mp4"
        }
        if ext in docling_supported:
            return self._docling.parse(file_path, doc_id)
        raise ValueError(f"Unsupported file format: {ext}")
