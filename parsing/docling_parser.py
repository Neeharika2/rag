import os
import re
from dataclasses import dataclass
from typing import List, Optional

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions


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


class DoclingParser:
    def __init__(self) -> None:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = True
        pipeline_options.images_scale = 1
        pipeline_options.generate_page_images = False

        self._converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            },
        )

    def parse(self, file_path: str, doc_id: str) -> ParsedDocument:
        if not file_path.lower().endswith(".pdf"):
            raise ValueError("Docling parser currently supports PDF files only")
        result = self._converter.convert(file_path)
        document = result.document
        markdown = document.export_to_markdown()

        pages = self._extract_pages(document, markdown)
        if not pages:
            pages = [PageContent(page_number=1, text=markdown)]

        return ParsedDocument(
            doc_id=doc_id,
            source_path=file_path,
            source_name=os.path.basename(file_path),
            pages=pages,
            raw_markdown=markdown,
        )

    def _extract_pages(self, document, markdown: str) -> List[PageContent]:
        pages: List[PageContent] = []

        if hasattr(document, "pages"):
            for index, page in enumerate(document.pages):
                text = getattr(page, "text", None)
                if not text and hasattr(page, "export_to_text"):
                    text = page.export_to_text()
                if not text and hasattr(page, "get_text"):
                    text = page.get_text()
                if text and text.strip():
                    pages.append(PageContent(page_number=index + 1, text=text))

        if pages:
            return pages

        page_matches = re.split(r"--- Page (\d+) ---", markdown)
        if len(page_matches) > 1:
            for i in range(1, len(page_matches), 2):
                page_number = int(page_matches[i])
                text = page_matches[i + 1].strip()
                if text:
                    pages.append(PageContent(page_number=page_number, text=text))

        return pages
