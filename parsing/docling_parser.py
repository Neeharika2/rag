import os
import re
from typing import List

from docling.document_converter import DocumentConverter, PdfFormatOption, ImageFormatOption, AudioFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, AsrPipelineOptions
from docling.pipeline.asr_pipeline import AsrPipeline
from docling.datamodel import asr_model_specs

from parsing.models import PageContent, ParsedDocument


class DoclingParser:
    def __init__(self, ocr_enabled: bool = True) -> None:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = ocr_enabled
        pipeline_options.do_table_structure = ocr_enabled
        pipeline_options.images_scale = 1
        pipeline_options.generate_page_images = False

        asr_pipeline_options = AsrPipelineOptions()
        asr_pipeline_options.asr_options = asr_model_specs.WHISPER_TURBO

        self._converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF, InputFormat.IMAGE, InputFormat.AUDIO],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),
                InputFormat.AUDIO: AudioFormatOption(
                    pipeline_cls=AsrPipeline,
                    pipeline_options=asr_pipeline_options,
                ),
            },
        )

    def parse(self, file_path: str, doc_id: str) -> ParsedDocument:
        ext = os.path.splitext(file_path)[1].lower()
        supported = {
            # PDF
            ".pdf",
            # Image
            ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff",
            # Audio / Video
            ".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".mp4"
        }
        if ext not in supported:
            raise ValueError(f"Docling parser does not support {ext} files")
        try:
            result = self._converter.convert(file_path)
            document = result.document
            markdown = document.export_to_markdown(page_break_placeholder="<!-- PAGE_BREAK -->")
            pages = self._extract_pages(document, markdown)
        except BaseException as e:
            if ext == ".pdf":
                print(f"Docling failed or ran out of memory: {e}. Falling back to PyPDF...")
                import pypdf
                pages = []
                with open(file_path, "rb") as f:
                    reader = pypdf.PdfReader(f)
                    for index, page in enumerate(reader.pages):
                        text = page.extract_text()
                        if text and text.strip():
                            pages.append(PageContent(page_number=index + 1, text=text.strip()))
                markdown = "\n\n<!-- PAGE_BREAK -->\n\n".join([p.text for p in pages])
            else:
                raise e

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
        page_chunks = markdown.split("<!-- PAGE_BREAK -->")
        for index, text in enumerate(page_chunks):
            text_str = text.strip()
            if text_str:
                pages.append(PageContent(page_number=index + 1, text=text_str))
        return pages
