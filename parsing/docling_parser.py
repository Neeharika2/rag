import logging
import os
from typing import Dict, List, Optional

from docling.document_converter import (
    AudioFormatOption,
    DocumentConverter,
    ImageFormatOption,
    PdfFormatOption,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import AsrPipelineOptions, PdfPipelineOptions
from docling.pipeline.asr_pipeline import AsrPipeline
from docling.datamodel import asr_model_specs

from parsing.models import (
    BBox,
    ImageMarker,
    PageContent,
    PageProvenance,
    ParsedDocument,
    TableMarker,
)
from parsing.structured_log import track_parse

logger = logging.getLogger(__name__)


class DoclingParser:
    def __init__(self, ocr_enabled: bool = True) -> None:
        self._ocr_enabled = ocr_enabled
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

    def parse(
        self,
        file_path: str,
        doc_id: str,
        use_ocr: Optional[bool] = None,
    ) -> ParsedDocument:
        ext = os.path.splitext(file_path)[1].lower()
        supported = {
            ".pdf",
            ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff",
            ".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".mp4",
        }
        if ext not in supported:
            raise ValueError(f"Docling parser does not support {ext} files")

        ocr_flag = use_ocr if use_ocr is not None else self._ocr_enabled
        if ocr_flag and not self._ocr_enabled:
            logger.warning(
                "use_ocr=True but parser initialized with ocr_enabled=False for %s",
                file_path,
            )

        with track_parse(
            doc_id, file_path, "docling",
            ocr_enabled=ocr_flag,
            fallback_used=None,
        ):
            result = self._converter.convert(file_path)
            document = result.document
            markdown = document.export_to_markdown(
                page_break_placeholder="<!-- PAGE_BREAK -->"
            )
            provenance = self._extract_provenance(document)
            pages = self._extract_pages(markdown, provenance)

        if not pages:
            pages = [PageContent(page_number=1, text=markdown)]

        return ParsedDocument(
            doc_id=doc_id,
            source_path=file_path,
            source_name=os.path.basename(file_path),
            pages=pages,
            raw_markdown=markdown,
            provenance=provenance,
        )

    def parse_fallback_pypdf(self, file_path: str, doc_id: str) -> ParsedDocument:
        import pypdf

        pages_list: List[PageContent] = []
        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for index, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    pages_list.append(
                        PageContent(page_number=index + 1, text=text.strip())
                    )

        markdown = "\n\n<!-- PAGE_BREAK -->\n\n".join(
            [p.text for p in pages_list]
        )
        if not pages_list:
            pages_list = [PageContent(page_number=1, text=markdown)]

        return ParsedDocument(
            doc_id=doc_id,
            source_path=file_path,
            source_name=os.path.basename(file_path),
            pages=pages_list,
            raw_markdown=markdown,
        )

    def _extract_provenance(self, document) -> Dict[int, PageProvenance]:
        provenance: Dict[int, PageProvenance] = {}
        try:
            pages = getattr(document, "pages", None)
            if pages is None:
                return provenance
            for page_no, page in pages.items():
                tables: List[TableMarker] = []
                images: List[ImageMarker] = []
                items = getattr(page, "items", []) or []
                for item in items:
                    bbox = self._extract_bbox(item)
                    if bbox is None:
                        continue
                    label = str(getattr(item, "label", "")).lower()
                    if "table" in label:
                        tables.append(
                            TableMarker(page_number=page_no, bbox=bbox)
                        )
                    elif any(
                        x in label
                        for x in ("picture", "figure", "chart", "image")
                    ):
                        images.append(
                            ImageMarker(page_number=page_no, bbox=bbox)
                        )

                page_bbox = self._extract_bbox(page)
                provenance[page_no] = PageProvenance(
                    page_number=page_no,
                    bbox=page_bbox,
                    tables=tables,
                    images=images,
                )
        except Exception:
            logger.warning(
                "Failed to extract document provenance from %s",
                getattr(document, "name", "unknown"),
                exc_info=True,
            )
        return provenance

    def _extract_bbox(self, obj):
        bbox = getattr(obj, "bbox", None)
        if bbox is None:
            return None
        try:
            return BBox(
                left=float(getattr(bbox, "l", 0)),
                top=float(getattr(bbox, "t", 0)),
                right=float(getattr(bbox, "r", 0)),
                bottom=float(getattr(bbox, "b", 0)),
            )
        except (TypeError, ValueError):
            return None

    def _extract_pages(
        self,
        markdown: str,
        provenance: Dict[int, PageProvenance],
    ) -> List[PageContent]:
        pages: List[PageContent] = []
        page_chunks = markdown.split("<!-- PAGE_BREAK -->")
        for index, text in enumerate(page_chunks):
            text_str = text.strip()
            if text_str:
                page_no = index + 1
                page_prov = provenance.get(page_no)
                pages.append(
                    PageContent(
                        page_number=page_no,
                        text=text_str,
                        provenance=page_prov,
                    )
                )
        return pages
