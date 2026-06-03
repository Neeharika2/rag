import logging
import os
from typing import Optional

from parsing.base import DocumentParser
from parsing.docling_parser import DoclingParser
from parsing.errors import ParseError, ParserExhaustedError
from parsing.fallback import ParserFallbackStrategy, file_size_mb
from parsing.models import ParsedDocument
from parsing.structured_log import track_parse

logger = logging.getLogger(__name__)


class MultiModalParser(DocumentParser):
    def __init__(
        self,
        ocr_enabled: bool = True,
        tesseract_cmd: Optional[str] = None,
    ) -> None:
        self._ocr_enabled = ocr_enabled
        self._fallback = ParserFallbackStrategy(ocr_enabled=ocr_enabled)
        self._docling = DoclingParser(ocr_enabled=True)

    def parse(
        self,
        file_path: str,
        doc_id: str,
        use_ocr: Optional[bool] = None,
    ) -> ParsedDocument:
        ext = os.path.splitext(file_path)[1].lower()
        docling_supported = {
            ".pdf",
            ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff",
            ".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".mp4",
        }
        if ext not in docling_supported:
            from parsing.errors import UnsupportedFormatError
            raise UnsupportedFormatError(file_path, doc_id, ext)

        strategy = self._fallback.select_strategy(file_path)
        should_use_ocr = (
            use_ocr if use_ocr is not None else strategy["use_ocr"]
        )
        size_mb = file_size_mb(file_path)

        logger.info(
            "Parsing %s (ext=%s, mime=%s, size=%.1fMB, ocr=%s, strategy=%s)",
            file_path, ext, strategy["mime_type"], size_mb,
            should_use_ocr, strategy["reason"],
        )

        if ext == ".pdf":
            if strategy.get("prefer_pypdf"):
                logger.info(
                    "Skipping Docling for %s: %s",
                    file_path, strategy["reason"],
                )
                with track_parse(
                    doc_id, file_path, "pypdf",
                    ocr_enabled=self._ocr_enabled,
                    ocr_used=False,
                    mime_type=strategy["mime_type"],
                    file_size_mb=size_mb,
                    strategy_reason=strategy["reason"],
                    fallback_used=None,
                ):
                    return self._docling.parse_fallback_pypdf(file_path, doc_id)

            try:
                with track_parse(
                    doc_id, file_path, "docling",
                    ocr_enabled=self._ocr_enabled,
                    ocr_used=should_use_ocr,
                    mime_type=strategy["mime_type"],
                    file_size_mb=size_mb,
                    strategy_reason=strategy["reason"],
                    fallback_used=None,
                ):
                    result = self._docling.parse(
                        file_path, doc_id, use_ocr=should_use_ocr
                    )
                return result
            except BaseException as docling_err:
                logger.warning(
                    "Docling failed for PDF %s: %s. Falling back to PyPDF.",
                    file_path, docling_err,
                )
                try:
                    with track_parse(
                        doc_id, file_path, "pypdf",
                        ocr_enabled=self._ocr_enabled,
                        ocr_used=False,
                        mime_type=strategy["mime_type"],
                        file_size_mb=size_mb,
                        strategy_reason="docling_fallback",
                        fallback_used="docling->pypdf",
                    ):
                        return self._docling.parse_fallback_pypdf(
                            file_path, doc_id
                        )
                except Exception as pypdf_err:
                    raise ParserExhaustedError(
                        file_path=file_path,
                        doc_id=doc_id,
                        errors=[str(docling_err), str(pypdf_err)],
                    ) from pypdf_err

        try:
            with track_parse(
                doc_id, file_path, "docling",
                ocr_enabled=self._ocr_enabled,
                ocr_used=should_use_ocr,
                mime_type=strategy["mime_type"],
                file_size_mb=size_mb,
                strategy_reason=strategy["reason"],
                fallback_used=None,
            ):
                return self._docling.parse(
                    file_path, doc_id, use_ocr=should_use_ocr
                )
        except BaseException as exc:
            raise ParseError(
                f"Docling failed for {ext} file: {exc}",
                file_path=file_path,
                doc_id=doc_id,
                parser="docling",
                cause=exc,
            ) from exc
