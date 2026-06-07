"""
vision/enricher.py
------------------
Orchestrates per-document image enrichment:
  ParsedDocument → Dict[page_number, List[description_str]]

For each page that has image provenance, it:
1. Calls :func:`~vision.cropper.crop_image_regions` to get PIL crops.
2. Calls :class:`~vision.describer.FigureDescriber` on each crop.
3. Collects descriptions keyed by page number.

The result is consumed by :class:`~chunking.recursive.RecursiveChunker`
to append figure descriptions to the appropriate text chunks.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from parsing.models import ParsedDocument

from vision.cropper import crop_image_regions
from vision.describer import FigureDescriber

logger = logging.getLogger(__name__)


class FigureEnricher:
    """
    Enriches a parsed PDF document with Gemini Vision descriptions.

    Parameters
    ----------
    api_key:
        Gemini API key (reused from main settings).
    model_name:
        Vision model to use (defaults to ``"gemini-2.5-flash"``).
    min_area_px:
        Minimum pixel area for a crop to be sent to Vision.  Shared between
        the cropper (to skip degenerate regions) and the describer (second
        filter).
    max_images_per_doc:
        Safety cap on Vision API calls per document.  ``None`` means no cap.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash",
        min_area_px: int = 5000,
        max_images_per_doc: Optional[int] = None,
    ) -> None:
        self._describer = FigureDescriber(
            api_key=api_key,
            model_name=model_name,
            min_area_px=min_area_px,
        )
        self._min_area_px = min_area_px
        self._max_images_per_doc = max_images_per_doc

    def enrich(
        self,
        pdf_path: str,
        parsed: "ParsedDocument",
    ) -> Dict[int, List[str]]:
        """
        Return a mapping of ``page_number → [description, ...]`` for all
        figures found in *parsed*.

        Only PDF files are supported.  For other file types (images, audio)
        the method returns an empty dict immediately.

        Parameters
        ----------
        pdf_path:
            Absolute path to the source PDF.
        parsed:
            :class:`~parsing.models.ParsedDocument` produced by the parser.
        """
        result: Dict[int, List[str]] = {}

        if not pdf_path.lower().endswith(".pdf"):
            return result

        if not parsed.provenance:
            logger.debug("No provenance data for %s — skipping enrichment", pdf_path)
            return result

        total_described = 0

        for page_no, page_prov in parsed.provenance.items():
            if not page_prov.images:
                continue

            logger.debug(
                "Processing %d image marker(s) on page %d of %s",
                len(page_prov.images), page_no, pdf_path,
            )

            try:
                crops = crop_image_regions(
                    pdf_path=pdf_path,
                    page_number=page_no,
                    image_markers=page_prov.images,
                    min_area_px=self._min_area_px,
                )
            except Exception as exc:
                logger.warning(
                    "Cropping failed for page %d of %s: %s",
                    page_no, pdf_path, exc,
                )
                continue

            page_descriptions: List[str] = []
            for crop in crops:
                if (
                    self._max_images_per_doc is not None
                    and total_described >= self._max_images_per_doc
                ):
                    logger.info(
                        "Reached max_images_per_doc=%d for %s — stopping",
                        self._max_images_per_doc, pdf_path,
                    )
                    break

                description = self._describer.describe(crop)
                if description:
                    page_descriptions.append(description)
                    total_described += 1
                    logger.info(
                        "Described figure on page %d of %s (%d chars)",
                        page_no, pdf_path, len(description),
                    )

            if page_descriptions:
                result[page_no] = page_descriptions

        logger.info(
            "Enrichment complete for %s: %d figure description(s) across %d page(s)",
            pdf_path, total_described, len(result),
        )
        return result
