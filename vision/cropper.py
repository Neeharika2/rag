"""
vision/cropper.py
-----------------
Crops image/chart regions from a PDF page using PyMuPDF (fitz).

PyMuPDF renders each page at a configurable DPI into a PIL image, then
PIL crops the bounding-box rectangles reported by Docling provenance.

BBox coordinates from Docling are in *PDF points* (72 pt = 1 inch),
measured from the top-left of the page.  PyMuPDF uses the same coordinate
system, so the conversion is straightforward.
"""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

from PIL import Image

if TYPE_CHECKING:
    from parsing.models import ImageMarker

logger = logging.getLogger(__name__)

# Default rendering resolution.  150 DPI gives sharp crops while keeping
# memory usage reasonable (~3–4 MB per A4 page).
_DEFAULT_DPI = 150


def _open_pdf(pdf_path: str):
    """Return a fitz.Document; raises ImportError if pymupdf is missing."""
    try:
        import fitz  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pymupdf is required for image cropping. "
            "Install it with: pip install pymupdf"
        ) from exc
    return fitz.open(pdf_path)


def _bbox_to_pixel_rect(
    bbox_pts: Tuple[float, float, float, float],
    page_width_pts: float,
    page_height_pts: float,
    img_width_px: int,
    img_height_px: int,
) -> Tuple[int, int, int, int]:
    """Convert PDF-point bbox to pixel coordinates in the rasterised image."""
    left_pt, top_pt, right_pt, bottom_pt = bbox_pts
    scale_x = img_width_px / page_width_pts
    scale_y = img_height_px / page_height_pts

    left_px = max(0, int(left_pt * scale_x))
    top_px = max(0, int(top_pt * scale_y))
    right_px = min(img_width_px, int(right_pt * scale_x))
    bottom_px = min(img_height_px, int(bottom_pt * scale_y))
    return left_px, top_px, right_px, bottom_px


def crop_image_regions(
    pdf_path: str,
    page_number: int,
    image_markers: List["ImageMarker"],
    min_area_px: int = 5000,
    dpi: int = _DEFAULT_DPI,
) -> List[Image.Image]:
    """
    Render *page_number* (1-indexed) of *pdf_path* and crop each bounding
    box listed in *image_markers*.

    Parameters
    ----------
    pdf_path:
        Absolute path to the PDF file.
    page_number:
        1-based page index matching Docling provenance page numbers.
    image_markers:
        List of :class:`~parsing.models.ImageMarker` objects for this page.
    min_area_px:
        Crops whose pixel area is below this threshold are silently skipped
        (removes logos, watermarks, tiny decorative icons).
    dpi:
        Rendering resolution.  Higher values give sharper crops but use more
        memory and are slower.

    Returns
    -------
    List of PIL images, one per qualifying bounding box (may be empty).
    """
    if not image_markers:
        return []

    doc = _open_pdf(pdf_path)
    try:
        # fitz pages are 0-indexed
        fitz_index = page_number - 1
        if fitz_index < 0 or fitz_index >= len(doc):
            logger.warning(
                "Page %d out of range for %s (total pages: %d)",
                page_number, pdf_path, len(doc),
            )
            return []

        page = doc[fitz_index]
        page_rect = page.rect  # fitz Rect in PDF points

        # Render the whole page to a pixmap then convert to PIL
        matrix = __import__("fitz").Matrix(dpi / 72, dpi / 72)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        img_bytes = pixmap.tobytes("png")
        page_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_w, img_h = page_img.size

        crops: List[Image.Image] = []
        for marker in image_markers:
            bbox = marker.bbox
            if bbox is None:
                continue

            # Docling bbox: left, top, right, bottom in PDF points
            left_px, top_px, right_px, bottom_px = _bbox_to_pixel_rect(
                (bbox.left, bbox.top, bbox.right, bbox.bottom),
                page_rect.width,
                page_rect.height,
                img_w,
                img_h,
            )

            area = (right_px - left_px) * (bottom_px - top_px)
            if area < min_area_px:
                logger.debug(
                    "Skipping small image on page %d (area=%d px²)",
                    page_number, area,
                )
                continue

            if right_px <= left_px or bottom_px <= top_px:
                logger.debug(
                    "Degenerate bbox on page %d, skipping", page_number
                )
                continue

            crop = page_img.crop((left_px, top_px, right_px, bottom_px))
            crops.append(crop)
            logger.debug(
                "Cropped image on page %d: %dx%d px (area=%d)",
                page_number, crop.width, crop.height, area,
            )

        return crops
    finally:
        doc.close()
