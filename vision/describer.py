"""
vision/describer.py
-------------------
Sends a cropped PIL image to Gemini Vision and returns a structured
plain-text description of the figure or chart.

The description is designed to be directly embeddable as text so that
the existing text-embedding pipeline can make chart content searchable.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)

_FIGURE_PROMPT = """\
You are analysing a figure or chart extracted from a document page.

Describe it concisely but completely so that someone who cannot see the image \
can understand its content.

Return your description in this exact structure:

Figure type: <type, e.g. bar chart / pie chart / line graph / table / diagram>
Summary: <one sentence summary of what the figure shows>
Key data points:
- <bullet per important value, label, or data point>
Trends and insights: <main takeaway or trend visible in the figure>
Axes / labels: <x-axis label, y-axis label, legend items if present; "N/A" if none>

Be factual and precise. Do not speculate beyond what is visible.
If the image is a logo, icon, watermark, or decorative element with no data, \
reply with exactly: NOT_A_FIGURE
"""


class FigureDescriber:
    """
    Calls Gemini Vision to produce a text description of a cropped image.

    Parameters
    ----------
    api_key:
        Gemini API key.
    model_name:
        Vision-capable Gemini model, e.g. ``"gemini-2.5-flash"``.
    min_area_px:
        Crops smaller than this area (width × height in pixels) are skipped
        without making an API call.  Acts as a second line of defence after
        the cropper's own area filter.
    """

    _NOT_A_FIGURE_MARKER = "NOT_A_FIGURE"

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash",
        min_area_px: int = 5000,
    ) -> None:
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY for FigureDescriber")
        self._model_name = model_name
        self._min_area_px = min_area_px
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=api_key)
        self._genai = genai

    def describe(self, image: Image.Image) -> Optional[str]:
        """
        Return a plain-text description of *image*, or ``None`` if:

        * the image area is below ``min_area_px``
        * Gemini identifies the image as decorative / not a figure
        * the API call fails (logged as a warning, not raised)

        Parameters
        ----------
        image:
            RGB PIL image (typically a cropped page region).

        Returns
        -------
        str or None
        """
        area = image.width * image.height
        if area < self._min_area_px:
            logger.debug(
                "Skipping description: image too small (%dx%d, area=%d px²)",
                image.width, image.height, area,
            )
            return None

        image_b64 = self._pil_to_b64(image)

        try:
            model = self._genai.GenerativeModel(self._model_name)
            response = model.generate_content(
                [
                    {"mime_type": "image/png", "data": image_b64},
                    _FIGURE_PROMPT,
                ]
            )
            text = (response.text or "").strip()
        except Exception as exc:
            logger.warning("Gemini Vision call failed: %s", exc)
            return None

        if not text or self._NOT_A_FIGURE_MARKER in text:
            logger.debug("Gemini identified image as decorative — skipping")
            return None

        logger.debug("Figure description generated (%d chars)", len(text))
        return text

    @staticmethod
    def _pil_to_b64(image: Image.Image) -> str:
        """Convert a PIL image to a base64-encoded PNG string."""
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
