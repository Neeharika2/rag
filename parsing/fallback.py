import logging
import os
from typing import Dict

logger = logging.getLogger(__name__)

_MAGIC_SIGNATURES: Dict[bytes, str] = {
    b"%PDF": "application/pdf",
    b"\x89PNG": "image/png",
    b"\xff\xd8": "image/jpeg",
    b"GIF8": "image/gif",
    b"BM": "image/bmp",
    b"II*\x00": "image/tiff",
    b"MM\x00*": "image/tiff",
}

_EXT_MAP: Dict[str, str] = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".mp4": "video/mp4",
}


def detect_mime_type(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    mime_fallback = _EXT_MAP.get(ext, "application/octet-stream")
    try:
        with open(file_path, "rb") as f:
            header = f.read(64)
        for sig, mime in _MAGIC_SIGNATURES.items():
            if header.startswith(sig):
                return mime
        if header.startswith(b"%PDF-"):
            return "application/pdf"
    except (OSError, IOError) as exc:
        logger.warning("MIME detection failed for %s: %s", file_path, exc)
    return mime_fallback


def estimate_text_ratio(file_path: str, sample_bytes: int = 65536) -> float:
    try:
        with open(file_path, "rb") as f:
            data = f.read(sample_bytes)
        if not data:
            return 0.0
        printable = sum(
            1 for b in data if 32 <= b <= 126 or b in (9, 10, 13)
        )
        return printable / len(data)
    except (OSError, IOError):
        return 0.0


def is_text_only_pdf(file_path: str, threshold: float = 0.3) -> bool:
    mime = detect_mime_type(file_path)
    if mime != "application/pdf":
        return False
    ratio = estimate_text_ratio(file_path)
    return ratio >= threshold


def file_size_mb(file_path: str) -> float:
    try:
        return os.path.getsize(file_path) / (1024 * 1024)
    except (OSError, IOError):
        return 0.0


class ParserFallbackStrategy:
    def __init__(self, ocr_enabled: bool = False) -> None:
        self.ocr_enabled = ocr_enabled

    def select_strategy(self, file_path: str) -> dict:
        ext = os.path.splitext(file_path)[1].lower()
        size_mb = file_size_mb(file_path)
        mime = detect_mime_type(file_path)

        strategy = {
            "mime_type": mime,
            "size_mb": round(size_mb, 2),
            "use_ocr": False,
            "skip_docling": False,
            "prefer_pypdf": False,
            "reason": "",
        }

        if mime.startswith("audio/") or ext in (
            ".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".mp4"
        ):
            strategy["reason"] = "Audio file; ASR pipeline"
            return strategy

        if mime.startswith("image/"):
            strategy["use_ocr"] = self.ocr_enabled
            strategy["reason"] = (
                "Image file; OCR enabled" if self.ocr_enabled
                else "Image file; OCR disabled"
            )
            return strategy

        if mime == "application/pdf":
            if "placement" in file_path.lower() or "placement" in os.path.basename(file_path).lower():
                strategy["use_ocr"] = False
                strategy["skip_docling"] = False
                strategy["prefer_pypdf"] = False
                strategy["reason"] = "Placement PDF; forcing Docling to preserve table structures"
                return strategy

            if self.ocr_enabled and is_text_only_pdf(file_path):
                strategy["use_ocr"] = False
                strategy["skip_docling"] = True
                strategy["prefer_pypdf"] = True
                strategy["reason"] = (
                    "OCR enabled but PDF is text-only (text_ratio >= 0.3); "
                    "skipping OCR, using PyPDF"
                )
            elif self.ocr_enabled:
                strategy["use_ocr"] = True
                strategy["reason"] = "OCR enabled for scanned PDF"
            else:
                strategy["prefer_pypdf"] = True
                strategy["reason"] = "OCR disabled; using PyPDF text extraction"

            if size_mb > 100:
                strategy["reason"] += (
                    f"; large file ({size_mb:.1f}MB)"
                )
            return strategy

        strategy["reason"] = f"Unknown mime type '{mime}'; attempting Docling"
        return strategy
