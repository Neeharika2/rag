import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class Settings:
    chroma_path: str
    chroma_collection: str
    gemini_api_key: str
    gemini_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    metadata_db_url: str
    upload_dir: str
    log_dir: str
    top_k: int
    ocr_enabled: bool
    tesseract_cmd: Optional[str]
    rewrite_query_by_default: bool

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            chroma_path=os.getenv("CHROMA_PATH", "./chroma"),
            chroma_collection=os.getenv("CHROMA_COLLECTION", "enterprise_docs"),
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-2"),
            chunk_size=_get_env_int("CHUNK_SIZE", 1500),
            chunk_overlap=_get_env_int("CHUNK_OVERLAP", 200),
            metadata_db_url=os.getenv("METADATA_DB_URL", "sqlite:///./metadata.db"),
            upload_dir=os.getenv("UPLOAD_DIR", "./uploads"),
            log_dir=os.getenv("LOG_DIR", "./logs"),
            top_k=_get_env_int("TOP_K", 5),
            ocr_enabled=_get_env_bool("OCR_ENABLED", False),
            tesseract_cmd=os.getenv("TESSERACT_CMD"),
            rewrite_query_by_default=_get_env_bool("REWRITE_QUERY_BY_DEFAULT", False),
        )

    def ensure_dirs(self) -> None:
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.chroma_path, exist_ok=True)
