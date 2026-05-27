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


@dataclass
class Settings:
    qdrant_url: str
    qdrant_collection: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    metadata_db_url: str
    upload_dir: str
    log_dir: str
    top_k: int

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            qdrant_collection=os.getenv("QDRANT_COLLECTION", "enterprise_docs"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
            chunk_size=_get_env_int("CHUNK_SIZE", 700),
            chunk_overlap=_get_env_int("CHUNK_OVERLAP", 120),
            metadata_db_url=os.getenv("METADATA_DB_URL", "sqlite:///./metadata.db"),
            upload_dir=os.getenv("UPLOAD_DIR", "./uploads"),
            log_dir=os.getenv("LOG_DIR", "./logs"),
            top_k=_get_env_int("TOP_K", 5),
        )

    def ensure_dirs(self) -> None:
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
