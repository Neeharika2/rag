from abc import ABC, abstractmethod
from typing import Optional

from parsing.models import ParsedDocument


class DocumentParser(ABC):
    @abstractmethod
    def parse(
        self,
        file_path: str,
        doc_id: str,
        use_ocr: Optional[bool] = None,
    ) -> ParsedDocument:
        raise NotImplementedError
