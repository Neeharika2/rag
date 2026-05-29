from abc import ABC, abstractmethod

from parsing.models import ParsedDocument


class DocumentParser(ABC):
    @abstractmethod
    def parse(self, file_path: str, doc_id: str) -> ParsedDocument:
        raise NotImplementedError
