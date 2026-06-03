from typing import List, Optional


class ParseError(Exception):
    def __init__(
        self,
        message: str,
        file_path: str,
        doc_id: str,
        parser: str,
        cause: Optional[Exception] = None,
    ) -> None:
        self.file_path = file_path
        self.doc_id = doc_id
        self.parser = parser
        self.cause = cause
        super().__init__(message)


class UnsupportedFormatError(ParseError):
    def __init__(self, file_path: str, doc_id: str, ext: str) -> None:
        self.ext = ext
        super().__init__(
            f"Unsupported file format: {ext}",
            file_path=file_path,
            doc_id=doc_id,
            parser="mime_detection",
        )


class ParserExhaustedError(ParseError):
    def __init__(self, file_path: str, doc_id: str, errors: List[str]) -> None:
        self.errors = errors
        super().__init__(
            f"All parsers in fallback chain failed for {file_path}: {'; '.join(errors)}",
            file_path=file_path,
            doc_id=doc_id,
            parser="fallback_chain",
        )
