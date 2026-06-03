import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from parsing.models import PageContent


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    page_start: Optional[int]
    page_end: Optional[int]
    metadata: Dict[str, Any]


class RecursiveChunker:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", " ", ""]
        import tiktoken
        self._encoder = tiktoken.get_encoding("cl100k_base")

    def chunk_pages(
        self,
        doc_id: str,
        pages: Iterable[PageContent],
        base_metadata: Dict[str, Any],
    ) -> List[Chunk]:
        chunks: List[Chunk] = []
        chunk_index = 0

        for page in pages:
            text = page.text.strip()
            if not text:
                continue

            split_chunks = self._split_text(text)
            split_chunks = self._apply_overlap(split_chunks)
            for part in split_chunks:
                chunk_id = f"{doc_id}_chunk_{chunk_index}_{uuid.uuid4().hex[:8]}"
                metadata = dict(base_metadata)
                if page.page_number is not None:
                    metadata["page"] = str(page.page_number)

                if page.provenance is not None:
                    provenance_dict: Dict[str, Any] = {
                        "page_number": page.provenance.page_number,
                    }
                    if page.provenance.bbox is not None:
                        provenance_dict["bbox"] = {
                            "left": page.provenance.bbox.left,
                            "top": page.provenance.bbox.top,
                            "right": page.provenance.bbox.right,
                            "bottom": page.provenance.bbox.bottom,
                        }
                    if page.provenance.tables:
                        provenance_dict["tables"] = [
                            {
                                "page_number": t.page_number,
                                "bbox": {
                                    "left": t.bbox.left,
                                    "top": t.bbox.top,
                                    "right": t.bbox.right,
                                    "bottom": t.bbox.bottom,
                                },
                            }
                            for t in page.provenance.tables
                        ]
                    if page.provenance.images:
                        provenance_dict["images"] = [
                            {
                                "page_number": im.page_number,
                                "bbox": {
                                    "left": im.bbox.left,
                                    "top": im.bbox.top,
                                    "right": im.bbox.right,
                                    "bottom": im.bbox.bottom,
                                },
                            }
                            for im in page.provenance.images
                        ]
                    metadata["provenance"] = provenance_dict

                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        text=part,
                        page_start=page.page_number,
                        page_end=page.page_number,
                        metadata=metadata,
                    )
                )
                chunk_index += 1

        return chunks

    def _split_text(self, text: str) -> List[str]:
        return self._split_recursive(text, self.separators)

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        if self._estimate_tokens(text) <= self.chunk_size:
            return [text]

        if not separators:
            return [text]

        separator = separators[0]
        if separator == "":
            return self._split_by_tokens(text)

        parts = text.split(separator)
        if len(parts) == 1:
            return self._split_recursive(text, separators[1:])

        chunks: List[str] = []
        current = ""
        for part in parts:
            candidate = part if not current else current + separator + part
            if self._estimate_tokens(candidate) <= self.chunk_size:
                current = candidate
                continue

            if current:
                chunks.append(current)

            if self._estimate_tokens(part) > self.chunk_size:
                chunks.extend(self._split_recursive(part, separators[1:]))
                current = ""
            else:
                current = part

        if current:
            chunks.append(current)

        return chunks

    def _split_by_tokens(self, text: str) -> List[str]:
        tokens = self._tokenize(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunks.append(self._detokenize(chunk_tokens))
            start = end
        return chunks

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        if self.chunk_overlap <= 0 or len(chunks) <= 1:
            return chunks

        overlapped = [chunks[0]]
        for idx in range(1, len(chunks)):
            prev_tokens = self._tokenize(chunks[idx - 1])
            overlap_tokens = prev_tokens[-self.chunk_overlap:]
            merged = self._detokenize(overlap_tokens) + " " + chunks[idx]
            overlapped.append(merged.strip())
        return overlapped

    def _estimate_tokens(self, text: str) -> int:
        return len(self._tokenize(text))

    def _tokenize(self, text: str) -> List[int]:
        return self._encoder.encode(text, disallowed_special=())

    def _detokenize(self, tokens: List[int]) -> str:
        return self._encoder.decode(tokens)
