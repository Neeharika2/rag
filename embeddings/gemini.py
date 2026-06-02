import google.generativeai as genai
from typing import List, Optional

from embeddings.base import EmbeddingProvider


class GeminiEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        api_key: str,
        model_name: str = "models/gemini-embedding-2",
        dimension: int = 3072,
    ) -> None:
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY")
        self._model_name = model_name
        self._dimension = dimension
        genai.configure(api_key=api_key)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        response = genai.embed_content(
            model=self._model_name,
            content=texts,
        )
        return response.get("embedding", [])

    def embed_query(self, text: str) -> List[float]:
        response = genai.embed_content(
            model=self._model_name,
            content=text,
        )
        return response.get("embedding", [])
