from typing import Optional
from generation.gemini import GeminiGenerator


class QueryRewriter:
    """Agent that reformulates search queries to optimize retrieval quality."""

    def __init__(self, generator: GeminiGenerator) -> None:
        self._generator = generator

    def rewrite(self, query: str) -> str:
        """
        Rewrites a conversational query into a search-optimized keyword-rich query.
        Returns the original query if rewriting fails or produces empty output.
        """
        if not query or not query.strip():
            return query

        prompt = (
            "You are an expert search query optimizer for a RAG (Retrieval-Augmented Generation) system.\n"
            "Your task is to analyze the user's input query and rewrite it to be optimized for vector search "
            "and keyword matching. Focus on the core semantic concepts, key terminology, and entities.\n\n"
            "Rules:\n"
            "1. Remove conversational filler (e.g., 'can you find', 'tell me about', 'please search for').\n"
            "2. Retain all key technical terms, dates, product names, and core topics.\n"
            "3. If the query is already concise and search-optimized, keep it as is.\n"
            "4. Do NOT try to answer the query.\n"
            "5. Respond ONLY with the rewritten search query. Do not include any explanations, introductory text, or quotation marks.\n\n"
            f"Original conversational query: \"{query}\"\n\n"
            "Optimized search query:"
        )

        try:
            rewritten = self._generator.generate(prompt)
            rewritten_clean = rewritten.strip().strip('"').strip("'")
            if rewritten_clean:
                return rewritten_clean
        except Exception:
            # Silently fall back to the original query if there is any LLM error
            pass

        return query
