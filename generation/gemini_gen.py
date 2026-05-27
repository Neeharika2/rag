import os
from typing import List

import google.generativeai as genai
from dotenv import load_dotenv

from config import GEMINI_MODEL

load_dotenv()


def _configure() -> None:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment")
    genai.configure(api_key=api_key)


def generate_answer(query: str, context_docs: List[str]) -> str:
    _configure()
    model = genai.GenerativeModel(GEMINI_MODEL)

    context = "\n\n---\n\n".join(context_docs)
    prompt = (
        "You are a helpful assistant that answers questions about government budgets and policy documents. "
        "Use the following retrieved context to answer the question. "
        "If the context does not contain enough information, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    response = model.generate_content(prompt)
    return response.text.strip()