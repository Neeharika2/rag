import os
from typing import List

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

_DEFAULT_MODEL = "gemini-2.5-flash"


def _configure() -> None:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment")
    genai.configure(api_key=api_key)


def generate_answer(query: str, context_docs: List[str], model_name: str = _DEFAULT_MODEL) -> str:
    _configure()
    model = genai.GenerativeModel(model_name)

    context = "\n\n---\n\n".join(context_docs)
    prompt = (
        "You are a helpful assistant. Use the following retrieved context to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    response = model.generate_content(prompt)
    return response.text.strip()
