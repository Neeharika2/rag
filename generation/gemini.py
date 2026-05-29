from typing import Optional

import google.generativeai as genai


class GeminiGenerator:
    def __init__(self, api_key: str, model_name: str) -> None:
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY")
        self._model_name = model_name
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model_name)

    def generate(self, prompt: str) -> str:
        response = self._model.generate_content(prompt)
        text: Optional[str] = getattr(response, "text", None)
        return (text or "").strip()
