from typing import Protocol


class AnswerGenerator(Protocol):
    def generate(self, prompt: str) -> str:
        ...
