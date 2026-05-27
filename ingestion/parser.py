import os

from dotenv import load_dotenv
from llama_parse import LlamaParse

from config import PARSED_DIR


def parse_pdf(pdf_path: str) -> str:
    load_dotenv()

    os.makedirs(PARSED_DIR, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_file = os.path.join(PARSED_DIR, f"{base_name}.md")

    if os.path.exists(output_file):
        print(f"  Skipping parse (already exists): {output_file}")
        return output_file

    api_key = os.getenv("LLAMA_PARSER_KEY")
    if not api_key:
        raise ValueError("LLAMA_PARSER_KEY not found in .env file")

    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown",
        premium_mode=True,
    )

    documents = parser.load_data(pdf_path)

    with open(output_file, "w", encoding="utf-8") as f:
        for i, doc in enumerate(documents):
            f.write(f"\n\n--- Page {i + 1} ---\n\n")
            f.write(doc.text)

    return output_file