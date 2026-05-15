import os
from dotenv import load_dotenv
from llama_parse import LlamaParse


def parse_pdf(pdf_path: str, output_dir: str = "parsed_output") -> str:
    load_dotenv()

    api_key = os.getenv("LLAMA_PARSER_KEY")
    if not api_key:
        raise ValueError("LLAMA_PARSER_KEY not found in .env file")

    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown",
        premium_mode=True,
    )

    documents = parser.load_data(pdf_path)

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_file = os.path.join(output_dir, f"{base_name}.md")

    with open(output_file, "w", encoding="utf-8") as f:
        for i, doc in enumerate(documents):
            f.write(f"\n\n--- Page {i + 1} ---\n\n")
            f.write(doc.text)

    return output_file


if __name__ == "__main__":
    output_file = parse_pdf("sample.pdf")
    print("Parsing completed.")
    print(f"Parsed file saved at: {output_file}")