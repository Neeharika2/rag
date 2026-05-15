import os
from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_file(
    input_file: str,
    output_dir: str = "chunks",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> str:
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = text_splitter.split_text(text)

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "chunks.txt")

    with open(output_file, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"\n\n--- Chunk {i + 1} ---\n\n")
            f.write(chunk)

    return output_file


if __name__ == "__main__":
    output_file = chunk_file("parsed_output/parsed_content.md")
    print("Chunking completed.")
    print(f"Chunks saved in: {output_file}")