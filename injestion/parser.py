import os
from dotenv import load_dotenv
from llama_parse import LlamaParse

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("LLAMA_PARSER_KEY")

if not api_key:
    raise ValueError("LLAMA_PARSER_KEY not found in .env file")

# Initialize parser
parser = LlamaParse(
    api_key=api_key,
    result_type="markdown",   # output format
    premium_mode=True         # better parsing for tables/images
)

# PDF file path
pdf_path = "sample.pdf"

# Parse PDF
documents = parser.load_data(pdf_path)

# Create output folder
os.makedirs("parsed_output", exist_ok=True)

# Save parsed content
output_file = "parsed_output/parsed_content.md"

with open(output_file, "w", encoding="utf-8") as f:
    for i, doc in enumerate(documents):
        f.write(f"\n\n--- Page {i+1} ---\n\n")
        f.write(doc.text)

print(f"Parsing completed.")
print(f"Parsed file saved at: {output_file}")