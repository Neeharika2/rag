import glob
import os
import sys

from config import UPLOADS_DIR
from ingestion.pipeline import run_pipeline


def ingest_all(directory: str = UPLOADS_DIR) -> None:
    pdf_files = sorted(glob.glob(os.path.join(directory, "*.pdf")))

    if not pdf_files:
        print(f"No PDF files found in {directory}")
        return

    print(f"Found {len(pdf_files)} PDF(s) in {directory}")
    total_chunks = 0

    for i, pdf_path in enumerate(pdf_files, 1):
        filename = os.path.basename(pdf_path)
        print(f"\n[{i}/{len(pdf_files)}] Processing: {filename}")

        try:
            result = run_pipeline(pdf_path)
            total_chunks += result.get("embedded_count", 0)
            print(
                f"  Done. Parsed -> {result.get('parsed_file', 'N/A')}, "
                f"Embedded {result.get('embedded_count', 0)} chunks"
            )
        except Exception as e:
            print(f"  ERROR processing {filename}: {e}")
            continue

    print(f"\nIngestion complete. Total chunks embedded: {total_chunks}")


if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else UPLOADS_DIR
    ingest_all(target_dir)