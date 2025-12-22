import os
import json

from ingestion.pdf_loader import load_pdfs
from ingestion.chunker import chunk_documents


PDF_DIR = "../data/raw_pdfs"
CHUNKS_DIR = "../data/chunks"
CHUNKS_PATH = os.path.join(CHUNKS_DIR, "all_chunks.json")


def main():
    os.makedirs(CHUNKS_DIR, exist_ok=True)

    print("📄 Loading PDFs...")
    documents = load_pdfs(PDF_DIR)
    print(f"✅ Loaded {len(documents)} pages")

    print("✂️ Chunking documents...")
    chunks = chunk_documents(
        documents,
        chunk_size=500,
        overlap=50,
    )
    print(f"✅ Created {len(chunks)} chunks")

    print(f"💾 Saving chunks to {CHUNKS_PATH}")
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print("🎉 Ingestion complete")


if __name__ == "__main__":
    main()