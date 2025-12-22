import json
from embeddings.embedder import Embedder
from retrieval.vector_store import VectorStore


CHUNKS_PATH = "../data/chunks/all_chunks.json"
INDEX_PATH = "vectorstore/faiss"
EMBED_DIM = 384


def main():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [c["text"] for c in chunks]

    embedder = Embedder()
    embeddings = embedder.embed_texts(texts)

    store = VectorStore(dim=EMBED_DIM, index_path=INDEX_PATH)
    store.add(embeddings, chunks)
    store.save()

    print(f"✅ Indexed {len(chunks)} chunks")


if __name__ == "__main__":
    main()