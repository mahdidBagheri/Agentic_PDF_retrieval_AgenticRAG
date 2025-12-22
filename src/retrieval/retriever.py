from src.embeddings.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.reranker import rerank


class Retriever:
    def __init__(self, index_path: str, embed_dim: int = 384):
        self.embedder = Embedder()
        self.store = VectorStore(dim=embed_dim, index_path=index_path)
        self.store.load()

    def retrieve(self, query: str, top_k: int = 5):
        query_embedding = self.embedder.embed_texts([query])[0]

        # 1️⃣ High-recall semantic search
        initial_results = self.store.search(query_embedding, top_k=15)

        # 2️⃣ Precision-focused reranking
        reranked_results = rerank(query, initial_results)

        return reranked_results[:top_k]