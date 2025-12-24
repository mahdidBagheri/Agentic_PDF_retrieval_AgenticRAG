import os
import pickle
from typing import List, Dict

import faiss
import numpy as np


class VectorStore:
    def __init__(self, dim: int, index_path: str):
        self.dim = dim
        self.index_path = index_path
        self.index_file = os.path.join(index_path, "metadata.faiss")
        self.meta_file = os.path.join(index_path, "metadata.pkl")

        self.index = faiss.IndexFlatIP(dim)
        self.metadata: List[Dict] = []

        os.makedirs(index_path, exist_ok=True)

    def add(self, embeddings: List[List[float]], metadata: List[Dict]):
        vectors = np.array(embeddings).astype("float32")
        self.index.add(vectors)
        self.metadata.extend(metadata)

    def save(self):
        faiss.write_index(self.index, self.index_file)
        with open(self.meta_file, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self):
        self.index = faiss.read_index(self.index_file)
        with open(self.meta_file, "rb") as f:
            self.metadata = pickle.load(f)

    def search(self, query_embedding: List[float], top_k: int = 5):
        query_vec = np.array([query_embedding]).astype("float32")
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            item = self.metadata[idx].copy()
            item["score"] = float(score)
            results.append(item)

        return results