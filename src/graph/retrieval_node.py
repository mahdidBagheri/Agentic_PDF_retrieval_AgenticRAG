from src.retrieval.retriever import Retriever

import os
INDEX_PATH =  os.path.join(os.getcwd(), "vectorstore", "faiss")


def retrieval_node(state):
    retriever = Retriever()
    contexts = retriever.retrieve(state["query"], top_k=5)

    state["contexts"] = contexts
    return state