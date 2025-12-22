from src.retrieval.retriever import Retriever


INDEX_PATH = "vectorstore/faiss"


def retrieval_node(state):
    retriever = Retriever(INDEX_PATH)
    contexts = retriever.retrieve(state["query"], top_k=5)

    state["contexts"] = contexts
    return state