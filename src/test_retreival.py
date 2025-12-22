from retrieval.retriever import Retriever


def main():
    retriever = Retriever("vectorstore/faiss")

    query = "What is the remote work policy?"
    results = retriever.retrieve(query, top_k=5)

    print(f"\n🔎 Query: {query}\n")

    for i, r in enumerate(results, start=1):
        print(f"\nResult #{i}")
        print(f"Semantic score: {r['score']:.4f}")
        print(f"Keyword score:  {r['keyword_score']:.4f}")
        print(f"Rerank score:   {r['rerank_score']:.4f}")
        print(r["text"][:300])


if __name__ == "__main__":
    main()