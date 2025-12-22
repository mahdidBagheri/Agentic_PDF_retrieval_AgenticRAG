from retrieval.retriever import Retriever
from llm.gemini_client import GeminiClient
from llm.prompts import build_rag_prompt
from llm.compressor import ContextCompressor


INDEX_PATH = "vectorstore/faiss"


def main():
    retriever = Retriever(INDEX_PATH)
    llm = GeminiClient()

    while True:
        query = input("\n❓ Ask a question (or 'exit'): ")
        if query.lower() == "exit":
            break

        contexts = retriever.retrieve(query, top_k=5)

        if not contexts:
            print("No relevant context found.")
            continue

        prompt = build_rag_prompt(query, contexts)
        answer = llm.generate(prompt)

        print("\n🤖 Answer:\n")
        print(answer)

        print("\n📚 Sources:")
        for c in contexts:
            print(f"- {c['source']} (page {c['page']})")


if __name__ == "__main__":
    main()