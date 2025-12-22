from graph.graph import app


def main():
    while True:
        query = input("\n❓ Ask a question (or 'exit'): ")
        if query.lower() == "exit":
            break

        result = app.invoke({"query": query})
        print("\n🤖 Answer:\n")
        print(result["answer"])


if __name__ == "__main__":
    main()