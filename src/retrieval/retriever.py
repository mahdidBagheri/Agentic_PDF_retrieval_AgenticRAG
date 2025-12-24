import os
from langchain_community.vectorstores import FAISS
from src.config import VECTORSTORE_PATH, INDEX_NAME, GEMINI_API_KEY
from src.ingestion.ingestor import GeminiEmbeddingWrapper  # We re-use the wrapper


class Retriever:
    def __init__(self):
        """
        Initializes the retriever by loading the FAISS vector store
        and the Gemini embedding model.
        """
        # 1. Check if the vector store actually exists
        index_file = os.path.join(VECTORSTORE_PATH, f"{INDEX_NAME}.faiss")
        if not os.path.exists(index_file):
            raise FileNotFoundError(
                f"Vector store not found at {VECTORSTORE_PATH}. "
                "Please run the ingestion process first by uploading a PDF."
            )

        # 2. Use the SAME embedding tool as the ingestor
        print("Retriever: Initializing Gemini Embedding model...")
        self.embedding_tool = GeminiEmbeddingWrapper(api_key=GEMINI_API_KEY)

        # 3. Load the vector store using LangChain, matching the ingestor's settings
        print(f"Retriever: Loading vector store from disk with index '{INDEX_NAME}'...")
        self.vectorstore = FAISS.load_local(
            folder_path=VECTORSTORE_PATH,
            embeddings=self.embedding_tool,
            index_name=INDEX_NAME,
            allow_dangerous_deserialization=True
        )
        print("✅ Retriever vector store loaded successfully.")

        # 4. Create the core retriever component from the loaded store
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={'k': 10}  # Ask for 10 initial results
        )

    def retrieve(self, query: str, top_k: int = 5):
        """
        Retrieves relevant documents from the vector store for a given query.
        """
        print(f"🔍 Retrieving documents for query: '{query}'")

        # ✅ THE FIX: Use .invoke() instead of the older .get_relevant_documents()
        documents = self.retriever.invoke(query)

        print(f"Retrieved {len(documents)} documents.")

        # NOTE: If you add a reranker later, you would process 'documents' here.

        return documents[:top_k]