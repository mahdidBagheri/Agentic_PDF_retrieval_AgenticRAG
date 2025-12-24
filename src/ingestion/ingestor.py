import os
import shutil
from typing import List
from src.config import GEMINI_API_KEY
import PyPDF2
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from google import genai

# Define where the vector store lives
# ✅ Use absolute path relative to execution to avoid "file not found" errors
VECTORSTORE_PATH = os.path.join(os.getcwd(), "vectorstore", "faiss")

# --- 1. THE BRIDGE: Custom Wrapper ---
# This converts your manual "client.models.embed_content"
# into the format FAISS.load_local requires.
class GeminiEmbeddingWrapper(Embeddings):
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Used when adding documents
        result = self.client.models.embed_content(
            model="gemini-embedding-001",
            contents=texts
        )
        return [e.values for e in result.embeddings]

    def embed_query(self, text: str) -> List[float]:
        # Used when searching later
        result = self.client.models.embed_content(
            model="gemini-embedding-001",
            contents=text
        )
        return result.embeddings[0].values

# --- 2. THE LOGIC ---

def ingest_pdf(file_path: str):
    print(f"📄 Starting ingestion for: {file_path}")

    text = extract_text_from_pdf(file_path)
    if not text:
        raise ValueError("No text could be extracted.")

    chunks = chunk_text(text, chunk_size=1000, overlap=100)
    print(f"🧩 Split into {len(chunks)} chunks.")

    documents = [
        Document(
            page_content=chunk,
            metadata={"source": os.path.basename(file_path)}
        )
        for chunk in chunks
    ]

    update_vectorstore(documents)
    print("✅ Ingestion complete.")


def update_vectorstore(documents):
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set in config.py")

    # 1. Instantiate our Custom Embedding Wrapper
    embedding_tool = GeminiEmbeddingWrapper(api_key=GEMINI_API_KEY)

    # 2. THE FIX: Define the index name as "metadata"
    # We will force both save and load to use this non-default name
    # to resolve the library version conflict.
    INDEX_NAME = "metadata"

    # 3. Check for the specific index file using the new name
    index_file = os.path.join(VECTORSTORE_PATH, f"{INDEX_NAME}.faiss")

    if os.path.exists(index_file):
        print(f"🔄 Loading existing vector store from {VECTORSTORE_PATH} (index: '{INDEX_NAME}')...")
        try:
            vectorstore = FAISS.load_local(
                VECTORSTORE_PATH,
                embedding_tool,
                index_name=INDEX_NAME,  # ✅ Tell it to load metadata.pkl/faiss
                allow_dangerous_deserialization=True
            )
            print("✅ Store loaded. Adding new documents...")
            vectorstore.add_documents(documents)
        except Exception as e:
            print(f"⚠️ Error loading index '{INDEX_NAME}' ({e}). Recreating...")
            vectorstore = FAISS.from_documents(documents, embedding_tool)
    else:
        print(f"🆕 No index found. Creating new vector store with index name '{INDEX_NAME}'...")
        vectorstore = FAISS.from_documents(documents, embedding_tool)

    # 4. Save the vector store using the new name
    print(f"💾 Saving vector store to {VECTORSTORE_PATH} (index: '{INDEX_NAME}')...")
    vectorstore.save_local(
        VECTORSTORE_PATH,
        index_name=INDEX_NAME  # ✅ Force it to create metadata.pkl/faiss
    )
    print("✅ Save complete.")

# --- Helpers ---
def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                t = page.extract_text()
                if t: text += t + "\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks