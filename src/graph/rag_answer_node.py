from src.llm.gemini_client import GeminiClient
from src.llm.prompts import build_rag_prompt
from src.llm.answer_guard import confident_enough


def rag_answer_node(state):
    """
    Generates an answer using the LLM based on the retrieved contexts.
    """
    contexts = state.get("contexts", [])

    # Gracefully handle cases where no documents were retrieved
    if not contexts:
        state["answer"] = "I could not find any relevant documents to answer your question."
        return state

    # --- THE FIX ---
    # Extract the raw text from each Document object into a list of strings
    context_texts = [doc.page_content for doc in contexts]
    # --- END FIX ---

    # TODO: Add reranker here if needed.

    # 3. Pass the clean list of context strings to the prompt builder
    llm = GeminiClient()
    prompt = build_rag_prompt(state["query"], context_texts)  # Use the new list of strings

    print("--- GENERATING FINAL ANSWER ---")
    state["answer"] = llm.generate(prompt)
    return state