from src.llm.gemini_client import GeminiClient
from src.llm.prompts import build_rag_prompt
from src.llm.answer_guard import confident_enough


def rag_answer_node(state):
    contexts = state.get("contexts", [])

    if not confident_enough(contexts):
        state["answer"] = (
            "I’m not confident enough to answer based on the documents."
        )
        return state

    llm = GeminiClient()
    prompt = build_rag_prompt(state["query"], contexts)
    state["answer"] = llm.generate(prompt)
    return state