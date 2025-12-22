from src.llm.gemini_client import GeminiClient


def direct_answer_node(state):
    llm = GeminiClient()
    state["answer"] = llm.generate(state["query"])
    return state