from src.llm.gemini_client import GeminiClient


ROUTER_PROMPT = """
You are a classifier that decides whether a question
requires consulting internal documents.

Rules:
- Respond ONLY with one word
- Valid responses: RAG, DIRECT

Use RAG if the question is about company policies,
internal documents, or proprietary information.

Use DIRECT if the question is general knowledge,
math, greetings, or opinion.

Question:
{query}
"""


def router_node(state):
    llm = GeminiClient()
    prompt = ROUTER_PROMPT.format(query=state["query"])

    decision = llm.generate(prompt).strip().upper()

    if decision not in {"RAG", "DIRECT"}:
        decision = "RAG"  # safe fallback

    state["route"] = decision
    return state