from langgraph.graph import StateGraph, END

from src.graph.state import GraphState
from src.graph.router_node import router_node
from src.graph.retrieval_node import retrieval_node
from src.graph.direct_answer_node import direct_answer_node
from src.graph.rag_answer_node import rag_answer_node

graph = StateGraph(GraphState)

graph.add_node("router", router_node)
graph.add_node("retrieve", retrieval_node)
graph.add_node("direct_answer", direct_answer_node)
graph.add_node("rag_answer", rag_answer_node)

graph.set_entry_point("router")

graph.add_conditional_edges(
    "router",
    lambda state: state["route"],
    {
        "RAG": "retrieve",
        "DIRECT": "direct_answer",
    },
)

graph.add_edge("retrieve", "rag_answer")
graph.add_edge("direct_answer", END)
graph.add_edge("rag_answer", END)

app = graph.compile()