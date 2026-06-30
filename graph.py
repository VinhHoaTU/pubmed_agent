from langgraph.graph import StateGraph, START, END
from state import State
from agents.RAG_agent import rag_agent
from agents.evaluator_agent import evaluator_agent

# from agents.sender_agent import sender_agent


def route_after_evaluation(state: State) -> str:
    """Routage conditionnel selon la décision de l'évaluateur."""
    if state["approved"]:
        return "sender"
    return "rag"  # Retour au RAG pour amélioration


def build_graph():
    graph = StateGraph(State)

    # Ajout des nœuds
    graph.add_node("rag", rag_agent)
    graph.add_node("evaluator", evaluator_agent)
    # graph.add_node("sender", sender_agent)

    # Flux principal
    graph.add_edge(START, "rag")
    graph.add_edge("rag", "evaluator")

    # # Routage conditionnel depuis l'évaluateur
    # graph.add_conditional_edges(
    #     "evaluator",
    #     route_after_evaluation,
    #     {"sender": "sender", "rag": "rag"},
    # )

    graph.add_edge("evaluator", END)

    return graph.compile()
