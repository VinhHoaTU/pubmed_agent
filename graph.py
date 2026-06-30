# from typing import Annotated
# from typing_extensions import TypedDict
# from langgraph.graph import StateGraph, START, END
# from langgraph.graph.message import add_messages
# from dotenv import load_dotenv
# from langgraph.prebuilt import ToolNode
# from langchain_openai import ChatOpenAI
# from langgraph.checkpoint.memory import MemorySaver
# from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
# from typing import List, Any, Optional, Dict
# from pydantic import BaseModel, Field
# # from sidekick_tools import playwright_tools, other_tools
# import uuid
# import asyncio
# from datetime import datetime
# from agent.RAG_rerank_agent import rag
# from state import State

# def build_graph():

#     # Set up Graph Builder with State
#     graph_builder = StateGraph(State)
#     # Add nodes
#     graph_builder.add_node("rag_rerank", rag)
#     # Add edges
#     graph_builder.add_edge(START, "rag_rerank")
#     # Compile the graph
#     memory = MemorySaver()
#     graph = graph_builder.compile(checkpointer=memory)

#     return graph

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