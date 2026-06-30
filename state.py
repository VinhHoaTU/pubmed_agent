from typing import Annotated, List, Optional, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class State(TypedDict):
    query: str                        # Question de l'utilisateur
    # user_email: str                   # Destinataire du mail
    retrieved_docs: List[str]         # Documents récupérés par le RAG
    rag_response: str                 # Réponse générée par le RAG
    feedback: Optional[str]           # Feedback de l'évaluateur
    approved: bool                    # Réponse approuvée ou non
    # email_sent: bool                  # Mail envoyé ou non
    iteration: int                    # Nombre de tentatives (anti-boucle infinie)
    messages: Annotated[List[Any], add_messages]


class EvaluatorOutput(BaseModel):
    feedback: str = Field(description="Feedback détaillé sur la réponse RAG")
    approved: bool = Field(description="True si la réponse est satisfaisante")