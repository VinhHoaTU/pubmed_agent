from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class State(TypedDict):
    """Shared state for the OSINT multi-agent system"""

    # # Conversation messages (for LLM reasoning)
    # messages: Annotated[list, add_messages]

    # # Retrival information
    question: str
    # retrival: list[list]
    # answer: str

    # # Workflow control
    # current_phase: str  # Current investigation phase
    # feedback: list[str]  # Phases we've finished
    # criteria_met: bool

    # # Final output
    # report: str  # Generated report
