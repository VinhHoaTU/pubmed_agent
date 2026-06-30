from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from state import State
import json

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)



def generator_agent(state: State) -> dict:
    context = "\n".join(state["retrieved_docs"])
    feedback = state.get("feedback")

    prompt = f"""Réponds à la question en te basant uniquement sur ce contexte.

Contexte :
{context}

Question : {state['query']}
"""
    if feedback:
        prompt += f"\nFeedback à intégrer dans cette nouvelle tentative :\n{feedback}"

    response = llm.invoke([HumanMessage(content=prompt)])


    return {
        "rag_response": response.content,
        "messages": [response],
    }