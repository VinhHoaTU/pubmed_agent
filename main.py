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


# # async 
# def retrival():
#     graph = build_graph()
#     initial_input = {"question": "What is the impact of nutrition on school children?"}
#     config = {"configurable": {"thread_id": "session_1"}}
    
#     # Now 'await' is inside an 'async def', so it is allowed!
#         # await / ainvoke
#     result = graph.invoke(initial_input, config=config)
#     # print(result["retrival"])



# def main():
#     retrival()
#     print(State["retrival"]) 


# if __name__ == "__main__":
#     # asyncio.run(main())
#     main()
# # initial_input = {"question": "What is the impact of nutrition on school children?"}
# # # 1. Create a config with a unique thread_id
# # config = {"configurable": {"thread_id": "session_1"}}

# # # 2. Pass the config into the invoke method
# # response = await graph.ainvoke(initial_input, config=config)
# # print(response)


from dotenv import load_dotenv
from graph import build_graph

load_dotenv()


def run(query: str, user_email: str):
    graph = build_graph()

    initial_state = {
        "query": query,
        "user_email": user_email,
        "retrieved_docs": [],
        "rag_response": "",
        "feedback": None,
        "approved": False,
        "email_sent": False,
        "iteration": 0,
        "messages": [],
    }

    print(f"\n🔍 Requête : {query}")
    # print(f"📧 Destinataire : {user_email}\n")

    result = graph.invoke(initial_state)

    print(f"\n📄 Réponse finale :\n{result['rag_response']}")
    # print(f"✅ Approuvé après {result['iteration']} itération(s)")
    # print(f"📬 Mail envoyé : {result['email_sent']}")
    return result


if __name__ == "__main__":
    run(
        query="What is the impact of nutrition on school children",
        user_email="tuvinhhoa192@gmail.com",
    )