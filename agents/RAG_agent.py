from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from langgraph.types import Overwrite
from state import State
from langchain_qdrant import QdrantVectorStore
import os
from dotenv import load_dotenv
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)

load_dotenv(override=True)


MODEL = "gpt-4.1-nano"
RERANKER_TOP_K = 3
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(temperature=0, model_name=MODEL)


# Se connecter à la collection existante (déjà remplie dans Qdrant Cloud) ---
vectorstore = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
    collection_name="mes_articles_chunks",
)


base_retriever = vectorstore.as_retriever(
    search_type="similarity",  # ou "mmr", "similarity_score_threshold"
    search_kwargs={"k": 20},  # nombre de chunks à retourner
)

# ── Step 2 : Cross-encoder reranker
reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
compressor = CrossEncoderReranker(model=reranker_model, top_n=RERANKER_TOP_K)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)


def rag_agent(state: State) -> dict:
    """
    Récupère les documents pertinents et génère une réponse.
    Tient compte du feedback de l'évaluateur si présent.
    """
    query = state["query"]
    feedback = state.get("feedback")

    # Recherche vectorielle
    docs = compression_retriever.invoke(f"{query}")

    retrieved_texts = [doc.page_content for doc in docs]

    return {"retrieved_docs": retrieved_texts}
