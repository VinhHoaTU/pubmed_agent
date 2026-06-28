import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

MODEL = "gpt-4.1-nano"
RERANKER_TOP_K = 3

load_dotenv(override=True)


# --- 1. Définir les embeddings et GPT model (même modèle que celui utilisé pour l'ingestion) ---
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(temperature=0, model_name=MODEL)

# --- 2. Se connecter à la collection existante (déjà remplie dans Qdrant Cloud) ---
vectorstore = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
    collection_name="mes_articles_chunks",
)

# --- 3. Transformer le vectorstore en retriever ---
# ── Step 1 : base retriever
base_retriever = vectorstore.as_retriever(
    search_type="similarity",   # ou "mmr", "similarity_score_threshold"
    search_kwargs={"k": 20},     # nombre de chunks à retourner
)

# ── Step 2 : MultiQueryRetriever 
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
)

# ── Step 3 : Cross-encoder reranker 
reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-large")
compressor = CrossEncoderReranker(model=reranker_model, top_n=RERANKER_TOP_K)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=multi_query_retriever,
)

# --- 4. Utilisation ---
results = compression_retriever.invoke("cancer treatment using immunotherapy")

for doc in results:
    print(doc.metadata, "->", doc.page_content[:150], "...")
