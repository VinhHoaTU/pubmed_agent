import os
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# --- 1. Définir les embeddings (même modèle que celui utilisé pour l'ingestion) ---
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# --- 2. Se connecter à la collection existante (déjà remplie dans Qdrant Cloud) ---
vectorstore = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
    collection_name="mes_articles_chunks",
)

# --- 3. Transformer le vectorstore en retriever ---
retriever = vectorstore.as_retriever(
    search_type="similarity",   # ou "mmr", "similarity_score_threshold"
    search_kwargs={"k": 3},     # nombre de chunks à retourner
)

# --- 4. Utilisation ---
results = retriever.invoke("cancer treatment using immunotherapy")

for doc in results:
    print(doc.metadata, "->", doc.page_content[:150], "...")