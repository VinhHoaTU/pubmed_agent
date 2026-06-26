import os
import glob
from datasets import load_dataset
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from dotenv import load_dotenv
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore

load_dotenv(override=True)

# ds is a list of dictionaries train, validation, test
# DatasetDict({
#     train: [{"article": content, "abstract": summary}])
#     validation: [{"article": content, "abstract": summary}]
#     test: [{"article": content, "abstract": summary}]
# })
# print(ds["train"][0]["abstract"])

# #  FULL dataset pour la production
# # Concatène les "article" des 3 splits en une seule liste
# # Convertir chaque Column en list avant de concaténer



# --- 1. Charger les données ---
ds = load_dataset("ccdv/pubmed-summarization", "section")

articles = list(ds["train"]["article"][:100])
# articles = list(ds["train"]["article"]) + list(ds["validation"]["article"]) + list(ds["test"]["article"])

# Filtre : élimine les articles vides ou ne contenant que des espaces
articles = [a for a in articles if a and a.strip()]

mes_metadonnees = [{"source": "pubmed", "index": i} for i in range(len(articles))]

# --- 2. Chunking ---
splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=256)
documents_morceles = splitter.create_documents(
    texts=articles,
    metadatas=mes_metadonnees,
)


# --- 3. Embedding + insertion Qdrant ---
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


qdrant = QdrantVectorStore.from_documents(
    documents=documents_morceles,
    embedding=embeddings,
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    collection_name="mes_articles_chunks",
    prefer_grpc=True,  # optionnel : plus rapide que HTTP, mais nécessite que le cluster le supporte
)
print(f"{len(documents_morceles)} chunks insérés avec succès dans Qdrant !")

# --- 4. Test de recherche ---
results = qdrant.similarity_search("cancer treatment using immunotherapy", k=3)
for r in results:
    print(r.metadata, "->", r.page_content[:150], "...")
