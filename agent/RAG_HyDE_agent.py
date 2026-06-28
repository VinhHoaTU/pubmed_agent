import os
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv(override=True)


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
MODEL = "gpt-4.1-nano"

vectorstore = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
    collection_name="mes_articles_chunks",
)

base_retriever = vectorstore.as_retriever(
    search_type="similarity",   # ou "mmr", "similarity_score_threshold"
    search_kwargs={"k": 3},     # nombre de chunks à retourner
)

llm = ChatOpenAI(temperature= 0 ,model = MODEL)
question = "cancer treatment using immunotherapy"
answer_hypothesis = llm.invoke(f"Create an answer to the following question {question} that will be used to search for the real document for similarity search. the answer need to be succint, do not ask for more information and do not give the unrealted information")

# --- 4. Utilisation ---
results = base_retriever.invoke(f"{answer_hypothesis}")

for doc in results:
    print(doc.metadata, "->", doc.page_content[:150], "...")