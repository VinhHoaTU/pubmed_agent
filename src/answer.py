import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv(override=True)
client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])
print(client.get_collections())