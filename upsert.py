from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from typing import List, Dict, Any
import config

"""
Connect to Qdrant vector database and return a Langchain vector store instance
"""
# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(
    model=config.EMBEDDING_MODEL,
    openai_api_key=config.OPENAI_API_KEY
)

client = QdrantClient(url="https://6736db8a-9198-4d40-be95-ce5852b3bfee.us-west-2-0.aws.cloud.qdrant.io", port=6333, api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.GagkebJM7moWJZZRC8NmFEldzPm5aQ5I8thO9iaiA98")

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

vector_store = QdrantVectorStore(
    client=client,
    collection_name="notes",
    embedding=embeddings,
)

from langchain_core.documents import Document

text = """The Godfather centers around the Corleone family, an influential Italian-American mafia dynasty led by the respected yet ruthless patriarch, Vito Corleone. When Vito refuses to enter the lucrative but dangerous drug trade, rival gangs launch an attack, leaving him critically wounded. As violence escalates, his reluctant son Michael—initially distant from the family’s criminal empire—steps forward, ultimately becoming deeply entangled in the mob’s brutal politics.

As the family faces betrayals and assassinations, Michael transforms from an idealistic war hero into a cold, calculated successor determined to protect and expand his family’s influence. His ascension brings power and stability to the Corleones, but at great personal cost: he sacrifices morality, love, and family bonds, evolving into the very kind of ruthless mafia boss he once despised. The film ends with Michael firmly entrenched as the new Godfather, cementing his family’s legacy through a combination of violence, strategic alliances, and ruthless ambition."""

document_1 = Document(
    page_content=text,
    metadata={"source": "tweet"},
)

documents = [
    document_1
]

vector_store.add_documents(documents=documents, ids=[3])