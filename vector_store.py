from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from typing import List, Dict, Any
import config

def get_vector_store():
    """
    Connect to Qdrant vector database and return a Langchain vector store instance
    """
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        openai_api_key=config.OPENAI_API_KEY
    )
    
    # Create Langchain Qdrant vector store
    qdrant = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=config.QDRANT_COLLECTION_NAME,
        url=config.QDRANT_HOST,
        api_key=config.QDRANT_API_KEY,
    )
    
    return qdrant

def query_vector_store(query: str) -> List[Dict[Any, Any]]:
    """
    Query the vector store for relevant documents
    
    Args:
        query: User question or search query
        
    Returns:
        List of document dictionaries with content and metadata
    """
    vector_store = get_vector_store()
    
    # Search for similar documents
    docs = vector_store.similarity_search_with_score(
        query=query,
        k=config.MAX_RESULTS
    )
    
    # Format results
    results = []
    for doc, score in docs:
        results.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "similarity_score": score
        })
    
    return results