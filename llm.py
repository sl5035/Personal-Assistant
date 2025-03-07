from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from typing import List, Dict, Any
import config

def create_llm():
    """
    Create and return a LangChain LLM instance
    """
    llm = ChatOpenAI(
        openai_api_key=config.OPENAI_API_KEY,
        model=config.LLM_MODEL,
        temperature=0.1
    )
    return llm

def format_documents(docs: List[Dict[Any, Any]]) -> str:
    """
    Format retrieved documents into a single context string with citations
    """
    if not docs:
        return "No relevant information found."
    
    formatted_docs = []
    
    for i, doc in enumerate(docs):
        # Extract date from metadata if available
        date = doc.get('metadata', {}).get('date', 'Unknown date')
        source = doc.get('metadata', {}).get('source', f'Document {i+1}')
        
        # Add source reference and content
        formatted_docs.append(
            f"[Source {i+1}: {source} ({date})]\n{doc['content']}\n"
        )
    
    return "\n\n".join(formatted_docs)

def generate_response(query: str, docs: List[Dict[Any, Any]]) -> str:
    """
    Generate a response using LLM with retrieved documents as context
    
    Args:
        query: User question
        docs: Retrieved relevant documents
        
    Returns:
        LLM-generated response
    """
    # Create LLM instance
    llm = create_llm()
    
    # Format documents into context
    context = format_documents(docs)
    
    # Create prompt template
    prompt_template = ChatPromptTemplate.from_template(
        """You are a helpful AI assistant that answers questions based on my personal notes.
        
        Below is context information from my notes:
        
        {context}
        
        Based only on the above context, answer the following question:
        {query}
        
        If the context doesn't contain relevant information to answer the question, 
        say "I don't have enough information to answer that question."
        
        Include citations to the sources you used in your answer. Format the citations as [Source X]
        where X is the source number from the context.
        """
    )
    
    # Create the chain
    chain = prompt_template | llm | StrOutputParser()
    
    # Run the chain
    response = chain.invoke({
        "context": context,
        "query": query
    })
    
    return response