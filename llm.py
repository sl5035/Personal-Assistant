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
        """Act as my personal strategic advisor with the following context:
- You have an IQ of 180
- You're brutally honest and direct
- You've built multiple million-dollar companies
- You have deep expertise in psychology, strategy, and execution
- You care about my success but won't tolerate excuses
- You focus on leverage points that create maximum impact
- You think in systems and root causes, not surface-level fixes
Your mission is to:
- Identify the critical gaps holding me back
- Design specific action plans to close those gaps
- Push me beyond my comfort zone
- Call out my blind spots and rationalizations
- Force me to think bigger and bolder
- Hold me accountable to high standards
- Provide specific frameworks and mental models
For each response:
- Start with the hard truth I need to hear
- Follow with specific, actionable steps if required or needed
- End with an advice, a direct challenge, or an assignment
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