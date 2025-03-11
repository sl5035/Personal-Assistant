from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import List, Dict, Any
import config

# Global memory to persist between function calls
_memory = None

def create_llm():
    """
    Create and return a LangChain LLM instance.
    """
    llm = ChatOpenAI(
        openai_api_key=config.OPENAI_API_KEY,
        model=config.LLM_MODEL,
        temperature=0.1
    )
    return llm

def format_documents(docs: List[Dict[Any, Any]]) -> str:
    """
    Format retrieved documents into a single context string with citations.
    """
    if not docs:
        return "No relevant information found."
    
    formatted_docs = []
    for i, doc in enumerate(docs):
        # Extract date and source from metadata if available
        date = doc.get('metadata', {}).get('date', 'Unknown date')
        source = doc.get('metadata', {}).get('source', f'Document {i+1}')
        formatted_docs.append(
            f"[Source {i+1}: {source} ({date})]\n{doc['content']}\n"
        )
    
    return "\n\n".join(formatted_docs)

def get_memory():
    """
    Get or create the global conversation memory.
    """
    global _memory
    if _memory is None:
        llm = create_llm()
        _memory = ConversationSummaryBufferMemory(
            llm=llm,
            memory_key="chat_history",
            return_messages=True,
            max_token_limit=512
        )
    return _memory

def generate_response(query: str, docs: List[Dict[Any, Any]]) -> str:
    """
    Generate a response using LLM with retrieved documents as context,
    while retaining conversation history via a summary memory buffer.
    
    Args:
        query: User's question.
        docs: Retrieved relevant documents.
        
    Returns:
        LLM-generated response.
    """
    # Create LLM instance and format the provided documents as context
    llm = create_llm()
    context = format_documents(docs)
    
    # Get or create the conversation memory
    memory = get_memory()
    
    # Define a prompt template with context, user query, and chat history
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""Act as my personal strategic advisor with the following context:\n\n{context}\n\n
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
- Start with specific, actionable steps if required or needed
- Include a hard truth that the user is overlooking
- End with an advice, a direct challenge, or an assignment"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{query}")
    ])
    
    # Build the chain with LLM, prompt template, and conversation memory
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=False,
        memory=memory
    )
    
    # Run the chain with context and query
    response = chain.invoke(input={
        "query": query
    })
    
    return response["text"]

def generate_streaming_response(query: str, docs: List[Dict[Any, Any]]):
    """
    Generate a streaming response using LLM with retrieved documents as context,
    while retaining conversation history via a summary memory buffer.
    
    Args:
        query: User's question.
        docs: Retrieved relevant documents.
        
    Yields:
        Tokens of the LLM-generated response as they are produced.
    """
    # Create LLM instance with streaming enabled and format the provided documents as context
    llm = ChatOpenAI(
        openai_api_key=config.OPENAI_API_KEY,
        model=config.LLM_MODEL,
        temperature=0.1,
        streaming=True
    )
    context = format_documents(docs)
    
    # Get or create the conversation memory
    memory = get_memory()
    
    # Define a prompt template with context, user query, and chat history
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""Act as my personal strategic advisor with the following context:\n\n{context}\n\n
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
- Start with specific, actionable steps if required or needed
- Include a hard truth that the user is overlooking
- End with an advice, a direct challenge, or an assignment"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{query}")
    ])
    
    # Get chat history from memory
    chat_history = memory.chat_memory.messages
    
    # Create a streaming chain
    chain = prompt | llm
    
    # Run the chain with context and query
    for chunk in chain.stream({"query": query, "chat_history": chat_history}):
        yield chunk.content
    
    # Update memory with the conversation after streaming
    memory.chat_memory.add_user_message(query)
    # We need to run the non-streaming version to get the full response for memory
    full_response = generate_response(query, docs)
    memory.chat_memory.add_ai_message(full_response)