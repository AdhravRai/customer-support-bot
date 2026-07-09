"""
LangGraph workflow for the GigaCorp Customer Support Assistant.
Defines the conversational state, the node operations, and compiles the execution graph.
"""

import sys
from pathlib import Path
from typing import Annotated, Sequence, TypedDict

# Add project root to sys.path to allow absolute imports
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from src.config.settings import LLM_MODEL_NAME, GEMINI_API_KEY
from src.retriever.search import retrieve_documents

# Define the State of the LangGraph agent
class AgentState(TypedDict):
    # Conversations are tracked by appending messages using add_messages annotation
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # Retained document context for citations and UI visibility
    context: list[dict]

def retrieve_node(state: AgentState):
    """Retrieves relevant customer support docs based on the user's latest query."""
    messages = state.get("messages", [])
    
    # Locate the last human query in the conversation
    last_human_message = None
    for msg in reversed(messages):
        if msg.type == "human":
            last_human_message = msg
            break
            
    if not last_human_message:
        # Fallback if no human message is found
        return {"context": []}
        
    query_text = last_human_message.content
    print(f"[Workflow] Retrieving docs for query: '{query_text}'")
    
    try:
        docs = retrieve_documents(query_text)
    except FileNotFoundError as e:
        print(f"[Workflow] Search failed: {e}")
        # Return empty context if vector store isn't compiled yet
        return {"context": []}
        
    # Serialize retrieved documents to pass in state
    serialized_docs = [
        {
            "content": doc.page_content,
            "section": doc.metadata.get("section", "General FAQ"),
            "source": doc.metadata.get("source_name", "gigacorp_faq.txt"),
            "similarity_score": doc.metadata.get("similarity_score", 0.0)
        }
        for doc in docs
    ]
    
    return {"context": serialized_docs}

def generate_node(state: AgentState):
    """Generates an answer citing retrieved source documents using the Gemini LLM."""
    messages = state.get("messages", [])
    context = state.get("context", [])
    
    if not context:
        # If no context was retrieved (e.g. vector store not built yet)
        fallback_msg = (
            "I'm sorry, I cannot access the GigaCorp knowledge base at the moment. "
            "Please make sure the vector database has been initialized or contact our support team at support@gigacorp.com."
        )
        from langchain_core.messages import AIMessage
        return {"messages": [AIMessage(content=fallback_msg)]}
        
    # Formulate context text block with indexes for citation matching
    context_str = ""
    for idx, doc in enumerate(context):
        context_str += f"\n--- DOCUMENT SOURCE [{idx + 1}] ---\n"
        context_str += f"Source: {doc['source']}\n"
        context_str += f"Section: {doc['section']}\n"
        context_str += f"Content:\n{doc['content']}\n"
        
    # System instructions setting character role, RAG constraints, and citation formatting
    system_prompt = (
        "You are GigaCorp's friendly, professional, and knowledgeable AI Customer Support Assistant.\n"
        "Your task is to answer the user's questions truthfully and accurately using ONLY the provided CONTEXT below.\n\n"
        f"CONTEXT INFORMATION:\n{context_str}\n"
        "RULES FOR GENERATION:\n"
        "1. Answer the query relying ONLY on the provided CONTEXT. If the answer cannot be found "
        "in the context, state that you do not have that information and suggest emailing support@gigacorp.com. "
        "Do NOT make up any details or extrapolate beyond what is written.\n"
        "2. Cite your sources in the text. Whenever you state a policy or fact from the context, append a bracketed "
        "citation pointing to the source document index (e.g., [1] or [2]) and mention the specific section name. "
        "For example: 'As per SECTION 1: Shipping Policies [1], domestic express shipping costs $14.99.'\n"
        "3. Maintain a helpful and polite conversational tone.\n"
        "4. Address the user directly based on the conversational history."
    )
    
    # Initialize Gemini model with zero temperature for deterministic, factual outputs
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME,
        google_api_key=GEMINI_API_KEY,
        temperature=0.0
    )
    
    # Construct complete input message sequence (system instructions + conversational history)
    input_messages = [SystemMessage(content=system_prompt)] + list(messages)
    
    response = llm.invoke(input_messages)
    return {"messages": [response]}

# ---------------------------------------------------------
# Graph Compilation
# ---------------------------------------------------------
# Instantiate the graph with our state definition
builder = StateGraph(AgentState)

# Add processing nodes
builder.add_node("retrieve", retrieve_node)
builder.add_node("generate", generate_node)

# Set up flow connections
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

# Compile into a runnable LangGraph application
graph_app = builder.compile()
