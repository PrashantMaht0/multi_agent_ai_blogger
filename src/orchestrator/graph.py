"""
src/orchestrator/graph.py
LangGraph Setup & Routing Logic.
"""

import os
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool  # <-- Add this import

from src.state import AgentState
from src.agents.editor import editor_node
from src.agents.researcher import researcher_node
from src.agents.validator import validator_node
from src.agents.writer import writer_node
from src.agents.publisher import publisher_node

# (Keep your existing validation_router and editor_router functions here)
def validation_router(state: AgentState):
    """Routes to Writer if sources are valid, otherwise loops back to Researcher."""
    if state.get("validation_status") == "VALIDATED":
        return "writer"
        
    # Circuit Breaker: Stop infinite loops if Ollama is down or search is broken
    if state.get("revision_count", 0) >= 3:
        print("⚠️ Validation circuit breaker tripped. Bypassing to Writer.")
        return "writer"
        
    print("⚠️ Research validation failed. Re-routing to Researcher...")
    return "researcher"

def editor_router(state: AgentState):
    if state.get("revision_count", 0) >= 3 or state.get("last_evaluation") == "PASS":
        return "publisher"
    return "writer"


# Ensure the database pool is created once globally so it stays alive for Gradio
db_uri = os.getenv("POSTGRES_DB_URL")
if db_uri:
    # Disable autocommit handling which can conflict with LangGraph's saver
    connection_pool = ConnectionPool(conninfo=db_uri, max_size=20, kwargs={"autocommit": True})
else:
    connection_pool = None

def build_graph():
    workflow = StateGraph(AgentState)
    
    # Register all nodes
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("validator", validator_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("editor", editor_node)
    workflow.add_node("publisher", publisher_node)
    
    # Execution Flow
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "validator")
    
    workflow.add_conditional_edges(
        "validator",
        validation_router,
        {"writer": "writer", "researcher": "researcher"}
    )
    
    workflow.add_edge("writer", "editor")
    
    workflow.add_conditional_edges(
        "editor",
        editor_router,
        {"writer": "writer", "publisher": "publisher"}
    )
    
    workflow.add_edge("publisher", END)
    
    # Database Integration
    if connection_pool:
        # Instantiate the checkpointer using the active pool
        checkpointer = PostgresSaver(connection_pool)
        
        # Setup creates the required tables (checkpoints, checkpoint_blobs) if they don't exist
        checkpointer.setup()
        
        return workflow.compile(
            checkpointer=checkpointer,
            interrupt_before=["publisher"]
        )
    
    print("⚠️ No POSTGRES_DB_URL found. Compiling without persistent memory.")
    return workflow.compile(interrupt_before=["publisher"])