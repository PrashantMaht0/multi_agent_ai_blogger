"""
src/orchestrator/graph.py
LangGraph Setup & Routing Logic.
"""

import os
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool  

from src.state import AgentState
from src.agents.editor import editor_node
from src.agents.researcher import researcher_node
from src.agents.validator import validator_node
from src.agents.writer import writer_node
from src.agents.publisher import publisher_node


MAX_RESEARCH_ATTEMPTS = 3


def validation_router(state: AgentState):
    """Routes to Writer if sources are valid, otherwise loops back to Researcher."""
    if state.get("validation_status") == "VALIDATED":
        return "writer"

    # Circuit Breaker: Stop infinite loops if Ollama is down or search is broken.
    # Aborts instead of bypassing to Writer, so unusable research never becomes a draft.
    if state.get("research_attempts", 0) >= MAX_RESEARCH_ATTEMPTS:
        print("⚠️ Validation circuit breaker tripped. Aborting run.")
        return "abort"

    print("⚠️ Research validation failed. Re-routing to Researcher...")
    return "researcher"


def editor_router(state: AgentState):
    if state.get("revision_count", 0) >= 3 or state.get("last_evaluation") == "PASS":
        return "publisher"
    return "writer"


def abort_node(state: AgentState) -> dict:
    """Terminal node for runs whose research could never be validated."""
    reason = state.get("research_error") or state.get("validation_feedback") or "Unknown research failure."
    print(f"🛑 Run aborted: {reason}")
    return {"run_status": "FAILED", "sender": "abort"}


_connection_pool = None


def _get_pool():
    """Lazily builds the Postgres pool so importing this module opens no connections."""
    global _connection_pool
    if _connection_pool is None:
        db_uri = os.getenv("POSTGRES_DB_URL")
        if db_uri:
            _connection_pool = ConnectionPool(conninfo=db_uri, max_size=20, kwargs={"autocommit": True})
    return _connection_pool


def build_graph(enable_hitl: bool = True, include_publisher: bool = True, use_checkpointer: bool = True):
    """
    Constructs and compiles the multi-agent graph.

    :param enable_hitl: If True, pauses execution before the 'publisher' node
                        to allow human review in Gradio.
    :param include_publisher: If False, the editor's PASS branch ends the run instead of
                              publishing. Used by evaluations so a full pipeline runs in a
                              single invocation without posting live.
    :param use_checkpointer: If False, compiles without persistence (no resumable threads).
    """
    workflow = StateGraph(AgentState)
    
    # Register all nodes
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("validator", validator_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("editor", editor_node)
    workflow.add_node("abort", abort_node)
    if include_publisher:
        workflow.add_node("publisher", publisher_node)

    # Execution Flow
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "validator")

    workflow.add_conditional_edges(
        "validator",
        validation_router,
        {"writer": "writer", "researcher": "researcher", "abort": "abort"}
    )

    workflow.add_edge("abort", END)
    workflow.add_edge("writer", "editor")

    workflow.add_conditional_edges(
        "editor",
        editor_router,
        {"writer": "writer", "publisher": "publisher" if include_publisher else END}
    )

    if include_publisher:
        workflow.add_edge("publisher", END)

    # Define interrupts based on HITL setting
    interrupts = ["publisher"] if (enable_hitl and include_publisher) else []

    # Database Integration
    connection_pool = _get_pool() if use_checkpointer else None
    if connection_pool:
        checkpointer = PostgresSaver(connection_pool)
        checkpointer.setup()

        return workflow.compile(
            checkpointer=checkpointer,
            interrupt_before=interrupts
        )

    if use_checkpointer:
        print("⚠️ No POSTGRES_DB_URL found. Compiling without persistent memory.")
    return workflow.compile(interrupt_before=interrupts)