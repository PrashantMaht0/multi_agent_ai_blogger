"""Graph wiring and routing logic."""

import os
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool  

from src.state import AgentState
from src.agents.editor import editor_node
from src.agents.sanitize import sanitize_html
from src.agents.researcher import researcher_node
from src.agents.validator import validator_node
from src.agents.writer import writer_node
from src.agents.publisher import publisher_node


# Two passes: a third search rarely finds what the first two missed.
MAX_RESEARCH_ATTEMPTS = 2


def validation_router(state: AgentState):
    """Routes to Writer if sources are valid, otherwise loops back to Researcher."""
    if state.get("validation_status") == "VALIDATED":
        return "writer"

    # Give up rather than hand unusable research to the writer.
    if state.get("research_attempts", 0) >= MAX_RESEARCH_ATTEMPTS:
        print("⚠️ Validation circuit breaker tripped. Aborting run.")
        return "abort"

    print("⚠️ Research validation failed. Re-routing to Researcher...")
    return "researcher"


def editor_router(state: AgentState):
    if state.get("revision_count", 0) >= 3 or state.get("last_evaluation") == "PASS":
        return "publisher"
    return "writer"


def sanitizer_node(state: AgentState) -> dict:
    """Strips unsafe markup from the draft before it reaches the publisher."""
    cleaned, removed = sanitize_html(state.get("draft", ""))
    if removed:
        print(f"Sanitizer removed: {', '.join(removed)}")
    return {"draft": cleaned, "sanitizer_removed": removed, "sender": "sanitizer"}


def abort_node(state: AgentState) -> dict:
    """Terminal node for runs whose research could never be validated."""
    reason = state.get("research_error") or state.get("validation_feedback") or "Unknown research failure."
    print(f"🛑 Run aborted: {reason}")
    return {"run_status": "FAILED", "sender": "abort"}


_connection_pool = None


def _get_pool():
    """Builds the Postgres pool on first use, so importing this module opens nothing."""
    global _connection_pool
    if _connection_pool is None:
        db_uri = os.getenv("POSTGRES_DB_URL")
        if db_uri:
            _connection_pool = ConnectionPool(conninfo=db_uri, max_size=20, kwargs={"autocommit": True})
    return _connection_pool


def build_graph(enable_hitl: bool = True, include_publisher: bool = True, use_checkpointer: bool = True):
    """Builds the graph: HITL pauses before publishing, and evaluations drop the publisher."""
    workflow = StateGraph(AgentState)

    workflow.add_node("researcher", researcher_node)
    workflow.add_node("validator", validator_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("editor", editor_node)
    workflow.add_node("sanitizer", sanitizer_node)
    workflow.add_node("abort", abort_node)
    if include_publisher:
        workflow.add_node("publisher", publisher_node)

    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "validator")

    workflow.add_conditional_edges(
        "validator",
        validation_router,
        {"writer": "writer", "researcher": "researcher", "abort": "abort"}
    )

    workflow.add_edge("abort", END)
    workflow.add_edge("writer", "editor")

    # Sanitise on the way out of the editor, so evaluations see the published draft.
    workflow.add_conditional_edges(
        "editor",
        editor_router,
        {"writer": "writer", "publisher": "sanitizer"}
    )

    if include_publisher:
        workflow.add_edge("sanitizer", "publisher")
        workflow.add_edge("publisher", END)
    else:
        workflow.add_edge("sanitizer", END)

    interrupts = ["publisher"] if (enable_hitl and include_publisher) else []

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