"""Graph wiring and routing logic."""

import os
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.types import Send
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool  

from src.state import AgentState, SubQuestion, Timeframe
from src.agents.auditor import (MAX_AUDIT_LOOPS, auditor_node, cross_check_node, gap_planner_node,
                                new_gaps)
from src.agents.editor import MAX_REVISIONS, editor_node
from src.tools.sanitize import clean_title, sanitize_html, unsupported_figures
from src.agents.planner import planner_node
from src.agents.research_branch import research_branch
from src.agents.writer import writer_node
from src.tools.publish import content_hash, publish_node


class BranchInput(TypedDict):
    sub_question: SubQuestion
    timeframe: Timeframe | None


def research_branch_node(payload: BranchInput) -> dict:
    """One sub question; writes only the appending claims and branch_errors fields."""
    return research_branch(payload["sub_question"], payload["timeframe"])


def dispatch_research(state: AgentState):
    """One branch per sub question not yet researched, all in the same step."""
    researched = ({c["sub_question_id"] for c in state.get("claims", [])}
                  | {e["sub_question_id"] for e in state.get("branch_errors", [])})
    pending = [q for q in state.get("sub_questions", []) if q["id"] not in researched]
    if not pending:
        return "auditor"
    return [Send("research_branch", {"sub_question": q, "timeframe": state.get("timeframe")}) for q in pending]


def audit_router(state: AgentState):
    """One gap round when research is short or a gap matters; otherwise write or abort."""
    gaps = new_gaps(state)
    if (state.get("audit_loops", 0) < MAX_AUDIT_LOOPS and gaps
            and (state.get("audit_status") == "FAILED" or any(g["important"] for g in gaps))):
        return "gap_planner"
    if state.get("audit_status") == "PASSED":
        return "writer"
    print(f"🛑 Research audit failed: {state.get('audit_feedback')}")
    return "abort"


def editor_router(state: AgentState):
    if state.get("revision_count", 0) >= MAX_REVISIONS or state.get("last_evaluation") == "PASS":
        return "sanitizer"
    return "writer"


def sanitizer_node(state: AgentState) -> dict:
    """Strips unsafe markup from the draft before it is published."""
    cleaned, removed = sanitize_html(state.get("draft", ""))
    if removed:
        print(f"Sanitizer removed: {', '.join(removed)}")
    figures = unsupported_figures(cleaned, state.get("research_notes", []))
    # The hash pins the exact draft the human reviews; the publish step checks it.
    update = {"draft": cleaned, "title": clean_title(state["topic"]), "approved_sha256": content_hash(cleaned),
              "sanitizer_removed": removed, "unsupported_figures": figures, "sender": "sanitizer"}
    if figures:
        print(f"Figures not in the research: {', '.join(figures)}")
        update["review_flag"] = "NEEDS_REVIEW"  # never cleared here: the editor may have set it
    return update


def abort_node(state: AgentState) -> dict:
    """Terminal node for runs whose research never passed the audit."""
    reason = state.get("audit_feedback") or "Unknown research failure."
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
    """Builds the graph: HITL pauses before publishing, and evaluations drop the publish step."""
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("research_branch", research_branch_node)
    workflow.add_node("auditor", auditor_node)
    workflow.add_node("cross_check", cross_check_node)
    workflow.add_node("gap_planner", gap_planner_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("editor", editor_node)
    workflow.add_node("sanitizer", sanitizer_node)
    workflow.add_node("abort", abort_node)
    if include_publisher:
        workflow.add_node("publish", publish_node)

    workflow.set_entry_point("planner")
    workflow.add_conditional_edges("planner", dispatch_research, ["research_branch", "auditor"])
    workflow.add_conditional_edges("gap_planner", dispatch_research, ["research_branch", "auditor"])
    workflow.add_edge("research_branch", "auditor")
    workflow.add_edge("auditor", "cross_check")
    workflow.add_conditional_edges(
        "cross_check",
        audit_router,
        {"writer": "writer", "gap_planner": "gap_planner", "abort": "abort"}
    )

    workflow.add_edge("abort", END)
    workflow.add_edge("writer", "editor")

    # Sanitise on the way out of the editor, so evaluations see the published draft.
    workflow.add_conditional_edges(
        "editor",
        editor_router,
        {"writer": "writer", "sanitizer": "sanitizer"}
    )

    if include_publisher:
        workflow.add_edge("sanitizer", "publish")
        workflow.add_edge("publish", END)
    else:
        workflow.add_edge("sanitizer", END)

    interrupts = ["publish"] if (enable_hitl and include_publisher) else []

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