"""
app.py
Gradio Frontend Dashboard for the AI Blogger Multi-Agent System.
Supports real-time agent execution streaming, draft inspection, and Human-in-the-Loop (HITL) approval.
"""

import os
import atexit
import threading
import uuid
import gradio as gr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the compiled LangGraph application
from src.orchestrator.graph import build_graph

# Initialize graph with HITL enabled
app_graph = build_graph(enable_hitl=True)


# ==========================================
# Run lifecycle tracking
# ==========================================
_stop_flags: dict[str, threading.Event] = {}      # thread_id -> stop signal
_session_threads: dict[str, set[str]] = {}        # gradio session_hash -> thread_ids
_paused_threads: set[str] = set()                 # parked at HITL, safe to resume later
_active_thread: str | None = None                 # only one run may be in flight


def _register_run(thread_id: str, session_hash: str | None) -> threading.Event:
    global _active_thread
    _active_thread = thread_id
    _stop_flags[thread_id] = threading.Event()
    _session_threads.setdefault(session_hash or "unknown", set()).add(thread_id)
    return _stop_flags[thread_id]


def _release_run(thread_id: str):
    global _active_thread
    _stop_flags.pop(thread_id, None)
    _paused_threads.discard(thread_id)
    for threads in _session_threads.values():
        threads.discard(thread_id)
    if _active_thread == thread_id:
        _active_thread = None


def _discard_thread(thread_id: str):
    """Deletes the persisted checkpoint so a cancelled run can never be resumed."""
    checkpointer = getattr(app_graph, "checkpointer", None)
    if checkpointer is not None:
        try:
            checkpointer.delete_thread(thread_id)
        except Exception as e:
            print(f"Could not delete checkpoint for {thread_id}: {e}")
    _release_run(thread_id)


def request_stop(thread_id: str, existing_logs: str):
    """Signals the running workflow to halt after the current agent finishes."""
    if not thread_id or thread_id not in _stop_flags:
        return existing_logs + "\n[STOP] No active run to stop.\n", gr.update(), gr.update()

    _stop_flags[thread_id].set()
    logs = existing_logs + "\n[STOP] Stop requested. Halting after the current agent finishes...\n"
    # New Blog is enabled straight away so the dashboard can never deadlock if the
    # generator is torn down (GeneratorExit) before it reaches its terminal yield.
    return logs, gr.update(interactive=False), gr.update(interactive=True)


def reset_dashboard():
    """Clears the workspace so a new blog run can be started."""
    return (
        "", "", "", "", "",
        gr.update(interactive=True),    # generate
        gr.update(interactive=False),   # stop
        gr.update(interactive=False),   # publish
        gr.update(interactive=False),   # new blog
        "",                             # active thread id
    )


def _cancel_run(thread_id: str, why: str):
    """Stops an in-flight run. Threads parked at the HITL pause are left resumable."""
    flag = _stop_flags.get(thread_id)
    if flag:
        flag.set()
    if thread_id in _paused_threads:
        print(f"[{why}] Run {thread_id} is awaiting approval; checkpoint kept.")
        _release_run(thread_id)
    else:
        print(f"[{why}] Cancelling run {thread_id}")
        _discard_thread(thread_id)


def on_unload(request: gr.Request):
    """Cancels this browser session's runs when the tab is closed or reloaded."""
    for thread_id in _session_threads.pop(request.session_hash, set()).copy():
        _cancel_run(thread_id, "UNLOAD")


@atexit.register
def _cancel_all_runs():
    """Cancels every tracked run when the application shuts down."""
    for thread_id in list(_stop_flags.keys()):
        _cancel_run(thread_id, "SHUTDOWN")


# ==========================================
# Workflow handlers
# ==========================================
def start_generation(topic: str, request: gr.Request):
    """
    Initializes the agentic workflow for a given topic.
    Runs Researcher -> Validator -> Writer -> Editor loop, then pauses before Publisher.
    """
    idle_buttons = (gr.update(interactive=True), gr.update(interactive=False),
                    gr.update(interactive=False), gr.update(interactive=False))

    if not topic.strip():
        yield "Please enter a valid topic.", "", "", "", *idle_buttons, ""
        return

    if _active_thread is not None:
        yield ("A run is already in progress. Stop it or click 'Start New Blog' first.",
               "", "", "", *idle_buttons, "")
        return

    # Generate a unique thread ID for this execution session
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    stop_flag = _register_run(thread_id, request.session_hash)

    initial_state = {
        "topic": topic,
        "research_notes": [],
        "research_error": None,
        "research_attempts": 0,
        "raw_sources": [],
        "validation_status": None,
        "validation_feedback": None,
        "run_status": None,
        "draft": "",
        "feedback": "",
        "last_evaluation": None,
        "blogger_url": None,
        "revision_count": 0,
        "sender": "user"
    }

    logs = f"[SESSION] Thread ID: {thread_id}\n"
    logs += f"[TOPIC] {topic}\n\n"
    current_draft = ""
    latest_feedback = ""

    # running: generate off, stop on, publish off, new blog off
    running_buttons = (gr.update(interactive=False), gr.update(interactive=True),
                       gr.update(interactive=False), gr.update(interactive=False))
    yield logs, current_draft, latest_feedback, "", *running_buttons, thread_id

    try:
        # Stream updates synchronously
        for event in app_graph.stream(initial_state, config=config, stream_mode="updates"):
            for node_name, state_update in event.items():
                logs += f"[{node_name.upper()}] Node completed.\n"

                if node_name == "researcher":
                    notes_count = len(state_update.get("research_notes", []))
                    research_error = state_update.get("research_error")
                    logs += f"   - Gathered research items: {notes_count}\n"
                    if research_error:
                        logs += f"   - Search failed: {research_error}\n"
                    logs += "\n"

                elif node_name == "validator":
                    val_status = state_update.get("validation_status", "UNKNOWN")
                    logs += f"   - Status: {val_status}\n\n"

                elif node_name == "writer":
                    current_draft = state_update.get("draft", current_draft)
                    logs += "   - Draft generated / updated.\n\n"

                elif node_name == "editor":
                    eval_status = state_update.get("last_evaluation")
                    # The editor returns empty feedback on PASS, which would leave the box blank
                    latest_feedback = state_update.get("feedback") or (
                        "PASS - no revisions requested." if eval_status == "PASS"
                        else "No feedback provided."
                    )
                    rev_count = state_update.get("revision_count", 1)
                    logs += f"   - Decision: {eval_status} | Loop Count: {rev_count}\n"
                    logs += f"   - Feedback: {latest_feedback}\n\n"

                yield logs, current_draft, latest_feedback, "", *running_buttons, thread_id

            if stop_flag.is_set():
                logs += "[STOPPED] Workflow terminated by user. Checkpoint discarded.\n"
                _discard_thread(thread_id)
                terminal = (gr.update(interactive=False), gr.update(interactive=False),
                            gr.update(interactive=False), gr.update(interactive=True))
                yield logs, current_draft, latest_feedback, "Run stopped.", *terminal, ""
                return

        # Check if the graph paused at an interrupt (before publisher)
        state_snapshot = app_graph.get_state(config)
        final_state = state_snapshot.values

        if state_snapshot.next and "publisher" in state_snapshot.next:
            logs += "[PAUSED] Draft approved by Editor. Awaiting human review.\n"
            _paused_threads.add(thread_id)
            paused = (gr.update(interactive=False), gr.update(interactive=False),
                      gr.update(interactive=True), gr.update(interactive=True))
            yield logs, current_draft, latest_feedback, "Ready for human review.", *paused, thread_id
            return

        terminal = (gr.update(interactive=False), gr.update(interactive=False),
                    gr.update(interactive=False), gr.update(interactive=True))

        if final_state.get("run_status") == "FAILED":
            reason = final_state.get("research_error") or final_state.get("validation_feedback") or "Unknown failure."
            logs += f"[FAILED] Research could not be validated. Reason: {reason}\n"
            _discard_thread(thread_id)
            yield logs, current_draft, latest_feedback, f"Run failed: {reason}", *terminal, ""
            return

        logs += "[COMPLETE] Execution finished.\n"
        _release_run(thread_id)
        yield logs, current_draft, latest_feedback, "Finished without pending actions.", *terminal, thread_id

    except Exception as e:
        logs += f"\n[ERROR] Execution failed: {str(e)}\n"
        _discard_thread(thread_id)
        terminal = (gr.update(interactive=False), gr.update(interactive=False),
                    gr.update(interactive=False), gr.update(interactive=True))
        yield logs, current_draft, latest_feedback, f"Error: {str(e)}", *terminal, ""

    finally:
        # Gradio tears the generator down with GeneratorExit when the tab reloads or the
        # event is cancelled. Without this the run would stay 'active' and block the next one.
        if _active_thread == thread_id and thread_id not in _paused_threads:
            _discard_thread(thread_id)


def approve_and_publish(thread_id: str, existing_logs: str):
    """
    Resumes the paused LangGraph workflow from the database checkpoint.
    """
    if not thread_id:
        yield (existing_logs + "\n[WARN] No active session thread found to resume.", "",
               gr.update(interactive=False), gr.update(interactive=True))
        return

    config = {"configurable": {"thread_id": thread_id}}
    logs = existing_logs + "\n[RESUMING] Human approved. Triggering Publisher Agent...\n"
    blogger_url = ""

    snapshot = app_graph.get_state(config)
    if not (snapshot.next and "publisher" in snapshot.next):
        logs += ("[ERROR] This checkpoint is no longer paused before the publisher "
                 "(it was cancelled or already published). Start a new run.\n")
        _release_run(thread_id)
        yield logs, "Nothing to publish.", gr.update(interactive=False), gr.update(interactive=True)
        return

    yield logs, blogger_url, gr.update(interactive=False), gr.update(interactive=False)

    try:
        published = False
        # Resume synchronous execution from the stored checkpoint
        for event in app_graph.stream(None, config=config, stream_mode="updates"):
            for node_name, state_update in event.items():
                if node_name == "publisher":
                    published = True
                    blogger_url = state_update.get("blogger_url", "URL not returned")
                    logs += f"[PUBLISHED] Live URL: {blogger_url}\n"

        if not published:
            logs += "[WARN] Publisher node produced no output.\n"

        logs += "[COMPLETE] Workflow finished successfully.\n"
        _release_run(thread_id)
        yield logs, blogger_url, gr.update(interactive=False), gr.update(interactive=True)

    except Exception as e:
        logs += f"\n[ERROR] Failed to publish: {str(e)}\n"
        yield logs, f"Error: {str(e)}", gr.update(interactive=True), gr.update(interactive=True)


def _load_theme():
    """Cartoon theme is fetched from the HF hub; fall back if the host is offline."""
    try:
        return gr.Theme.from_hub("harsh8001/cartoon-style")
    except Exception as e:
        print(f"Could not load hub theme, falling back to Soft: {e}")
        return gr.themes.Soft()


# ==========================================
# Gradio UI Layout Definition
# ==========================================
# The cartoon theme's large radii plus `overflow: hidden` on each block clip text at the
# rounded corners (the leading letter of the heading disappears), so tighten them here.
_CSS = """
.gradio-container { padding: 20px 24px !important; max-width: 1400px; }
.gradio-container { --radius-sm: 4px; --radius-md: 6px; --radius-lg: 8px;
                    --radius-xl: 10px; --radius-xxl: 12px; }
.block { overflow: visible !important; }
.prose, .md { padding-left: 6px; }
"""

with gr.Blocks(title="AI Blogger - Multi-Agent Studio") as demo:
    active_thread_id = gr.State("")

    gr.Markdown(
        """
        # Multi-Agent Blogger Studio
        **LangGraph Orchestration** with gemma4 (Editor & Validator) & qwen3 (Researcher, Writer, Publisher).
        """
    )

    topic_input = gr.Textbox(
        label="Article Topic / Seed Idea",
        placeholder="e.g., Deep dive into Model Context Protocol (MCP) in multi-agent systems",
        lines=1,
        max_lines=2
    )

    with gr.Row():
        generate_btn = gr.Button("Generate", variant="primary", size="sm", scale=1)
        stop_btn = gr.Button("Stop", variant="stop", size="sm", scale=1, interactive=False)
        publish_btn = gr.Button("Approve & Publish", variant="secondary", size="sm", scale=1, interactive=False)
        new_blog_btn = gr.Button("New Blog", variant="secondary", size="sm", scale=1, interactive=False)

    with gr.Row(equal_height=True):
        with gr.Column(scale=2):
            gr.Markdown("### Real-Time Pipeline Traces")
            agent_logs = gr.TextArea(
                label="Agent Execution Flow",
                lines=10,
                max_lines=10,
                interactive=False,
                autoscroll=True
            )
            editor_feedback_display = gr.Textbox(
                label="Latest Editor Feedback",
                lines=2,
                max_lines=4,
                interactive=False
            )

        with gr.Column(scale=3):
            gr.Markdown("### Content Inspection")
            draft_display = gr.TextArea(
                label="Generated HTML / Markdown Draft",
                lines=10,
                max_lines=10,
                interactive=True
            )
            deployment_status = gr.Textbox(
                label="Blogger Deployment Status / Live URL",
                lines=2,
                max_lines=2,
                interactive=False
            )

    # Wire Event Handlers
    generate_btn.click(
        fn=start_generation,
        inputs=[topic_input],
        outputs=[
            agent_logs,
            draft_display,
            editor_feedback_display,
            deployment_status,
            generate_btn,
            stop_btn,
            publish_btn,
            new_blog_btn,
            active_thread_id
        ]
    )

    stop_btn.click(
        fn=request_stop,
        inputs=[active_thread_id, agent_logs],
        outputs=[agent_logs, stop_btn, new_blog_btn]
    )

    publish_btn.click(
        fn=approve_and_publish,
        inputs=[active_thread_id, agent_logs],
        outputs=[agent_logs, deployment_status, publish_btn, new_blog_btn]
    )

    new_blog_btn.click(
        fn=reset_dashboard,
        inputs=[],
        outputs=[
            topic_input,
            agent_logs,
            draft_display,
            editor_feedback_display,
            deployment_status,
            generate_btn,
            stop_btn,
            publish_btn,
            new_blog_btn,
            active_thread_id
        ]
    )

    demo.unload(on_unload)

if __name__ == "__main__":
    server_port = int(os.getenv("GRADIO_SERVER_PORT", 7860))
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    # Gradio 6 moved `theme` and `css` from the Blocks constructor to launch()
    demo.queue().launch(server_name=server_name, server_port=server_port,
                        theme=_load_theme(), css=_CSS)
