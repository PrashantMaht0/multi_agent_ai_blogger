"""
app.py
Gradio Frontend Dashboard for the AI Blogger Multi-Agent System.
Supports real-time agent execution streaming, draft inspection, and Human-in-the-Loop (HITL) approval.
"""

import os
import uuid
import gradio as gr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the compiled LangGraph application
from src.orchestrator.graph import build_graph

# Initialize graph with checkpointing enabled
app_graph = build_graph()


def start_generation(topic: str):
    """
    Initializes the agentic workflow for a given topic.
    Runs Researcher -> Validator -> Writer -> Editor loop, then pauses before Publisher.
    """
    if not topic.strip():
        yield "⚠️ Please enter a valid topic.", "", "", "", gr.update(interactive=False), ""
        return

    # Generate a unique thread ID for this execution session
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    initial_state = {
        "topic": topic,
        "research_notes": [],
        "raw_sources": [],
        "validation_status": None,
        "validation_feedback": None,
        "draft": "",
        "feedback": "",
        "last_evaluation": None,
        "blogger_url": None,
        "revision_count": 0,
        "sender": "user"
    }

    logs = f"🚀 [Session Initialized] Thread ID: {thread_id}\n"
    logs += f"📌 Topic: {topic}\n\n"
    current_draft = ""
    latest_feedback = ""
    
    yield logs, current_draft, latest_feedback, "", gr.update(interactive=False), thread_id

    try:
        # Stream updates synchronously
        for event in app_graph.stream(initial_state, config=config, stream_mode="updates"):
            for node_name, state_update in event.items():
                logs += f"⚙️ Node Completed: [{node_name.upper()}]\n"

                if node_name == "researcher":
                    notes_count = len(state_update.get("research_notes", []))
                    logs += f"   └── Gathered research items: {notes_count}\n\n"
                
                elif node_name == "validator":
                    val_status = state_update.get("validation_status", "UNKNOWN")
                    logs += f"   └── Status: {val_status}\n\n"

                elif node_name == "writer":
                    current_draft = state_update.get("draft", current_draft)
                    logs += "   └── Draft generated / updated.\n\n"

                elif node_name == "editor":
                    eval_status = state_update.get("last_evaluation")
                    latest_feedback = state_update.get("feedback", "No feedback provided.")
                    rev_count = state_update.get("revision_count", 1)
                    logs += f"   └── Decision: {eval_status} | Loop Count: {rev_count}\n"
                    logs += f"   └── Feedback: {latest_feedback}\n\n"

                yield logs, current_draft, latest_feedback, "", gr.update(interactive=False), thread_id

        # Check if the graph paused at an interrupt (before publisher)
        state_snapshot = app_graph.get_state(config)
        if state_snapshot.next and "publisher" in state_snapshot.next:
            logs += "⏸️ [PAUSED] Draft approved by Editor. Awaiting Human Review.\n"
            yield logs, current_draft, latest_feedback, "Ready for human review.", gr.update(interactive=True), thread_id
        else:
            logs += "✅ Execution completed.\n"
            yield logs, current_draft, latest_feedback, "Finished without pending actions.", gr.update(interactive=False), thread_id

    except Exception as e:
        logs += f"\n❌ Error during execution: {str(e)}\n"
        yield logs, current_draft, latest_feedback, f"Error: {str(e)}", gr.update(interactive=False), thread_id


def approve_and_publish(thread_id: str, existing_logs: str):
    """
    Resumes the paused LangGraph workflow from the database checkpoint.
    """
    if not thread_id:
        yield existing_logs + "\n⚠️ No active session thread found to resume.", "", gr.update(interactive=False)
        return

    config = {"configurable": {"thread_id": thread_id}}
    logs = existing_logs + "\n▶️ [RESUMING] Human approved. Triggering Publisher Agent...\n"
    blogger_url = ""
    
    yield logs, blogger_url, gr.update(interactive=False)

    try:
        # Resume synchronous execution from the stored checkpoint
        for event in app_graph.stream(None, config=config, stream_mode="updates"):
            for node_name, state_update in event.items():
                if node_name == "publisher":
                    blogger_url = state_update.get("blogger_url", "URL not returned")
                    logs += f"🌐 Post Published Live: {blogger_url}\n"

        logs += "🎉 [COMPLETE] Workflow finished successfully!\n"
        yield logs, blogger_url, gr.update(interactive=False)

    except Exception as e:
        logs += f"\n❌ Failed to publish: {str(e)}\n"
        yield logs, f"Error: {str(e)}", gr.update(interactive=True)


# ==========================================
# Gradio UI Layout Definition
# ==========================================
with gr.Blocks(title="AI Blogger - Multi-Agent Studio", theme=gr.themes.Soft()) as demo:
    active_thread_id = gr.State("")

    gr.Markdown(
        """
        # 🖋️ AI Blogger Multi-Agent Studio
        **LangGraph Orchestration** with `gemma4:12b` (Editor & Validator) & `llama3:8b` (Researcher, Writer, Publisher).
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            topic_input = gr.Textbox(
                label="Article Topic / Seed Idea",
                placeholder="e.g., Deep dive into Model Context Protocol (MCP) in multi-agent systems",
                lines=2
            )
        with gr.Column(scale=1):
            generate_btn = gr.Button("🚀 Generate Blog Post", variant="primary", scale=2)
            publish_btn = gr.Button("✅ Approve & Publish to Blogger", variant="stop", interactive=False, scale=1)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🔍 Real-Time Pipeline Traces")
            agent_logs = gr.TextArea(
                label="Agent Execution Flow",
                lines=20,
                interactive=False,
                autoscroll=True
            )
            editor_feedback_display = gr.Textbox(
                label="Latest Editor Feedback",
                lines=3,
                interactive=False
            )

        with gr.Column(scale=2):
            gr.Markdown("### 📄 Content Inspection")
            draft_display = gr.TextArea(
                label="Generated HTML / Markdown Draft",
                lines=16,
                interactive=True
            )
            deployment_status = gr.Textbox(
                label="Blogger Deployment Status / Live URL",
                lines=2,
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
            publish_btn,
            active_thread_id
        ]
    )

    publish_btn.click(
        fn=approve_and_publish,
        inputs=[active_thread_id, agent_logs],
        outputs=[
            agent_logs,
            deployment_status,
            publish_btn
        ]
    )

if __name__ == "__main__":
    server_port = int(os.getenv("GRADIO_SERVER_PORT", 7860))
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    demo.queue().launch(server_name=server_name, server_port=server_port)