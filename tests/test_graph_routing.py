"""Routing and state tests, with no model calls."""

from src.orchestrator.graph import abort_node, editor_router


def test_abort_node_marks_run_failed_with_reason():
    result = abort_node({"audit_feedback": "3 supported claims across 2 sub questions; need 5 across 3"})
    assert result["run_status"] == "FAILED"
    assert result["sender"] == "abort"


def test_editor_pass_goes_to_sanitizer():
    assert editor_router({"last_evaluation": "PASS", "revision_count": 1}) == "sanitizer"


def test_editor_fail_loops_back_to_writer():
    assert editor_router({"last_evaluation": "FAIL", "revision_count": 1}) == "writer"


def test_editor_circuit_breaker_forces_sanitizer():
    assert editor_router({"last_evaluation": "FAIL", "revision_count": 3}) == "sanitizer"


def test_research_notes_replace_rather_than_accumulate():
    """A new research pass replaces the previous notes."""
    from src.orchestrator.graph import build_graph

    graph = build_graph(enable_hitl=False, include_publisher=False, use_checkpointer=False)
    channel = graph.channels["research_notes"]

    channel.update([["first pass"]])
    channel.update([["second pass"]])
    assert channel.get() == ["second pass"]


def test_sanitizer_pins_the_reviewed_draft_and_cleans_the_title():
    """The hash the publish step checks is taken from the draft shown for review."""
    from src.tools.publish import content_hash
    from src.orchestrator.graph import sanitizer_node

    result = sanitizer_node({"topic": "RAM <script>x</script>prices", "draft": "<p>ok</p><img src=x>"})

    assert result["draft"] == "<p>ok</p>"
    assert result["approved_sha256"] == content_hash("<p>ok</p>")
    assert result["title"] == "RAM prices"
