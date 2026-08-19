"""Routing and state tests, with no model calls."""

from src.orchestrator.graph import (
    MAX_RESEARCH_ATTEMPTS,
    abort_node,
    editor_router,
    validation_router,
)


def test_validated_research_goes_to_writer():
    assert validation_router({"validation_status": "VALIDATED"}) == "writer"


def test_rejected_research_retries_researcher():
    state = {"validation_status": "REJECTED", "research_attempts": 1}
    assert validation_router(state) == "researcher"


def test_exhausted_research_aborts_instead_of_drafting():
    """Exhausted research aborts instead of reaching the writer."""
    state = {
        "validation_status": "REJECTED",
        "research_attempts": MAX_RESEARCH_ATTEMPTS,
        "research_error": "unhandled errors in a TaskGroup",
    }
    assert validation_router(state) == "abort"


def test_research_budget_allows_one_retry_then_stops():
    """The first rejection re-searches, the second gives up."""
    assert MAX_RESEARCH_ATTEMPTS == 2
    assert validation_router({"validation_status": "REJECTED", "research_attempts": 1}) == "researcher"
    assert validation_router({"validation_status": "REJECTED", "research_attempts": 2}) == "abort"


def test_editor_budget_is_separate_from_research_budget():
    """The research loop must not spend the writer's revision budget."""
    state = {"validation_status": "REJECTED", "research_attempts": 3, "revision_count": 0}
    assert validation_router(state) == "abort"
    assert editor_router({"last_evaluation": "FAIL", "revision_count": 0}) == "writer"


def test_abort_node_marks_run_failed_with_reason():
    result = abort_node({"research_error": "search server crashed"})
    assert result["run_status"] == "FAILED"
    assert result["sender"] == "abort"


def test_editor_pass_goes_to_publisher():
    assert editor_router({"last_evaluation": "PASS", "revision_count": 1}) == "publisher"


def test_editor_fail_loops_back_to_writer():
    assert editor_router({"last_evaluation": "FAIL", "revision_count": 1}) == "writer"


def test_editor_circuit_breaker_forces_publisher():
    assert editor_router({"last_evaluation": "FAIL", "revision_count": 3}) == "publisher"


def test_misspelled_verdict_is_normalised_to_validated(monkeypatch):
    """A misspelled verdict is read as VALIDATED."""
    import src.agents.validator as validator

    class FakeResponse:
        content = "STATUS: VALIDED\nFEEDBACK: looks good"

    class FakeLLM:
        def invoke(self, _messages):
            return FakeResponse()

    monkeypatch.setattr(validator, "validator_llm", FakeLLM())
    result = validator.validator_node({"topic": "t", "research_notes": ["notes"]})

    assert result["validation_status"] == "VALIDATED"
    assert validation_router(result) == "writer"


def test_unknown_verdict_is_treated_as_rejection(monkeypatch):
    import src.agents.validator as validator

    class FakeResponse:
        content = "STATUS: MAYBE\nFEEDBACK: unsure"

    class FakeLLM:
        def invoke(self, _messages):
            return FakeResponse()

    monkeypatch.setattr(validator, "validator_llm", FakeLLM())
    result = validator.validator_node({"topic": "t", "research_notes": ["notes"]})

    assert result["validation_status"] == "REJECTED"


def test_researcher_burns_an_attempt_on_any_unaccepted_verdict(monkeypatch):
    """Every researcher pass spends an attempt."""
    import src.agents.researcher as researcher

    monkeypatch.setattr(researcher, "_run_research_agent", lambda topic: topic)
    monkeypatch.setattr(researcher.asyncio, "run", lambda _coro: "fresh findings")

    state = {"topic": "t", "research_notes": ["stale"], "research_attempts": 1,
             "validation_status": "VALIDED"}
    result = researcher.researcher_node(state)

    assert result["research_attempts"] == 2
    assert result["research_notes"] == ["fresh findings"]


def test_research_notes_replace_rather_than_accumulate():
    """A new research pass replaces the previous notes."""
    from src.orchestrator.graph import build_graph

    graph = build_graph(enable_hitl=False, include_publisher=False, use_checkpointer=False)
    channel = graph.channels["research_notes"]

    channel.update([["first pass"]])
    channel.update([["second pass"]])
    assert channel.get() == ["second pass"]
