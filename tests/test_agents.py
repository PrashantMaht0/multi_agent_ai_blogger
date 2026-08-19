"""Per-agent unit tests, with every model and MCP call mocked."""

import pytest

import src.agents.editor as editor
import src.agents.publisher as publisher
import src.agents.researcher as researcher
import src.agents.validator as validator
import src.agents.writer as writer


def test_researcher_stores_findings_and_burns_an_attempt(monkeypatch):
    monkeypatch.setattr(researcher, "_run_research_agent", lambda topic: topic)
    monkeypatch.setattr(researcher.asyncio, "run", lambda _coro: "- fact one")

    result = researcher.researcher_node({"topic": "MCP", "research_notes": []})

    assert result["research_notes"] == ["- fact one"]
    assert result["research_error"] is None
    assert result["research_attempts"] == 1


def test_researcher_reports_the_root_cause_not_the_taskgroup_wrapper(monkeypatch):
    """The reported error is the leaf cause, not the task-group wrapper."""
    def boom(_coro):
        raise ExceptionGroup(
            "unhandled errors in a TaskGroup",
            [ExceptionGroup("unhandled errors in a TaskGroup",
                            [ConnectionError("nodename nor servname provided")])],
        )

    monkeypatch.setattr(researcher, "_run_research_agent", lambda topic: topic)
    monkeypatch.setattr(researcher.asyncio, "run", boom)

    result = researcher.researcher_node({"topic": "MCP", "research_notes": []})

    assert result["research_error"] == "ConnectionError: nodename nor servname provided"
    assert "TaskGroup" not in result["research_error"]


def test_researcher_keeps_failures_out_of_research_notes(monkeypatch):
    """A failed search must not land in research_notes."""
    def boom(_coro):
        raise RuntimeError("unhandled errors in a TaskGroup")

    monkeypatch.setattr(researcher, "_run_research_agent", lambda topic: topic)
    monkeypatch.setattr(researcher.asyncio, "run", boom)

    result = researcher.researcher_node({"topic": "MCP", "research_notes": []})

    assert result["research_notes"] == []
    assert "TaskGroup" in result["research_error"]
    assert result["research_attempts"] == 1


def test_llm_is_built_per_call_not_at_import():
    """A module-level client would outlive the loop asyncio.run() closes."""
    assert not hasattr(researcher, "llm")
    assert not hasattr(publisher, "llm")


def test_researcher_skips_search_once_research_is_validated(monkeypatch):
    def fail(_coro):
        raise AssertionError("must not search again after VALIDATED")

    monkeypatch.setattr(researcher.asyncio, "run", fail)

    state = {"topic": "MCP", "research_notes": ["notes"], "validation_status": "VALIDATED"}
    assert researcher.researcher_node(state) == {"sender": "researcher"}


def test_validator_returns_verdict_and_feedback(monkeypatch, fake_llm):
    llm = fake_llm("STATUS: VALIDATED\nFEEDBACK: solid sources")
    monkeypatch.setattr(validator, "validator_llm", llm)

    result = validator.validator_node({"topic": "MCP", "research_notes": ["a fact"]})

    assert result["validation_status"] == "VALIDATED"
    assert result["validation_feedback"] == "solid sources"
    assert "a fact" in llm.prompts[0]


def test_validator_reads_a_misspelled_verdict(monkeypatch, fake_llm):
    """A misspelled verdict still resolves."""
    monkeypatch.setattr(validator, "validator_llm", fake_llm("STATUS: VALIDED\nFEEDBACK: fine"))

    result = validator.validator_node({"topic": "MCP", "research_notes": ["a fact"]})

    assert result["validation_status"] == "VALIDATED"


def test_validator_does_not_read_a_verdict_out_of_prose(monkeypatch, fake_llm):
    """Wording in the feedback must not flip the verdict."""
    reply = "STATUS: REJECTED\nFEEDBACK: The data cannot be validated against the topic."
    monkeypatch.setattr(validator, "validator_llm", fake_llm(reply))

    result = validator.validator_node({"topic": "t", "research_notes": ["n"]})

    assert result["validation_status"] == "REJECTED"


def test_validator_passes_research_through_when_the_judge_says_nothing(monkeypatch, fake_llm):
    """Silence from the judge is not a rejection."""
    monkeypatch.setattr(validator, "validator_llm", fake_llm(""))

    result = validator.validator_node({"topic": "MCP", "research_notes": ["a fact"]})

    assert result["validation_status"] == "VALIDATED"
    assert "returned nothing" in result["validation_feedback"]
    assert "revision_count" not in result


def test_validator_surfaces_research_error_in_prompt(monkeypatch, fake_llm):
    llm = fake_llm('{"status": "REJECTED", "feedback": "no data"}')
    monkeypatch.setattr(validator, "validator_llm", llm)

    validator.validator_node({"topic": "MCP", "research_notes": [], "research_error": "search died"})

    assert "search died" in llm.prompts[0]


def test_writer_drafts_from_research_and_feedback(monkeypatch, fake_llm):
    llm = fake_llm("<h2>Draft</h2>")
    monkeypatch.setattr(writer, "writer_llm", llm)

    result = writer.writer_node({
        "topic": "MCP",
        "research_notes": ["fact A", "fact B"],
        "feedback": "add more detail",
    })

    assert result["draft"] == "<h2>Draft</h2>"
    assert result["sender"] == "writer"
    prompt = llm.prompts[0]
    assert "fact A" in prompt and "fact B" in prompt and "add more detail" in prompt


def test_editor_pass_increments_revision_count(monkeypatch, fake_llm):
    monkeypatch.setattr(editor, "editor_llm", fake_llm('{"status": "PASS", "feedback": ""}'))

    result = editor.editor_node({
        "topic": "MCP", "research_notes": ["fact"], "draft": "<p>x</p>", "revision_count": 1,
    })

    assert result["last_evaluation"] == "PASS"
    assert result["revision_count"] == 2


def test_editor_passes_draft_through_when_the_judge_says_nothing(monkeypatch, fake_llm):
    """Silence from the judge is not a verdict on the draft."""
    monkeypatch.setattr(editor, "editor_llm", fake_llm(""))

    result = editor.editor_node({
        "topic": "MCP", "research_notes": ["fact"], "draft": "<p>x</p>", "revision_count": 0,
    })

    assert result["last_evaluation"] == "PASS"
    assert result["revision_count"] == 1
    assert "returned nothing" in result["feedback"]


def test_editor_reads_the_line_contract(monkeypatch, fake_llm):
    monkeypatch.setattr(editor, "editor_llm",
                        fake_llm("STATUS: FAIL\nFEEDBACK: a script tag near the end"))

    result = editor.editor_node({
        "topic": "MCP", "research_notes": ["fact"], "draft": "<p>x</p>", "revision_count": 1,
    })

    assert result["last_evaluation"] == "FAIL"
    assert result["feedback"] == "a script tag near the end"
    assert result["revision_count"] == 2


def test_publisher_returns_live_url(monkeypatch):
    monkeypatch.setattr(publisher, "_publish_via_mcp", lambda title, draft: (title, draft))
    monkeypatch.setattr(publisher.asyncio, "run", lambda _coro: "https://example.blogspot.com/post")

    result = publisher.publisher_node({"topic": "MCP", "draft": "<p>x</p>"})

    assert result["blogger_url"] == "https://example.blogspot.com/post"
    assert result["sender"] == "publisher"


def test_publisher_reports_failure_without_raising(monkeypatch):
    def boom(_coro):
        raise RuntimeError("oauth token missing")

    monkeypatch.setattr(publisher, "_publish_via_mcp", lambda title, draft: (title, draft))
    monkeypatch.setattr(publisher.asyncio, "run", boom)

    result = publisher.publisher_node({"topic": "MCP", "draft": "<p>x</p>"})

    assert result["blogger_url"] == "Failed to publish."
