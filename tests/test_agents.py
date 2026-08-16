"""
tests/test_agents.py
Per-agent unit tests. Every LLM and MCP call is mocked; no network, no .env, no credentials.
"""

import pytest

import src.agents.editor as editor
import src.agents.publisher as publisher
import src.agents.researcher as researcher
import src.agents.validator as validator
import src.agents.writer as writer


# ---------------------------------------------------------------- researcher

def test_researcher_stores_findings_and_burns_an_attempt(monkeypatch):
    monkeypatch.setattr(researcher, "_run_research_agent", lambda topic: topic)
    monkeypatch.setattr(researcher.asyncio, "run", lambda _coro: "- fact one")

    result = researcher.researcher_node({"topic": "MCP", "research_notes": []})

    assert result["research_notes"] == ["- fact one"]
    assert result["research_error"] is None
    assert result["research_attempts"] == 1


def test_researcher_reports_the_root_cause_not_the_taskgroup_wrapper(monkeypatch):
    """anyio's ExceptionGroup always stringifies to the same useless sentence."""
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
    """Regression: the error string used to be appended as if it were research."""
    def boom(_coro):
        raise RuntimeError("unhandled errors in a TaskGroup")

    monkeypatch.setattr(researcher, "_run_research_agent", lambda topic: topic)
    monkeypatch.setattr(researcher.asyncio, "run", boom)

    result = researcher.researcher_node({"topic": "MCP", "research_notes": []})

    assert result["research_notes"] == []
    assert "TaskGroup" in result["research_error"]
    assert result["research_attempts"] == 1


def test_llm_is_built_per_call_not_at_import():
    """Regression: a module-level ChatOllama caches an httpx client bound to the first
    event loop, so the second asyncio.run() died with 'Event loop is closed'."""
    assert not hasattr(researcher, "llm")
    assert not hasattr(publisher, "llm")


def test_researcher_skips_search_once_research_is_validated(monkeypatch):
    def fail(_coro):
        raise AssertionError("must not search again after VALIDATED")

    monkeypatch.setattr(researcher.asyncio, "run", fail)

    state = {"topic": "MCP", "research_notes": ["notes"], "validation_status": "VALIDATED"}
    assert researcher.researcher_node(state) == {"sender": "researcher"}


# ----------------------------------------------------------------- validator

def test_validator_returns_verdict_and_feedback(monkeypatch, fake_llm):
    llm = fake_llm('{"status": "VALIDATED", "feedback": "solid sources"}')
    monkeypatch.setattr(validator, "validator_llm", llm)

    result = validator.validator_node({"topic": "MCP", "research_notes": ["a fact"]})

    assert result["validation_status"] == "VALIDATED"
    assert result["validation_feedback"] == "solid sources"
    assert "a fact" in llm.prompts[0]


def test_validator_survives_key_value_inversion(monkeypatch, fake_llm):
    """Observed live: gemma4:12b returned this shape and a VALIDATED run read as REJECTED,
    looping the researcher until the circuit breaker aborted the whole run."""
    mangled = ('{"```json( { ": "status", "value": "VALIDATED", '
               '"feedback": "The research provides specific technical differences."}')
    monkeypatch.setattr(validator, "validator_llm", fake_llm(mangled))

    result = validator.validator_node({"topic": "WebSockets", "research_notes": ["a fact"]})

    assert result["validation_status"] == "VALIDATED"
    assert "technical differences" in result["validation_feedback"]


def test_validator_does_not_read_a_verdict_out_of_prose(monkeypatch, fake_llm):
    """'cannot be validated' in the feedback must not flip the verdict to VALIDATED."""
    payload = '{"status": "REJECTED", "feedback": "The data cannot be validated against the topic."}'
    monkeypatch.setattr(validator, "validator_llm", fake_llm(payload))

    result = validator.validator_node({"topic": "t", "research_notes": ["n"]})

    assert result["validation_status"] == "REJECTED"


def test_editor_survives_key_value_inversion(monkeypatch, fake_llm):
    mangled = '{"```json": "status", "value": "PASS", "feedback": ""}'
    monkeypatch.setattr(editor, "editor_llm", fake_llm(mangled))

    result = editor.editor_node({
        "topic": "t", "research_notes": ["fact"], "draft": "<p>x</p>", "revision_count": 0,
    })

    assert result["last_evaluation"] == "PASS"


def test_validator_rejects_unparseable_output(monkeypatch, fake_llm):
    monkeypatch.setattr(validator, "validator_llm", fake_llm("not json at all"))

    result = validator.validator_node({"topic": "MCP", "research_notes": ["a fact"]})

    assert result["validation_status"] == "REJECTED"
    assert "system error" in result["validation_feedback"]
    # The research loop budget belongs to the researcher, not this node
    assert "revision_count" not in result


def test_validator_surfaces_research_error_in_prompt(monkeypatch, fake_llm):
    llm = fake_llm('{"status": "REJECTED", "feedback": "no data"}')
    monkeypatch.setattr(validator, "validator_llm", llm)

    validator.validator_node({"topic": "MCP", "research_notes": [], "research_error": "search died"})

    assert "search died" in llm.prompts[0]


# -------------------------------------------------------------------- writer

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


# -------------------------------------------------------------------- editor

def test_editor_pass_increments_revision_count(monkeypatch, fake_llm):
    monkeypatch.setattr(editor, "editor_llm", fake_llm('{"status": "PASS", "feedback": ""}'))

    result = editor.editor_node({
        "topic": "MCP", "research_notes": ["fact"], "draft": "<p>x</p>", "revision_count": 1,
    })

    assert result["last_evaluation"] == "PASS"
    assert result["revision_count"] == 2


def test_editor_falls_back_to_fail_on_malformed_json(monkeypatch, fake_llm):
    monkeypatch.setattr(editor, "editor_llm", fake_llm("```json {broken"))

    result = editor.editor_node({
        "topic": "MCP", "research_notes": ["fact"], "draft": "<p>x</p>", "revision_count": 0,
    })

    assert result["last_evaluation"] == "FAIL"
    assert result["revision_count"] == 1
    assert "parse" in result["feedback"]


# ----------------------------------------------------------------- publisher

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
