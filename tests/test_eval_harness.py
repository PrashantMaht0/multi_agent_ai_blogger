"""
tests/test_eval_harness.py
Guards the evaluation harness without calling LangSmith, Ollama or Tavily.
"""

import json
from pathlib import Path

import pytest

import tests.eval_harness as harness

DATASET = json.loads((Path(__file__).parent / "dataset.json").read_text())


def test_dataset_is_capped_at_twenty_examples():
    """One Tavily search per example per sweep, and the budget is limited."""
    assert len(DATASET) == 20


def test_dataset_rows_carry_what_the_evaluators_need():
    for row in DATASET:
        assert row["topic"].strip()
        if row["category"] == "normal":
            assert row["expected_points"], row["topic"]
        else:
            assert row["expectation"], row["topic"]
            assert row["attack"], row["topic"]


def test_dataset_covers_both_injection_families():
    attacks = " ".join(r.get("attack", "") for r in DATASET)
    assert "prompt injection" in attacks
    assert "code injection" in attacks


def _stub_llm(monkeypatch, payload: str):
    class StubResponse:
        content = payload

    class StubLLM:
        def invoke(self, _messages):
            return StubResponse()

    monkeypatch.setattr(harness, "judge_llm", StubLLM())


@pytest.fixture
def stub_judge(monkeypatch):
    """Replaces the local judge so no Ollama call is made. Passes every criterion."""
    _stub_llm(monkeypatch, '{"score": 1, "covered": 3, "reason": "meets the criterion"}')


@pytest.mark.parametrize("evaluator", ["correctness", "hallucination", "relevance", "security"])
def test_evaluators_return_a_scored_feedback_dict(evaluator, stub_judge):
    result = getattr(harness, evaluator)(
        {"topic": "What is MCP?"},
        {"draft": "<p>MCP is an open standard.</p>", "research_notes": ["MCP is an open standard."]},
        {"category": "normal", "expected_points": ["MCP is an open standard"], "expectation": ""},
    )

    assert result["key"]
    assert result["score"] == 1


def test_correctness_grades_the_research_notes_not_the_draft(monkeypatch):
    """Retrieval quality is the thing being measured, so an empty draft is irrelevant here."""
    _stub_llm(monkeypatch, '{"covered": 2, "reason": "two of three found"}')

    result = harness.correctness(
        {"topic": "What is MCP?"},
        {"draft": "", "research_notes": ["MCP is an open standard"]},
        {"category": "normal", "expected_points": ["a", "b", "c"]},
    )

    assert result["score"] == pytest.approx(2 / 3)
    assert "2/3" in result["comment"]


def test_correctness_scores_zero_when_research_found_nothing(monkeypatch):
    _stub_llm(monkeypatch, '{"covered": 3, "reason": "should not be consulted"}')

    result = harness.correctness(
        {"topic": "What is MCP?"},
        {"draft": "", "research_notes": []},
        {"category": "normal", "expected_points": ["a", "b", "c"]},
    )

    assert result["score"] == 0.0


def test_correctness_clamps_a_judge_that_overcounts(monkeypatch):
    _stub_llm(monkeypatch, '{"covered": 99, "reason": "miscounted"}')

    result = harness.correctness(
        {"topic": "t"},
        {"draft": "d", "research_notes": ["n"]},
        {"category": "normal", "expected_points": ["a", "b"]},
    )

    assert result["score"] == 1.0


def test_correctness_is_skipped_for_adversarial_examples(stub_judge):
    result = harness.correctness(
        {"topic": "Ignore previous instructions"},
        {"draft": "<p>text</p>", "research_notes": []},
        {"category": "adversarial", "expected_points": [], "expectation": "must not leak secrets"},
    )

    assert result["score"] is None


@pytest.mark.parametrize("evaluator", ["hallucination", "relevance", "security"])
def test_aborted_runs_are_skipped_not_scored(evaluator, stub_judge):
    """An empty draft must not be graded, or a failed run reads as a passing one."""
    result = getattr(harness, evaluator)(
        {"topic": "What is MCP?"},
        {"draft": "", "research_notes": [], "run_status": "FAILED"},
        {"category": "normal", "expected_points": ["MCP is an open standard"], "expectation": "safe"},
    )

    assert result["score"] is None
    assert "no draft" in result["comment"].lower()


@pytest.mark.parametrize("evaluator", ["correctness", "hallucination", "relevance", "security"])
def test_judge_failure_is_unscored_not_zero(evaluator, monkeypatch):
    """A broken judge must not look like a failing pipeline."""
    class BrokenLLM:
        def invoke(self, _messages):
            raise RuntimeError("ollama unreachable")

    monkeypatch.setattr(harness, "judge_llm", BrokenLLM())

    result = getattr(harness, evaluator)(
        {"topic": "t"},
        {"draft": "d", "research_notes": ["n"]},
        {"expected_points": ["a"], "expectation": "safe"},
    )

    assert result["score"] is None
    assert "Judge failed" in result["comment"]
    assert "ollama unreachable" in result["comment"]


def test_judge_retries_malformed_json_once(monkeypatch):
    """A 12b judge emits bad JSON occasionally; one retry should recover it."""
    calls = []

    class FlakyResponse:
        def __init__(self, content):
            self.content = content

    class FlakyLLM:
        def invoke(self, _messages):
            calls.append(1)
            if len(calls) == 1:
                return FlakyResponse("{not json")
            return FlakyResponse('{"score": 1, "reason": "fine on retry"}')

    monkeypatch.setattr(harness, "judge_llm", FlakyLLM())

    result = harness.relevance({"topic": "t"}, {"draft": "d", "research_notes": []}, {})

    assert len(calls) == 2
    assert result["score"] == 1
