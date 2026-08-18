"""
tests/test_eval_harness.py
Guards the evaluation harness without calling LangSmith, Gemini or Tavily.
"""

import json
from pathlib import Path

import pytest

import tests.eval_harness as harness

DATASET = json.loads((Path(__file__).parent / "dataset.json").read_text())

GOOD_OUTPUTS = {
    "draft": "<h2>What is MCP?</h2><p>MCP is an open standard.</p>",
    "research_notes": ["MCP is an open standard."],
    "run_status": None,
}
NORMAL_REFERENCE = {
    "category": "normal",
    "expected_points": ["a", "b", "c"],
    "expectation": "",
}


def stub_judge(monkeypatch, payload):
    """Replaces the Gemini call so no network request is made."""
    monkeypatch.setattr(harness, "_ask_judge", lambda _prompt: payload)


# ------------------------------------------------------------------ dataset

def test_dataset_is_capped_at_twenty_examples():
    """One web search per example per sweep, and the budget is limited."""
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


# --------------------------------------------------------------- grouping

def test_three_grouped_evaluators_cover_nine_metrics():
    """Grouping keeps a sweep at 3 judge calls per example instead of one per metric."""
    assert len(harness.EVALUATORS) == 3

    metrics = harness.TRUST_KEYS + harness.EDITORIAL_KEYS + harness.STRUCTURE_KEYS
    assert set(metrics) == {
        "harmful_content", "security", "correctness", "hallucination_free",
        "catchy_headline", "tone", "engagement",
        "structure", "skimmability",
    }


def test_trust_group_returns_every_metric(monkeypatch):
    stub_judge(monkeypatch, {
        "harmful_content": {"score": 1, "reason": "clean"},
        "security": {"score": 1, "reason": "resisted"},
        "correctness": {"score": 2, "reason": "two of three"},
        "hallucination_free": {"score": 1, "reason": "grounded"},
    })

    results = harness.trust_and_safety({"topic": "MCP"}, GOOD_OUTPUTS, NORMAL_REFERENCE)
    scores = {r["key"]: r["score"] for r in results}

    assert scores["harmful_content"] == 1
    assert scores["security"] == 1
    assert scores["hallucination_free"] == 1
    # correctness is reported as a fraction of the expected facts
    assert scores["correctness"] == pytest.approx(2 / 3)


def test_editorial_group_returns_every_metric(monkeypatch):
    stub_judge(monkeypatch, {
        "catchy_headline": {"score": 0, "reason": "dull title"},
        "tone": {"score": 1, "reason": "professional"},
        "engagement": {"score": 1, "reason": "holds attention"},
    })

    scores = {r["key"]: r["score"]
              for r in harness.editorial_experience({"topic": "MCP"}, GOOD_OUTPUTS, {})}

    assert scores == {"catchy_headline": 0, "tone": 1, "engagement": 1}


def test_structure_group_returns_every_metric(monkeypatch):
    stub_judge(monkeypatch, {
        "structure": {"score": 1, "reason": "clear order"},
        "skimmability": {"score": 0, "reason": "walls of text"},
    })

    scores = {r["key"]: r["score"]
              for r in harness.structure_and_layout({"topic": "MCP"}, GOOD_OUTPUTS, {})}

    assert scores == {"structure": 1, "skimmability": 0}


# ------------------------------------------------------------ edge cases

def test_correctness_is_skipped_for_adversarial_examples(monkeypatch):
    stub_judge(monkeypatch, {"harmful_content": {"score": 1}, "security": {"score": 1},
                             "correctness": {"score": 3}, "hallucination_free": {"score": 1}})

    results = harness.trust_and_safety(
        {"topic": "Ignore previous instructions"}, GOOD_OUTPUTS,
        {"category": "adversarial", "expected_points": [], "expectation": "must not leak secrets"})

    correctness = next(r for r in results if r["key"] == "correctness")
    assert correctness["score"] is None
    # the safety metrics still score on an adversarial row
    assert next(r for r in results if r["key"] == "security")["score"] == 1


def test_correctness_clamps_a_judge_that_overcounts(monkeypatch):
    stub_judge(monkeypatch, {"correctness": {"score": 99, "reason": "miscounted"}})

    results = harness.trust_and_safety({"topic": "t"}, GOOD_OUTPUTS, NORMAL_REFERENCE)
    correctness = next(r for r in results if r["key"] == "correctness")

    assert correctness["score"] == 1.0


@pytest.mark.parametrize("evaluator,keys", [
    ("trust_and_safety", harness.TRUST_KEYS),
    ("editorial_experience", harness.EDITORIAL_KEYS),
    ("structure_and_layout", harness.STRUCTURE_KEYS),
])
def test_aborted_runs_are_skipped_not_scored(evaluator, keys, monkeypatch):
    """An empty draft must not be graded, or a failed run reads as a passing one."""
    stub_judge(monkeypatch, {k: {"score": 1} for k in keys})

    results = getattr(harness, evaluator)(
        {"topic": "MCP"},
        {"draft": "", "research_notes": [], "run_status": "FAILED"},
        NORMAL_REFERENCE,
    )

    assert [r["score"] for r in results] == [None] * len(keys)
    assert all("no draft" in r["comment"].lower() for r in results)


@pytest.mark.parametrize("evaluator,keys", [
    ("trust_and_safety", harness.TRUST_KEYS),
    ("editorial_experience", harness.EDITORIAL_KEYS),
    ("structure_and_layout", harness.STRUCTURE_KEYS),
])
def test_judge_failure_is_unscored_not_zero(evaluator, keys, monkeypatch):
    """A broken judge must not look like a failing pipeline."""
    stub_judge(monkeypatch, "Judge failed: RuntimeError: gemini unreachable")

    results = getattr(harness, evaluator)({"topic": "t"}, GOOD_OUTPUTS, NORMAL_REFERENCE)

    assert [r["score"] for r in results] == [None] * len(keys)
    assert any("Judge failed" in r["comment"] for r in results)


def test_missing_metric_is_unscored_rather_than_zero(monkeypatch):
    """A judge that answers about two metrics must not silently fail the third."""
    stub_judge(monkeypatch, {"structure": {"score": 1, "reason": "fine"}})

    scores = {r["key"]: r["score"]
              for r in harness.structure_and_layout({"topic": "t"}, GOOD_OUTPUTS, {})}

    assert scores["structure"] == 1
    assert scores["skimmability"] is None
