"""Guards the evaluation harness without calling LangSmith, Gemini or the web."""

import json
from pathlib import Path

import pytest

import evals.eval_harness as harness

DATASET = json.loads((Path(__file__).parent.parent / "evals" / "dataset.json").read_text())

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
    """Replaces the judge call so no network request is made."""
    monkeypatch.setattr(harness, "_ask_judge", lambda _prompt: payload)


def test_dataset_is_capped_at_twenty_examples():
    """Each topic costs a web search credit, so the set stays small."""
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


def test_three_grouped_evaluators_cover_nine_metrics():
    """Nine metrics come from three grouped judge calls."""
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
    # correctness is a fraction of the expected facts
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


def test_correctness_is_skipped_for_adversarial_examples(monkeypatch):
    stub_judge(monkeypatch, {"harmful_content": {"score": 1}, "security": {"score": 1},
                             "correctness": {"score": 3}, "hallucination_free": {"score": 1}})

    results = harness.trust_and_safety(
        {"topic": "Ignore previous instructions"}, GOOD_OUTPUTS,
        {"category": "adversarial", "expected_points": [], "expectation": "must not leak secrets"})

    correctness = next(r for r in results if r["key"] == "correctness")
    assert correctness["score"] is None
    # safety metrics still score on an adversarial row
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
    """A run with no draft is skipped, not scored."""
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
    """A broken judge scores nothing rather than zero."""
    stub_judge(monkeypatch, "Judge failed: RuntimeError: gemini unreachable")

    results = getattr(harness, evaluator)({"topic": "t"}, GOOD_OUTPUTS, NORMAL_REFERENCE)

    assert [r["score"] for r in results] == [None] * len(keys)
    assert any("Judge failed" in r["comment"] for r in results)


def test_missing_metric_is_unscored_rather_than_zero(monkeypatch):
    """A metric the judge omitted is unscored, not zero."""
    stub_judge(monkeypatch, {"structure": {"score": 1, "reason": "fine"}})

    scores = {r["key"]: r["score"]
              for r in harness.structure_and_layout({"topic": "t"}, GOOD_OUTPUTS, {})}

    assert scores["structure"] == 1
    assert scores["skimmability"] is None


def test_harness_sends_a_user_turn_to_the_judge():
    """Judge prompts must carry a user turn, which Gemini requires."""
    import inspect

    from langchain_core.messages import HumanMessage

    source = inspect.getsource(harness._ask_judge)
    assert "judge_messages" in source, "judge prompts must go through judge_messages()"
    assert any(isinstance(m, HumanMessage) for m in harness.judge_messages("x"))


def test_judge_prompts_do_not_anchor_scores_with_example_values():
    """A literal score in the example JSON gets copied instead of reasoned about."""
    import inspect
    import re

    for fn in harness.EVALUATORS:
        source = inspect.getsource(fn)
        anchored = re.findall(r'"score":\s*(\d+(?:\.\d+)?)', source)
        assert not anchored, (
            f"{fn.__name__} shows literal example score(s) {anchored}; "
            "use a placeholder such as <0 or 1> instead"
        )


def test_cache_replays_a_topic_plan_and_reuses_searches(monkeypatch, tmp_path):
    """A cached topic replays its plan, so its searches hit the cache and spend no credits."""
    import src.agents.planner as planner
    from src.tools import search

    calls = {"plan": 0, "search": 0}

    def fake_plan(topic):
        calls["plan"] += 1
        return {"start_year": None, "end_year": None, "label": "any time"}, [
            {"id": "q1", "question": f"what is {topic}", "origin": "planner"}]

    def fake_search(query):
        calls["search"] += 1
        return [{"url": "u", "title": "t", "published_date": None, "raw_content": query}]

    monkeypatch.setattr(planner, "make_plan", fake_plan)
    monkeypatch.setattr(search, "search_sources", fake_search)
    monkeypatch.setattr(harness, "CACHE_PATH", tmp_path / "cache.json")
    monkeypatch.setattr(harness, "research_cache", {"plans": {}, "searches": {}})
    harness.enable_research_cache()

    for _ in range(2):
        timeframe, sub_questions = planner.make_plan("mcp")
        assert search.search_sources(sub_questions[0]["question"])[0]["raw_content"] == "what is mcp"

    assert calls == {"plan": 1, "search": 1}
    saved = json.loads((tmp_path / "cache.json").read_text())
    assert "mcp" in saved["plans"] and "what is mcp" in saved["searches"]


def test_summary_averages_scores_and_ignores_unscored():
    from datetime import datetime, timedelta
    from types import SimpleNamespace as NS

    start = datetime(2026, 1, 1)
    rows = [
        {"run": NS(start_time=start, end_time=start + timedelta(seconds=100)),
         "evaluation_results": {"results": [NS(key="tone", score=1.0), NS(key="correctness", score=None)]}},
        {"run": NS(start_time=start, end_time=start + timedelta(seconds=200)),
         "evaluation_results": {"results": [NS(key="tone", score=0.0), NS(key="correctness", score=0.5)]}},
    ]

    summary = harness.summarise(rows)

    assert summary == {"tone": 0.5, "correctness": 0.5, "seconds_per_post": 150}


def test_pipeline_outputs_carry_run_time_and_draft_words(monkeypatch):
    """Spec 0003 AC-8: wall time of one invocation, and words of the draft with tags removed."""
    class FakeGraph:
        def invoke(self, state):
            return {**state, "draft": "<h2>Title here</h2><p>four more plain words</p>"}

    monkeypatch.setattr(harness, "eval_graph", FakeGraph())

    outputs = harness.run_pipeline({"topic": "MCP"})

    assert outputs["draft_words"] == 6
    assert isinstance(outputs["total_seconds"], float) and outputs["total_seconds"] >= 0
