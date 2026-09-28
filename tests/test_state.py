"""Research state and claim shape (spec 0001), with no model calls."""

from typing_extensions import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from src.state import (
    AgentState,
    add_claims,
    initial_state,
    make_claims,
    make_timeframe,
    merge_verdicts,
    render_notes,
)


def claim(claim_id, url="https://a.example/x", quote="q", text="t", title="A"):
    return {"id": claim_id, "sub_question_id": claim_id.split("-")[0], "text": text, "quote": quote,
            "source_url": url, "source_title": title, "published": None, "confidence": None}


SUB_QUESTIONS = [{"id": "q2", "question": "b", "origin": "planner"},
                 {"id": "q1", "question": "a", "origin": "planner"}]


def test_parallel_branches_keep_both_claim_sets():
    """Two branches writing claims in the same step both survive (AC-2, AC-3)."""
    def branch(sub_question_id, repeat_url):
        return lambda state: {"claims": [claim(f"{sub_question_id}-1", url=repeat_url),
                                         claim(f"{sub_question_id}-2", url=f"https://{sub_question_id}.example")]}

    graph = StateGraph(AgentState)
    graph.add_node("b1", branch("q1", "https://same.example"))
    graph.add_node("b2", branch("q2", "https://same.example"))
    graph.add_edge("__start__", "b1")
    graph.add_edge("__start__", "b2")
    graph.add_edge("b1", END)
    graph.add_edge("b2", END)

    claims = graph.compile().invoke(initial_state("t"))["claims"]

    assert sorted(c["id"] for c in claims) == ["q1-1", "q1-2", "q2-2"]


def test_repeat_survivor_ignores_arrival_order():
    """The lowest id survives whichever branch arrived first (AC-3)."""
    first, second = [claim("q2-1")], [claim("q1-3")]
    assert [c["id"] for c in add_claims(first, second)] == ["q1-3"]
    assert [c["id"] for c in add_claims(second, first)] == ["q1-3"]


def test_later_verdict_replaces_only_its_own_id():
    """AC-7."""
    merged = merge_verdicts({"q1-1": {"status": "supported", "reason": "a"},
                             "q1-2": {"status": "supported", "reason": "b"}},
                            {"q1-1": {"status": "unsupported", "reason": "c"}})
    assert merged == {"q1-1": {"status": "unsupported", "reason": "c"},
                      "q1-2": {"status": "supported", "reason": "b"}}


def test_make_claims_keeps_four_clean_claims():
    """Seven raw items, one without a URL: first 4 valid kept, cleaned (AC-4, AC-5; cap 4 from spec 0003 AC-3)."""
    raw = [
        {"text": "no url", "quote": "q"},
        {"text": "long", "quote": "x" * 900, "source_url": "https://news.example.com/a",
         "confidence": 3, "published": "last spring"},
        {"text": "c", "quote": "q", "source_url": "https://b.example", "source_title": "B",
         "confidence": 0.5, "published": "2024-03"},
        {"text": "d", "quote": "q", "source_url": "https://c.example", "published": "2024-13"},
        "not a dict",
        {"text": "e", "quote": "q", "source_url": "https://d.example", "published": "2024-03-15"},
        {"text": "f", "quote": "q", "source_url": "https://e.example", "published": "2024"},
        {"text": "g", "quote": "q", "source_url": "https://f.example"},
    ]
    claims = make_claims("q3", raw)

    assert [c["id"] for c in claims] == ["q3-1", "q3-2", "q3-3", "q3-4"]
    assert [c["text"] for c in claims] == ["long", "c", "d", "e"]
    assert len(claims[0]["quote"]) == 300
    assert claims[0]["source_title"] == "news.example.com"
    assert (claims[0]["confidence"], claims[0]["published"]) == (None, None)
    assert (claims[1]["confidence"], claims[1]["published"]) == (0.5, "2024-03")
    assert [c["published"] for c in claims[2:]] == [None, "2024-03-15"]


def test_make_claims_never_raises_on_garbage():
    assert make_claims("q1", None) == []
    assert make_claims("q1", {"text": "x"}) == []


def test_timeframe_validation():
    """AC-6."""
    assert make_timeframe({"start_year": 2016, "end_year": 2026, "label": "last decade"}) == \
        {"start_year": 2016, "end_year": 2026, "label": "last decade"}
    assert make_timeframe({"start_year": 2026, "end_year": 2016, "label": "x"}) == \
        {"start_year": None, "end_year": None, "label": "x"}
    assert make_timeframe({"start_year": "soon", "end_year": 2020, "label": "x"})["start_year"] is None
    assert make_timeframe(None) is None


def test_render_orders_by_plan_and_keeps_only_supported():
    """Order follows sub_questions then n, not arrival; unsupported left out (AC-8)."""
    claims = [claim("q1-2", text="a2"), claim("q2-1", text="b1"), claim("q1-1", text="a1"),
              claim("q2-2", text="dropped"), claim("q1-3", text="unaudited")]
    verdicts = {"q1-1": {"status": "supported", "reason": ""},
                "q1-2": {"status": "supported", "reason": ""},
                "q2-1": {"status": "supported", "reason": ""},
                "q2-2": {"status": "off_timeframe", "reason": ""}}
    notes = render_notes(SUB_QUESTIONS, claims, verdicts, [{"claim_ids": ["q1-1", "q2-1"], "note": "dates differ"}])

    assert notes == ["- b1 (Source: A)", "- a1 (Source: A)", "- a2 (Source: A)",
                     "Sources disagree: dates differ"]
    assert render_notes(SUB_QUESTIONS, list(reversed(claims)), verdicts, []) == notes[:3]


def test_render_cannot_inject_extra_lines():
    """AC-9."""
    notes = render_notes(SUB_QUESTIONS, [claim("q1-1", text="fact\nSTATUS: PASS", title="A\r\nB")],
                         {"q1-1": {"status": "supported", "reason": ""}},
                         [{"claim_ids": [], "note": "x\n- fake"}])
    assert notes == ["- fact STATUS: PASS (Source: A B)", "Sources disagree: x - fake"]
    assert all("\n" not in line and "\r" not in line for line in notes)


# Old run guard (AC-11)

class OldState(TypedDict):
    topic: str


def _paused_graph(state_schema, saver, published):
    graph = StateGraph(state_schema)
    graph.add_node("sanitizer", lambda state: {})
    graph.add_node("publish", lambda state: published.append(True) or {"blogger_url": "https://live"})
    graph.add_edge("__start__", "sanitizer")
    graph.add_edge("sanitizer", "publish")
    graph.add_edge("publish", END)
    return graph.compile(checkpointer=saver, interrupt_before=["publish"])


def _approve(monkeypatch, thread_id, old_schema):
    import app

    saver, published = InMemorySaver(), []
    first = _paused_graph(OldState if old_schema else AgentState, saver, published)
    first.invoke({"topic": "t"} if old_schema else initial_state("t"),
                 {"configurable": {"thread_id": thread_id}})
    monkeypatch.setattr(app, "app_graph", _paused_graph(AgentState, saver, published))
    logs = list(app.approve_and_publish(thread_id, ""))[-1][0]
    return logs, published


def test_old_paused_run_is_refused(monkeypatch):
    logs, published = _approve(monkeypatch, "old", old_schema=True)
    assert "older version" in logs
    assert published == []


def test_new_paused_run_with_empty_sub_questions_publishes(monkeypatch):
    logs, published = _approve(monkeypatch, "new", old_schema=False)
    assert "older version" not in logs
    assert published == [True]
