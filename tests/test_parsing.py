"""Verdict extraction from judge replies."""

import pytest

from src.agents.parsing import parse_verdict_lines

VERDICTS = ("VALIDATED", "REJECTED")


def test_reads_the_two_line_contract():
    verdict, feedback = parse_verdict_lines(
        "STATUS: VALIDATED\nFEEDBACK: sources are concrete", VERDICTS, "REJECTED")

    assert verdict == "VALIDATED"
    assert feedback == "sources are concrete"


def test_prefix_match_survives_a_misspelled_verdict():
    """A misspelled verdict still resolves."""
    verdict, _ = parse_verdict_lines("STATUS: VALIDED\nFEEDBACK: fine", VERDICTS, "REJECTED")
    assert verdict == "VALIDATED"


def test_feedback_wording_cannot_flip_the_verdict():
    """Wording in the feedback must not flip the verdict."""
    reply = "STATUS: REJECTED\nFEEDBACK: The data cannot be validated against the topic."
    verdict, _ = parse_verdict_lines(reply, VERDICTS, "REJECTED")
    assert verdict == "REJECTED"


def test_tolerates_markdown_decoration():
    verdict, feedback = parse_verdict_lines(
        "**STATUS:** FAIL\n- FEEDBACK: a script tag near the end", ("PASS", "FAIL"), "FAIL")

    assert verdict == "FAIL"
    assert feedback == "a script tag near the end"


def test_falls_back_to_a_verdict_stated_without_the_label():
    verdict, _ = parse_verdict_lines("PASS - the draft is clean", ("PASS", "FAIL"), "FAIL")
    assert verdict == "PASS"


def test_unrecognised_reply_uses_the_default_and_keeps_the_text():
    verdict, feedback = parse_verdict_lines("I am not sure about this one", VERDICTS, "REJECTED")

    assert verdict == "REJECTED"
    assert "not sure" in feedback


@pytest.mark.parametrize("raw", ["", None])
def test_empty_reply_uses_the_default(raw):
    verdict, feedback = parse_verdict_lines(raw, VERDICTS, "REJECTED")

    assert verdict == "REJECTED"
    assert feedback == ""


def test_judge_messages_include_a_user_turn():
    """Judge prompts carry a user turn, which Gemini requires."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from src.agents.parsing import judge_messages

    messages = judge_messages("grade this")

    assert isinstance(messages[0], SystemMessage)
    assert any(isinstance(m, HumanMessage) for m in messages), "Gemini needs a user turn"
