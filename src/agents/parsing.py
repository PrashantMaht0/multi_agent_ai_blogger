"""Helpers for reading a judge model's reply."""

import json

from langchain_core.messages import HumanMessage, SystemMessage


def prompt_messages(prompt: str, ask: str) -> list:
    """Wraps a prompt as a system turn plus a user turn."""
    return [SystemMessage(content=prompt), HumanMessage(content=ask)]


def judge_messages(prompt: str) -> list:
    """Wraps a judging prompt as messages."""
    return prompt_messages(prompt, "Give your verdict now.")


def message_text(response) -> str:
    """Returns a reply's text, whether the provider sent a string or content blocks."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content or "")


def parse_verdict_lines(raw: str, options: tuple[str, ...], default: str) -> tuple[str, str]:
    """Reads a `STATUS: X` / `FEEDBACK: ...` reply into (verdict, feedback)."""
    verdict, feedback, found = default, "", False

    for line in (raw or "").splitlines():
        stripped = line.strip().lstrip("*-# ").strip()
        upper = stripped.upper()
        if not found and upper.startswith("STATUS"):
            value = stripped.split(":", 1)[-1].strip().upper()
            for option in options:
                if value.startswith(option[:4]):
                    verdict, found = option, True
                    break
        elif upper.startswith("FEEDBACK"):
            feedback = stripped.split(":", 1)[-1].strip()

    if found:
        return verdict, feedback

    # No STATUS line: look for a verdict before the feedback text.
    head = (raw or "").upper().split("FEEDBACK", 1)[0]
    for option in options:
        if option[:4] in head:
            return option, feedback or (raw or "").strip()[:300]

    return default, feedback or (raw or "").strip()[:300]


def judge_text(llm, messages) -> str:
    """Calls a judge and returns its raw reply, or '' if the call failed."""
    try:
        return message_text(llm.invoke(messages))
    except Exception as e:
        print(f"Judge call failed: {type(e).__name__}: {e}")
        return ""


def judge_json(llm, messages, attempts: int = 2):
    """Calls a judge and returns parsed JSON, retrying once, or None if it never parses."""
    for attempt in range(attempts):
        raw = ""
        try:
            raw = message_text(llm.invoke(messages))
            return json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"Judge returned unparseable JSON (attempt {attempt + 1}/{attempts}): "
                  f"{e} | raw={raw[:160]!r}")
    return None


def _verdict_candidates(decision) -> list[str]:
    """Collects the single-word strings from a payload, documented key first."""
    if not isinstance(decision, dict):
        return []

    ordered = []
    status = decision.get("status")
    if isinstance(status, str):
        ordered.append(status)
    ordered.extend(v for v in decision.values() if isinstance(v, str))
    ordered.extend(k for k in decision.keys() if isinstance(k, str))

    return [s.strip() for s in ordered if len(s.split()) == 1]


def extract_verdict(decision, options: tuple[str, ...], default: str) -> str:
    """Returns whichever option the payload names, matched on a prefix, else the default."""
    for candidate in _verdict_candidates(decision):
        upper = candidate.upper()
        for option in options:
            if upper.startswith(option[:4]):
                return option
    return default


def extract_feedback(decision, fallback: str = "") -> str:
    """Returns the payload's explanation: the documented key, else its longest sentence."""
    if not isinstance(decision, dict):
        return fallback

    feedback = decision.get("feedback")
    if isinstance(feedback, str) and feedback.strip():
        return feedback

    sentences = [v for v in decision.values() if isinstance(v, str) and len(v.split()) > 1]
    return max(sentences, key=len) if sentences else fallback
