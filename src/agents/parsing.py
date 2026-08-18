"""
src/agents/parsing.py
Pulls a verdict out of the JSON a local judge model returns.

gemma4:12b under format="json" produces structurally valid JSON that does not always
match the requested shape. Observed in the wild:
    {"status": "VALIDED", ...}                       - misspelled verdict
    {"```json( { ": "status", "value": "VALIDATED"}   - key/value inversion
The second parses cleanly, so decision.get("status") returns None and a VALIDATED run is
read as REJECTED. These helpers look past the shape for the verdict itself.
"""

import json

from langchain_core.messages import HumanMessage, SystemMessage


def prompt_messages(prompt: str, ask: str) -> list:
    """Message list any agent prompt is sent as.

    Gemini rejects a request whose only message is a SystemMessage with
    "ValueError: contents are required", so instructions go in the system turn and a short
    user turn asks for the output. Ollama is happy with the same shape, which keeps every
    agent portable between the two providers.
    """
    return [SystemMessage(content=prompt), HumanMessage(content=ask)]


def judge_messages(prompt: str) -> list:
    """Message list for a judging prompt."""
    return prompt_messages(prompt, "Give your verdict now.")


def message_text(response) -> str:
    """Returns a reply's text whatever shape the provider used.

    Ollama sets .content to a string. Gemini sets it to a list of content blocks,
    e.g. [{"type": "text", "text": "STATUS: PASS", "extras": {...}}], so reading
    .content directly yields a list and every downstream parser breaks.
    """
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
    """Reads a `STATUS: X` / `FEEDBACK: ...` reply, returning (verdict, feedback).

    Used instead of JSON because gemma4:12b under format="json" reliably answers with a
    single {"thought": "..."} object and stops, never emitting the verdict - no wording
    of the prompt prevented it. A line format has no grammar to degenerate into.
    """
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

    # No STATUS line. Fall back to the first option named anywhere in the reply - but only
    # scan the part before FEEDBACK, so wording like "cannot be validated" in an
    # explanation cannot decide the verdict.
    head = (raw or "").upper().split("FEEDBACK", 1)[0]
    for option in options:
        if option[:4] in head:
            return option, feedback or (raw or "").strip()[:300]

    return default, feedback or (raw or "").strip()[:300]


def judge_text(llm, messages) -> str:
    """Single call returning the raw reply, or '' if the model produced nothing."""
    try:
        return message_text(llm.invoke(messages))
    except Exception as e:
        print(f"Judge call failed: {type(e).__name__}: {e}")
        return ""


def judge_json(llm, messages, attempts: int = 2):
    """Invokes a judge and returns parsed JSON, retrying once on malformed output.

    A response that will not parse is the judge malfunctioning, not a verdict. Treating
    it as one costs a whole revision loop (editor) or a research attempt (validator);
    with a two-attempt budget a single hiccup could abort an otherwise healthy run.
    Returns None when every attempt fails.
    """
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
    """Single-word strings from the payload, documented key first."""
    if not isinstance(decision, dict):
        return []

    ordered = []
    status = decision.get("status")
    if isinstance(status, str):
        ordered.append(status)
    ordered.extend(v for v in decision.values() if isinstance(v, str))
    ordered.extend(k for k in decision.keys() if isinstance(k, str))

    # A verdict is one word. Skipping longer strings keeps a feedback sentence such as
    # "the data cannot be validated" from being mistaken for the verdict.
    return [s.strip() for s in ordered if len(s.split()) == 1]


def extract_verdict(decision, options: tuple[str, ...], default: str) -> str:
    """Returns whichever of `options` the payload names, else `default`.

    Matching is on a prefix so near-misses ("VALIDED" for "VALIDATED") still resolve.
    """
    for candidate in _verdict_candidates(decision):
        upper = candidate.upper()
        for option in options:
            if upper.startswith(option[:4]):
                return option
    return default


def extract_feedback(decision, fallback: str = "") -> str:
    """Returns the explanation: the documented key, else the longest sentence present."""
    if not isinstance(decision, dict):
        return fallback

    feedback = decision.get("feedback")
    if isinstance(feedback, str) and feedback.strip():
        return feedback

    sentences = [v for v in decision.values() if isinstance(v, str) and len(v.split()) > 1]
    return max(sentences, key=len) if sentences else fallback
