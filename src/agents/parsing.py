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
