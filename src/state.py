"""Shared state passed between every node in the graph."""

import operator
import re
from datetime import datetime
from typing import Annotated, List, Optional, Literal
from urllib.parse import urlparse
from typing_extensions import TypedDict

MAX_CLAIMS_PER_BRANCH = 4
MAX_QUOTE_CHARS = 300


class Timeframe(TypedDict):
    start_year: Optional[int]
    end_year: Optional[int]
    label: str


class SubQuestion(TypedDict):
    id: str  # q1, q2, ...
    question: str
    origin: Literal["planner", "gap"]


class Claim(TypedDict):
    id: str  # <sub_question_id>-<n>, e.g. q2-1
    sub_question_id: str
    text: str
    quote: str
    source_url: str
    source_title: str
    published: Optional[str]
    confidence: Optional[float]


class BranchError(TypedDict):
    sub_question_id: str
    error: str


class Verdict(TypedDict):
    status: Literal["supported", "unsupported", "off_timeframe"]
    reason: str


class Contradiction(TypedDict):
    claim_ids: List[str]
    note: str


class Gap(TypedDict):
    question: str
    reason: str
    important: bool


def _claim_order(claim_id: str) -> tuple[int, int]:
    """(sub-question number, n) for an id such as q2-1."""
    sub_question_id, n = claim_id.rsplit("-", 1)
    return int(sub_question_id[1:]), int(n)


def add_claims(existing: List[Claim], new: List[Claim]) -> List[Claim]:
    """Appends claims; of two with the same source_url and quote, the lowest id survives."""
    combined = existing + new
    survivors: dict[tuple[str, str], Claim] = {}
    for claim in combined:
        key = (claim["source_url"], claim["quote"])
        kept = survivors.get(key)
        if kept is None or _claim_order(claim["id"]) < _claim_order(kept["id"]):
            survivors[key] = claim
    return [c for c in combined if survivors[(c["source_url"], c["quote"])] is c]


def merge_verdicts(existing: dict[str, Verdict], new: dict[str, Verdict]) -> dict[str, Verdict]:
    """A later verdict for the same claim id replaces the earlier one."""
    return {**existing, **new}


class AgentState(TypedDict):
    topic: str
    # Replaced on every research pass, so stale notes cannot linger.
    research_notes: List[str]
    run_status: Optional[Literal["FAILED"]]
    # What the sanitizer stripped from the draft.
    sanitizer_removed: List[str]
    # Figures in the final draft that no research note contains (spec 0003).
    unsupported_figures: List[str]

    # Deep research. Fields parallel branches write must append, never replace.
    timeframe: Optional[Timeframe]
    sub_questions: Annotated[List[SubQuestion], operator.add]
    claims: Annotated[List[Claim], add_claims]
    branch_errors: Annotated[List[BranchError], operator.add]
    claim_verdicts: Annotated[dict[str, Verdict], merge_verdicts]
    contradictions: List[Contradiction]
    gaps: List[Gap]
    audit_status: Optional[Literal["PASSED", "FAILED"]]
    audit_feedback: Optional[str]
    audit_loops: int

    draft: str
    title: str
    # SHA-256 of the sanitised draft shown for review.
    approved_sha256: Optional[str]
    # NEEDS_REVIEW when the editor still fails after its last revision.
    review_flag: Optional[Literal["NEEDS_REVIEW"]]
    feedback: str
    last_evaluation: Optional[Literal["PASS", "FAIL"]]
    blogger_url: Optional[str]
    revision_count: int
    sender: Optional[str]


def _text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def _published(value) -> Optional[str]:
    """ISO date text of any precision (2024, 2024-03, 2024-03-15), else None."""
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}(-\d{2}(-\d{2})?)?", value):
        return None
    formats = {4: "%Y", 7: "%Y-%m", 10: "%Y-%m-%d"}
    try:
        datetime.strptime(value, formats[len(value)])
    except ValueError:
        return None
    return value


def _confidence(value) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if 0 <= value <= 1 else None


def make_claims(sub_question_id: str, raw) -> List[Claim]:
    """Turns a branch's raw model output into at most MAX_CLAIMS_PER_BRANCH clean claims. Never raises."""
    claims: List[Claim] = []
    for item in raw if isinstance(raw, list) else []:
        if len(claims) == MAX_CLAIMS_PER_BRANCH:
            break
        if not isinstance(item, dict):
            continue
        text, quote, url = _text(item.get("text")), _text(item.get("quote")), _text(item.get("source_url"))
        if not (text and quote and url):
            continue
        claims.append({
            "id": f"{sub_question_id}-{len(claims) + 1}",
            "sub_question_id": sub_question_id,
            "text": text,
            "quote": quote[:MAX_QUOTE_CHARS],
            "source_url": url,
            "source_title": _text(item.get("source_title")) or (urlparse(url).hostname or ""),
            "published": _published(item.get("published")),
            "confidence": _confidence(item.get("confidence")),
        })
    return claims


def make_timeframe(raw) -> Optional[Timeframe]:
    """Bad years become None; a start after the end clears both and keeps the label."""
    if not isinstance(raw, dict):
        return None

    def year(value) -> Optional[int]:
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    start, end = year(raw.get("start_year")), year(raw.get("end_year"))
    if start is not None and end is not None and start > end:
        start = end = None
    return {"start_year": start, "end_year": end, "label": _text(raw.get("label"))}


def _one_line(value: str) -> str:
    """Collapses whitespace, so untrusted text cannot add lines to a prompt."""
    return " ".join(value.split())


def render_notes(
    sub_questions: List[SubQuestion],
    claims: List[Claim],
    verdicts: dict[str, Verdict],
    contradictions: List[Contradiction],
) -> List[str]:
    """Supported claims as the bullet notes the writer already reads."""
    position = {q["id"]: i for i, q in enumerate(sub_questions)}
    supported = [c for c in claims if verdicts.get(c["id"], {}).get("status") == "supported"]
    supported.sort(key=lambda c: (position.get(c["sub_question_id"], len(position)), _claim_order(c["id"])[1]))
    lines = [f"- {_one_line(c['text'])} (Source: {_one_line(c['source_title'])})" for c in supported]
    lines += [f"Sources disagree: {_one_line(c['note'])}" for c in contradictions]
    return lines


def initial_state(topic: str) -> AgentState:
    """Full starting state. sub_questions is always present, which marks a current run."""
    return {
        "topic": topic,
        "research_notes": [],
        "run_status": None,
        "sanitizer_removed": [],
        "unsupported_figures": [],
        "timeframe": None,
        "sub_questions": [],
        "claims": [],
        "branch_errors": [],
        "claim_verdicts": {},
        "contradictions": [],
        "gaps": [],
        "audit_status": None,
        "audit_feedback": None,
        "audit_loops": 0,
        "draft": "",
        "title": "",
        "approved_sha256": None,
        "review_flag": None,
        "feedback": "",
        "last_evaluation": None,
        "blogger_url": None,
        "revision_count": 0,
        "sender": "user",
    }
