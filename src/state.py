"""Shared state passed between every node in the graph."""

from typing import List, Optional, Literal, Dict, Any
from typing_extensions import TypedDict


class AgentState(TypedDict):
    topic: str
    # Replaced on every research pass, so stale notes cannot linger.
    research_notes: List[str]
    research_error: Optional[str]
    research_attempts: int
    raw_sources: List[Dict[str, Any]]
    validation_status: Optional[Literal["VALIDATED", "REJECTED"]]
    validation_feedback: Optional[str]
    run_status: Optional[Literal["FAILED"]]
    # What the sanitizer stripped from the draft.
    sanitizer_removed: List[str]

    draft: str
    feedback: str
    last_evaluation: Optional[Literal["PASS", "FAIL"]]
    blogger_url: Optional[str]
    revision_count: int
    sender: Optional[str]