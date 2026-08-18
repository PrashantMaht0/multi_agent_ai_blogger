"""
src/state.py
Updated state schema including validation metrics.
"""

from typing import List, Optional, Literal, Dict, Any
from typing_extensions import TypedDict


class AgentState(TypedDict):
    topic: str
    # Replaced (not appended) on every research pass so failed/stale notes cannot linger
    research_notes: List[str]
    research_error: Optional[str]
    research_attempts: int
    # New validator tracking fields
    raw_sources: List[Dict[str, Any]]
    validation_status: Optional[Literal["VALIDATED", "REJECTED"]]
    validation_feedback: Optional[str]
    run_status: Optional[Literal["FAILED"]]
    # What the deterministic sanitizer stripped from the draft before publishing
    sanitizer_removed: List[str]

    draft: str
    feedback: str
    last_evaluation: Optional[Literal["PASS", "FAIL"]]
    blogger_url: Optional[str]
    revision_count: int
    sender: Optional[str]