"""
src/state.py
Updated state schema including validation metrics.
"""

from typing import List, Optional, Literal, Dict, Any
import operator
from typing_extensions import TypedDict, Annotated


class AgentState(TypedDict):
    topic: str
    research_notes: Annotated[List[str], operator.add]
    # New validator tracking fields
    raw_sources: List[Dict[str, Any]]
    validation_status: Optional[Literal["VALIDATED", "REJECTED"]]
    validation_feedback: Optional[str]
    
    draft: str
    feedback: str
    last_evaluation: Optional[Literal["PASS", "FAIL"]]
    blogger_url: Optional[str]
    revision_count: int
    sender: Optional[str]