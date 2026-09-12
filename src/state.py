"""Pydantic schema for the shared graph state."""
from typing import List

from pydantic import BaseModel, Field


class ResearchState(BaseModel):
    query: str

    research_notes: List[str] = Field(default_factory=list)

    draft: str = ""
    revision_count: int = 0

    critique: str = ""
    approved: bool = False
    score: float = 0.0

    decision: str = ""

    max_iterations: int = 3

    # Human-in-the-loop escalation
    needs_human_review: bool = False
    escalation_reason: str = ""