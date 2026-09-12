"""FastAPI backend exposing the research/decision graph as an HTTP endpoint,
plus a human review queue for escalated (low-confidence/unapproved) decisions.
"""
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.config import settings
from src.db import get_db, init_db
from src.graph import build_graph
from src.models import ReviewItem
from src.state import ResearchState

app = FastAPI(title="Agentic Research & Decision Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_graph = build_graph()


@app.on_event("startup")
def on_startup() -> None:
    init_db()


class ResearchRequest(BaseModel):
    query: str


class ResearchResponse(BaseModel):
    query: str
    research_notes: list[str]
    draft: str
    revision_count: int
    critique: str
    approved: bool
    score: float
    decision: str
    needs_human_review: bool
    escalation_reason: str
    review_id: Optional[int] = None


class ReviewSummary(BaseModel):
    id: int
    query: str
    score: float
    escalation_reason: str
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


class ReviewDetail(ReviewSummary):
    draft: str
    decision: str
    critique: str
    approved_by_critic: bool
    reviewer_notes: str
    reviewed_at: Optional[datetime] = None


class ReviewActionRequest(BaseModel):
    notes: str = ""


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/research", response_model=ResearchResponse)
def research(request: ResearchRequest, db: Session = Depends(get_db)) -> ResearchResponse:
    initial_state = ResearchState(query=request.query, max_iterations=settings.max_iterations)
    final_state = _graph.invoke(initial_state)

    review_id = None
    if final_state["needs_human_review"]:
        item = ReviewItem(
            query=final_state["query"],
            draft=final_state["draft"],
            decision=final_state["decision"],
            critique=final_state["critique"],
            score=final_state["score"],
            approved_by_critic=final_state["approved"],
            escalation_reason=final_state["escalation_reason"],
            status="pending",
        )
        db.add(item)
        db.commit()
        db.refresh(item)
        review_id = item.id

    return ResearchResponse(**final_state, review_id=review_id)


@app.get("/reviews", response_model=list[ReviewSummary])
def list_reviews(status: str = "pending", db: Session = Depends(get_db)):
    items = (
        db.query(ReviewItem)
        .filter(ReviewItem.status == status)
        .order_by(ReviewItem.created_at.desc())
        .all()
    )
    return items


@app.get("/reviews/{review_id}", response_model=ReviewDetail)
def get_review(review_id: int, db: Session = Depends(get_db)):
    item = db.query(ReviewItem).filter(ReviewItem.id == review_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Review item not found")
    return item


@app.post("/reviews/{review_id}/approve", response_model=ReviewDetail)
def approve_review(review_id: int, request: ReviewActionRequest, db: Session = Depends(get_db)):
    item = db.query(ReviewItem).filter(ReviewItem.id == review_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Review item not found")
    item.status = "approved"
    item.reviewer_notes = request.notes
    item.reviewed_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(item)
    return item


@app.post("/reviews/{review_id}/reject", response_model=ReviewDetail)
def reject_review(review_id: int, request: ReviewActionRequest, db: Session = Depends(get_db)):
    item = db.query(ReviewItem).filter(ReviewItem.id == review_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Review item not found")
    item.status = "rejected"
    item.reviewer_notes = request.notes
    item.reviewed_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(item)
    return item