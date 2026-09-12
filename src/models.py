"""ORM model for escalated decisions awaiting human review."""
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.db import Base


class ReviewItem(Base):
    __tablename__ = "review_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    draft: Mapped[str] = mapped_column(Text, nullable=False)
    decision: Mapped[str] = mapped_column(Text, nullable=False)
    critique: Mapped[str] = mapped_column(Text, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    approved_by_critic: Mapped[bool] = mapped_column(Boolean, nullable=False)
    escalation_reason: Mapped[str] = mapped_column(Text, nullable=False)

    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    reviewer_notes: Mapped[str] = mapped_column(Text, nullable=True, default="")

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    reviewed_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)