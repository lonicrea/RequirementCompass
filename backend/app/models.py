"""SQLAlchemy 資料表模型。"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def utcnow() -> datetime:
    return datetime.utcnow()


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    idea: Mapped[str] = mapped_column(Text, nullable=False)
    questions: Mapped[str] = mapped_column(Text, default="[]")  # JSON 字串
    answers: Mapped[str] = mapped_column(Text, default="[]")  # JSON 字串
    reports: Mapped[str] = mapped_column(Text, default="[]")  # JSON 字串
    final_doc_path: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    rounds: Mapped[List["Round"]] = relationship(
        "Round", back_populates="session", cascade="all, delete-orphan"
    )


class Round(Base):
    __tablename__ = "rounds"
    __table_args__ = (UniqueConstraint("session_id", "round_number"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.id"), nullable=False)
    round_number: Mapped[int] = mapped_column(Integer, nullable=False)
    questions: Mapped[str] = mapped_column(Text, nullable=False)  # JSON 字串
    answers: Mapped[str] = mapped_column(Text, nullable=False)  # JSON 字串
    report: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    session: Mapped[Session] = relationship("Session", back_populates="rounds")


class TokenUsage(Base):
    __tablename__ = "token_usage"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[str] = mapped_column(String, unique=True)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)
