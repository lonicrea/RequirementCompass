"""每日 token 用量限制工具。"""

from __future__ import annotations

from datetime import datetime
from functools import wraps
from typing import Callable, Any

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.config import settings
from app.models import TokenUsage


def _get_today() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")


def add_token_usage(db: Session, tokens: int) -> None:
    today = _get_today()
    usage = db.query(TokenUsage).filter(TokenUsage.date == today).first()
    if usage:
        usage.total_tokens += tokens
    else:
        usage = TokenUsage(date=today, total_tokens=tokens)
        db.add(usage)
    db.commit()


def get_today_usage(db: Session) -> int:
    today = _get_today()
    usage = db.query(TokenUsage).filter(TokenUsage.date == today).first()
    return usage.total_tokens if usage else 0


def check_token_limit(func: Callable[..., Any]):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        db: Session = kwargs.get("db")
        if db is None:
            raise RuntimeError("需要提供資料庫 Session")
        if settings.daily_token_limit and settings.daily_token_limit > 0:
            if get_today_usage(db) >= settings.daily_token_limit:
                raise HTTPException(status_code=429, detail="token_limit_reached")
        return await func(*args, **kwargs)

    return wrapper
