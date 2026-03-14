"""SQLAlchemy 資料庫工具。"""

from __future__ import annotations

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker


DB_PATH = os.getenv("DB_PATH", "requirement_compass.db")
# 若 DB 路徑含資料夾，先確保資料夾存在。
db_dir = os.path.dirname(DB_PATH)
if db_dir:
    os.makedirs(db_dir, exist_ok=True)

DATABASE_URL = f"sqlite:///{DB_PATH}"


class Base(DeclarativeBase):
    pass


engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db():
    from app import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
