"""需求羅盤後端 FastAPI 應用工廠。"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import routes
from app.config import settings
from app.database import init_db


def _parse_cors_origins(raw_origins: str) -> list[str]:
    if not raw_origins:
        return []
    return [item.strip() for item in raw_origins.split(",") if item.strip()]


def create_app() -> FastAPI:
    app = FastAPI(title="需求羅盤 API", version="0.1.0")

    allowed_origins = _parse_cors_origins(settings.cors_allow_origins)
    origin_regex = settings.cors_allow_origin_regex or None

    # 允許本機開發與 Vercel 前端，避免 Render preflight 被擋。
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins or ["*"],
        allow_origin_regex=origin_regex,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 初始化資料庫。
    init_db()

    # 掛載 API 路由。
    app.include_router(routes.router, prefix="/api")

    return app
