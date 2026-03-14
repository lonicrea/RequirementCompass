"""需求羅盤後端 FastAPI 應用工廠。"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import routes
from app.database import init_db


def create_app() -> FastAPI:
    app = FastAPI(title="需求羅盤 API", version="0.1.0")

    # CORS 目前開放全部來源，正式環境請收斂白名單。
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 初始化資料庫。
    init_db()

    # 掛載 API 路由。
    app.include_router(routes.router, prefix="/api")

    return app
