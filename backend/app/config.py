"""使用環境變數的應用設定。"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    secret_key: str = "dev-secret-key"
    port: int = 5000
    daily_token_limit: int = 0
    qwen_api_key: str | None = None
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_model: str = "qwen-max"
    openai_api_key: str | None = None
    db_path: str = "requirement_compass.db"
    cors_allow_origins: str = "http://localhost:5175,http://127.0.0.1:5175"
    cors_allow_origin_regex: str = r"^https:\/\/.*\.vercel\.app$"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
