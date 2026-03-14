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

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
