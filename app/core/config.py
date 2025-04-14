# app/core/config.py
import secrets

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Face Recognition API"
    DESCRIPTION: str = "API for face recognition and comparison"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # Security
    API_KEY: str = secrets.token_urlsafe(32)
    API_KEY_NAME: str = "X-API-Key"

    class Config:
        env_file = ".env"


settings = Settings()
