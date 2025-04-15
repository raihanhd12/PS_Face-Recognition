# app/core/config.py
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Face Recognition API"
    DESCRIPTION: str = "API for face recognition and comparison"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # Security
    API_KEY: str
    API_KEY_NAME: str

    # Server settings
    API_HOST: str
    API_PORT: int

    class Config:
        env_file = ".env"


settings = Settings()
