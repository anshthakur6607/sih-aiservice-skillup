"""
Pydantic Settings for MoSPI SkillUp AI Service
Centralized config loaded from .env
"""
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    google_api_key: Optional[str] = None
    ai_service_api_key: str
    supabase_url: Optional[str] = None
    supabase_service_role_key: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_s3_bucket: str = "mospi-course-materials"
    aws_region: str = "ap-south-1"
    sarvam_api_key: Optional[str] = None
    bhashini_api_key: Optional[str] = None
    chroma_persist_directory: str = "./chroma_data"
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

settings = Settings()  # type: ignore
