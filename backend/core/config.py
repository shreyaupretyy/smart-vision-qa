from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings"""
    
    # App
    app_name: str = "SmartVisionQA"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    
    # CORS
    cors_origins: str = "http://localhost:5173,http://localhost:3000"
    
    # File Storage
    upload_dir: str = "./uploads"
    temp_dir: str = "./temp"
    max_upload_size: int = 524288000  # 500MB
    
    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    
    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"
    
    # Models
    yolo_model: str = "yolov8n.pt"
    blip_model: str = "Salesforce/blip-image-captioning-base"
    whisper_model: str = "base"
    
    # Processing
    frame_sample_rate: int = 1
    max_concurrent_jobs: int = 2
    
    # Security
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
