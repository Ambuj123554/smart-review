import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

class Settings:
    """Application Settings and Configuration."""
    PROJECT_NAME: str = "Smart Code Review Assistant"
    VERSION: str = "0.1.0"
    
    # Gemini API Key and model identifier
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    
    # GitHub API settings
    GITHUB_WEBHOOK_SECRET: str = os.getenv("GITHUB_WEBHOOK_SECRET", "")
    GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")
    # Guard rails for very large PRs (0 = no cap)
    GITHUB_MAX_FILES: int = int(os.getenv("GITHUB_MAX_FILES", "50"))
    GITHUB_MAX_LINES: int = int(os.getenv("GITHUB_MAX_LINES", "3000"))
    
    # Database configuration
    DATABASE_PATH: str = os.getenv("DATABASE_PATH", "reviews.db")
    
    # CORS: comma-separated allowed origins (use * only for public read APIs;
    # credentials are never enabled alongside *).
    ALLOWED_ORIGINS: str = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000")
    
    # Server configuration
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")

settings = Settings()
