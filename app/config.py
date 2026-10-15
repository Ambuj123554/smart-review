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
    
    # LLM provider selection: "gemini" (default) or "groq"
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    
    # GitHub API settings
    GITHUB_WEBHOOK_SECRET: str = os.getenv("GITHUB_WEBHOOK_SECRET", "")
    GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")
    # Guard rails for very large PRs (0 = no cap)
    MAX_FILES_PER_PR: int = int(os.getenv("MAX_FILES_PER_PR", "20"))
    MAX_LINES_PER_FILE: int = int(os.getenv("MAX_LINES_PER_FILE", "800"))
    GITHUB_MAX_LINES: int = int(os.getenv("GITHUB_MAX_LINES", "3000"))

    # Lowest severity posted as an inline PR comment (LOW|MEDIUM|HIGH|
    # CRITICAL). Lower findings are still saved to the DB and counted in
    # the PR summary. Validated at startup; invalid values fall back to LOW.
    MIN_SEVERITY: str = os.getenv("MIN_SEVERITY", "LOW")
    
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
