"""
Centralized application settings.

All configuration is loaded from environment variables (via a .env file in
development, or real environment variables / secrets manager in production).
Nothing here should be hardcoded for a specific deployment.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (no-op if it doesn't exist, e.g. in prod
# where real env vars / Streamlit secrets are injected instead).
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


class Settings:
    # ---- Google Gemini ----
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_CHAT_MODEL: str = os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-pro")
    GEMINI_EMBEDDING_MODEL: str = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-2")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.3"))

    # ---- RAG / chunking ----
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    RETRIEVER_TOP_K: int = int(os.getenv("RETRIEVER_TOP_K", "4"))

    # ---- Paths ----
    DATA_DIR: Path = BASE_DIR / os.getenv("DATA_DIR", "data/documents")
    VECTORSTORE_DIR: Path = BASE_DIR / os.getenv("VECTORSTORE_DIR", "vectorstore")

    # ---- Auth ----
    APP_USERNAME: str = os.getenv("APP_USERNAME", "admin")
    APP_PASSWORD_HASH: str = os.getenv("APP_PASSWORD_HASH", "")
    AUTH_SECRET_KEY: str = os.getenv("AUTH_SECRET_KEY", "insecure-dev-key-change-me")
    SESSION_TIMEOUT_MINUTES: int = int(os.getenv("SESSION_TIMEOUT_MINUTES", "60"))

    # ---- App ----
    APP_TITLE: str = os.getenv("APP_TITLE", "Real Estate AI Assistant")

    @classmethod
    def validate(cls) -> list[str]:
        """Return a list of human-readable config problems (empty if fine)."""
        problems = []
        if not cls.GOOGLE_API_KEY:
            problems.append("GOOGLE_API_KEY is not set. Add it to your .env file.")
        if not cls.APP_PASSWORD_HASH:
            problems.append(
                "APP_PASSWORD_HASH is not set. Run `python -m auth.generate_password_hash` "
                "to create one and add it to your .env file."
            )
        return problems


settings = Settings()
