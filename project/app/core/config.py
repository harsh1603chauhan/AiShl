from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "shl-assessment-recommender")
    app_env: str = os.getenv("APP_ENV", "development")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    shl_catalog_url: str = os.getenv("SHL_CATALOG_URL", "https://www.shl.com/solutions/products/product-catalog/")
    raw_catalog_path: Path = BASE_DIR / os.getenv("RAW_CATALOG_PATH", "data/raw/catalog.html")
    processed_catalog_path: Path = BASE_DIR / os.getenv("PROCESSED_CATALOG_PATH", "data/processed/catalog.json")
    faiss_index_path: Path = BASE_DIR / os.getenv("FAISS_INDEX_PATH", "vectorstore/shl_catalog.faiss")
    faiss_meta_path: Path = BASE_DIR / os.getenv("FAISS_META_PATH", "vectorstore/shl_catalog_meta.json")
    embeddings_provider: str = os.getenv("EMBEDDINGS_PROVIDER", "sentence_transformers")
    embeddings_model: str = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    openai_embeddings_model: str = os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small")
    llm_provider: str = os.getenv("LLM_PROVIDER", "none")
    llm_model: str = os.getenv("LLM_MODEL", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    llm_timeout_seconds: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "20"))
    http_timeout_seconds: int = int(os.getenv("HTTP_TIMEOUT_SECONDS", "30"))
    user_agent: str = os.getenv("USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SHL-Recommender/1.0")
    top_k_retrieval: int = int(os.getenv("TOP_K_RETRIEVAL", "20"))
    top_k_recommendations: int = int(os.getenv("TOP_K_RECOMMENDATIONS", "10"))
    min_recommendations: int = int(os.getenv("MIN_RECOMMENDATIONS", "1"))
    max_turns: int = int(os.getenv("MAX_TURNS", "8"))


settings = Settings()
