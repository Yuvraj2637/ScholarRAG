from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Provider & Keys
    LLM_PROVIDER: Literal["gemini", "openai", "mock"] = "gemini"
    GEMINI_API_KEY: str = ""
    OPENAI_API_KEY: str = ""

    # Models
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-base-en-v1.5"
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    GENERATION_MODEL: str = "gemini-2.5-flash"
    FAITHFULNESS_JUDGE_MODEL: str = "gemini-2.5-flash"

    # Chunking & Ingestion Defaults
    CHUNK_MAX_TOKENS: int = 500
    CHUNK_MIN_TOKENS: int = 40
    CHUNK_OVERLAP_TOKENS: int = 50

    # Retrieval Defaults
    DENSE_TOP_K: int = 20
    BM25_TOP_K: int = 20
    RRF_K: int = 60
    RERANK_TOP_K: int = 5

    # File System Directories
    DATA_DIR: Path = PROJECT_ROOT / "data"
    PAPERS_DIR: Path = PROJECT_ROOT / "data" / "papers"
    PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
    EVAL_DIR: Path = PROJECT_ROOT / "data" / "eval"
    RESULTS_DIR: Path = PROJECT_ROOT / "results"
    CHROMA_DIR: Path = PROJECT_ROOT / "data" / "processed" / "chroma_db"

    # Processed File Artifacts
    CHUNKS_FILE: Path = PROJECT_ROOT / "data" / "processed" / "chunks.json"
    GROUND_TRUTH_FILE: Path = PROJECT_ROOT / "data" / "eval" / "ground_truth.json"

    def ensure_directories(self) -> None:
        """Create project directory hierarchy if missing."""
        for path in [
            self.DATA_DIR,
            self.PAPERS_DIR,
            self.PROCESSED_DIR,
            self.EVAL_DIR,
            self.RESULTS_DIR,
            self.CHROMA_DIR,
        ]:
            path.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.ensure_directories()
