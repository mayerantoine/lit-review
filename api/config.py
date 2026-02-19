import os
from pathlib import Path
from enum import Enum
from dotenv import load_dotenv

# Load .env file from project root (parent directory of api/)
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class VectorStorageMode(str, Enum):
    """Vector storage mode configuration"""
    PERSISTENT = "persistent"
    IN_MEMORY = "in_memory"


class Config:
    """Application configuration from environment variables"""

    # Deployment environment: "local" (default) | "azure"
    # When set to "azure", ChromaDB defaults to /data/chromadb (Azure Files mount)
    DEPLOYMENT_ENV: str = os.getenv("DEPLOYMENT_ENV", "local")

    # Vector Storage Configuration
    VECTOR_STORAGE_MODE = VectorStorageMode(
        os.getenv("VECTOR_STORAGE_MODE", "in_memory")
    )

    # Azure-aware default: /data/chromadb on Azure, ./corpus-data/chroma_db locally
    # Can be overridden explicitly via VECTOR_PERSIST_DIRECTORY env var
    _default_persist_dir: str = (
        "/data/chromadb" if os.getenv("DEPLOYMENT_ENV", "local") == "azure"
        else "./corpus-data/chroma_db"
    )
    VECTOR_PERSIST_DIRECTORY: str = os.getenv(
        "VECTOR_PERSIST_DIRECTORY",
        _default_persist_dir
    )

    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    @classmethod
    def is_azure(cls) -> bool:
        """Check if running in Azure deployment"""
        return cls.DEPLOYMENT_ENV == "azure"

    @classmethod
    def is_persistent_storage(cls) -> bool:
        """Check if using persistent storage mode"""
        return cls.VECTOR_STORAGE_MODE == VectorStorageMode.PERSISTENT

    @classmethod
    def is_in_memory_storage(cls) -> bool:
        """Check if using in-memory storage mode"""
        return cls.VECTOR_STORAGE_MODE == VectorStorageMode.IN_MEMORY


# Global config instance
config = Config()
