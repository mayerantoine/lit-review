import os
import logging
from pathlib import Path
from enum import Enum
from dotenv import load_dotenv

# Load .env file from project root (parent directory of api/)
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Create logger
logger = logging.getLogger(__name__)


class VectorStorageMode(str, Enum):
    """Vector storage mode configuration"""
    PERSISTENT = "persistent"
    IN_MEMORY = "in_memory"


class Config:
    """Application configuration from environment variables"""

    # Vector Storage Configuration
    VECTOR_STORAGE_MODE = VectorStorageMode(
        os.getenv("VECTOR_STORAGE_MODE", "persistent")
    )
    VECTOR_PERSIST_DIRECTORY = os.getenv(
        "VECTOR_PERSIST_DIRECTORY",
        "./corpus-data/chroma_db"
    )

    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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


# Export logger for use in other modules
def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module"""
    return logging.getLogger(name)
