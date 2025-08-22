from dataclasses import dataclass
from dotenv import load_dotenv
import os

load_dotenv()

@dataclass
class Settings:
    chat_model: str = os.getenv("CHAT_MODEL", "llama3")
    embed_model: str = os.getenv("EMBED_MODEL", "nomic-embed-text")
    vector_db_path: str = os.getenv("VECTOR_DB_PATH", ".chroma")
    docs_path: str = os.getenv("DOCS_PATH", "./data")
    sqlite_path: str = os.getenv("SQLITE_PATH", "./.memory.db")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    top_k: int = int(os.getenv("TOP_K", "4"))

settings = Settings()
