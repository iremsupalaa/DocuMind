"""DocuMind ortam ayarları."""

import os
from pathlib import Path

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql:///ollama_library")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
THINGSBOARD_URL = os.environ.get("THINGSBOARD_URL", "").rstrip("/")
THINGSBOARD_AUTH_TIMEOUT_SECONDS = float(
    os.environ.get("THINGSBOARD_AUTH_TIMEOUT_SECONDS", "10")
)
LIBRARY_SCAN_SECONDS = float(os.environ.get("LIBRARY_SCAN_SECONDS", "2"))
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "embeddinggemma")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "768"))
LIBRARY_DIR = Path(os.environ.get(
    "LIBRARY_DIR",
    str(Path.home() / "Desktop" / "Library-Connector"),
)).expanduser()
