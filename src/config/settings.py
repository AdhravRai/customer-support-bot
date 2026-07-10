"""
Application configuration for the Customer Support RAG Agent.

This module centralizes:
- Environment variable loading
- Project paths
- Model configuration
- Retrieval settings
"""

from pathlib import Path
import os

from dotenv import load_dotenv

# ============================================================================
# Project Paths
# ============================================================================

BASE_DIR: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = BASE_DIR / "data" / "documents"

VECTORSTORE_DIR: Path = BASE_DIR / "vectorstore" / "faiss_index"

# ============================================================================
# Environment Variables
# ============================================================================

load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing.\n"
        "Create a '.env' file in the project root and add:\n\n"
        "GROQ_API_KEY=your_api_key_here"
    )

# ============================================================================
# Model Configuration
# ============================================================================

# Groq model
GROQ_MODEL: str = "llama-3.3-70b-versatile"

# Embedding model
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

TEMPERATURE: float = 0.2

# ============================================================================
# Document Processing
# ============================================================================

CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200

# ============================================================================
# Retrieval Configuration
# ============================================================================

TOP_K: int = 4
SEARCH_TYPE: str = "similarity"