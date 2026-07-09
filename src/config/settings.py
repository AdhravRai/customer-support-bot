"""
Application configuration for the Customer Support RAG Agent.

This module centralizes:
- Environment variable loading
- Project paths
- Model configuration
- Retrieval settings

The module follows a fail-fast approach by validating required
environment variables during startup.
"""

from pathlib import Path
import os

from dotenv import load_dotenv

# ============================================================================
# Project Paths
# ============================================================================

# customer-support-rag/
BASE_DIR: Path = Path(__file__).resolve().parents[2]

# data/documents/
DATA_DIR: Path = BASE_DIR / "data" / "documents"

# vectorstore/faiss_index/
VECTORSTORE_DIR: Path = BASE_DIR / "vectorstore" / "faiss_index"

# ============================================================================
# Environment Variables
# ============================================================================

load_dotenv(BASE_DIR / ".env")

GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY is missing.\n"
        "Create a '.env' file in the project root and add:\n\n"
        "GOOGLE_API_KEY=your_api_key_here"
    )

# ============================================================================
# Model Configuration
# ============================================================================

# Gemini model used for response generation
GEMINI_MODEL: str = "gemini-2.5-flash"

# HuggingFace embedding model
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

# LLM generation parameters
TEMPERATURE: float = 0.2

# ============================================================================
# Document Processing
# ============================================================================

# Number of characters per chunk
CHUNK_SIZE: int = 1000

# Overlap between consecutive chunks
CHUNK_OVERLAP: int = 200

# ============================================================================
# Retrieval Configuration
# ============================================================================

# Number of documents to retrieve
TOP_K: int = 4

# Retrieval strategy
SEARCH_TYPE: str = "similarity"