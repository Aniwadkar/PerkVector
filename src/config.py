"""Configuration settings for PerkVector."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

# API Configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Model Configuration
HAIKU_MODEL = os.getenv("HAIKU_MODEL", "claude-3-5-haiku-20241022")
SONNET_MODEL = os.getenv("SONNET_MODEL", "claude-sonnet-4-20250514")

# Embedding Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Data paths
CARDS_JSON_PATH = PROJECT_ROOT / os.getenv("CARDS_JSON_PATH", "data/raw/credit_cards_llm_special_features_filled.json")
VECTOR_DB_PATH = PROJECT_ROOT / os.getenv("VECTOR_DB_PATH", "data/vector_db/")

# RAG Configuration
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))

# Point values for different reward types
POINT_VALUES = {
    "cash_back": 1.0,
    "travel": 1.5,
    "flexible_points": 1.2,
}

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
