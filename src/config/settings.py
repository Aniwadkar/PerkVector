"""Configuration settings for PerkVector."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# API Configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
USE_MOCK_MODE = os.getenv("USE_MOCK_MODE", "true").lower() == "true" or not ANTHROPIC_API_KEY
HAIKU_MODEL = os.getenv("HAIKU_MODEL", "claude-3-5-haiku-20241022")
SONNET_MODEL = os.getenv("SONNET_MODEL", "claude-sonnet-4-20250514")

# Vertex AI / grounded explanation configuration
AI_EXPLANATIONS_ENABLED = os.getenv("AI_EXPLANATIONS_ENABLED", "false").lower() == "true"
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "cardiq-anish-2026")
GCP_LOCATION = os.getenv("GCP_LOCATION", "global")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Embedding Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Paths
RAW_CARDS_JSON_PATH = PROJECT_ROOT / os.getenv(
    "RAW_CARDS_JSON_PATH", "data/raw/credit_cards_llm_special_features_filled.json"
)
PROCESSED_CARDS_JSON_PATH = PROJECT_ROOT / os.getenv(
    "PROCESSED_CARDS_JSON_PATH", "data/processed/cards_processed.json"
)
CARDS_JSON_PATH = PROJECT_ROOT / os.getenv(
    "CARDS_JSON_PATH", str(PROCESSED_CARDS_JSON_PATH.relative_to(PROJECT_ROOT))
)
CARD_QUALITY_REPORT_PATH = PROJECT_ROOT / os.getenv(
    "CARD_QUALITY_REPORT_PATH", "data/quality/card_quality_report.json"
)
VECTOR_DB_PATH = PROJECT_ROOT / os.getenv("VECTOR_DB_PATH", "data/vector_db/")

# RAG Configuration
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))

# Spending Categories
SPENDING_CATEGORIES = [
    "dining",
    "groceries", 
    "travel",
    "flights",
    "hotels",
    "gas",
    "streaming",
    "transit",
    "other"
]

# Point Valuation (cents per point)
POINT_VALUES = {
    "cash_back": 1.0,
    "flexible_points": 1.25,
    "travel": 1.25,
    "business": 1.0,
    "hotel_points": 1.25,
    "dining_rewards": 1.25
}

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
