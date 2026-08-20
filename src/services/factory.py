"""Build the production recommendation service from environment settings."""
from __future__ import annotations

import logging

from src.ai import GeminiRAGExplainer
from src.config.settings import AI_EXPLANATIONS_ENABLED
from src.rag.evidence_retriever import CardEvidenceRetriever
from src.repositories import CardRepository
from src.services.rag_recommendation_service import RAGRecommendationService
from src.services.recommendation_service import RecommendationService


LOGGER = logging.getLogger(__name__)


def create_recommendation_service():
    repository = CardRepository()
    base_service = RecommendationService(card_repository=repository)
    if not AI_EXPLANATIONS_ENABLED:
        return base_service
    try:
        return RAGRecommendationService(
            base_service=base_service,
            retriever=CardEvidenceRetriever(repository.list_cards()),
            explainer=GeminiRAGExplainer(),
        )
    except Exception as exc:
        LOGGER.warning("AI initialization failed; deterministic service enabled: %s", exc)
        return base_service
