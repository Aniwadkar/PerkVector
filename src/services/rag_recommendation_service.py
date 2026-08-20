"""RAG enrichment around the deterministic recommendation engine."""
from __future__ import annotations

import logging
from typing import Dict, List

from src.ai.gemini_explainer import GeminiRAGExplainer
from src.models.agent_outputs import EvidenceCitation, RecommendationOutput
from src.models.user_input import UserProfile
from src.rag.evidence_retriever import CardEvidenceRetriever, EvidenceChunk
from src.services.recommendation_service import RecommendationService


LOGGER = logging.getLogger(__name__)


class RAGRecommendationService:
    """Preserve deterministic ranking while enriching explanations with Gemini."""

    def __init__(
        self,
        base_service: RecommendationService,
        retriever: CardEvidenceRetriever,
        explainer: GeminiRAGExplainer,
    ):
        self.base_service = base_service
        self.retriever = retriever
        self.explainer = explainer

    def recommend(self, user_profile: UserProfile, limit: int = 3) -> RecommendationOutput:
        output = self.base_service.recommend(user_profile, limit=limit)
        if not output.recommendations:
            return output.model_copy(update={"ai_status": "not_applicable"})

        evidence_by_card = self._retrieve_evidence(user_profile, output)
        try:
            generated = self.explainer.explain(user_profile, output, evidence_by_card)
            generated_by_card = {item.card_id: item for item in generated.explanations}

            for recommendation in output.recommendations:
                explanation = generated_by_card.get(recommendation.card_id)
                available = {
                    chunk.evidence_id: chunk
                    for chunk in evidence_by_card[recommendation.card_id]
                }
                if explanation is None:
                    continue
                cited = [available[evidence_id] for evidence_id in explanation.evidence_ids if evidence_id in available]
                if not cited:
                    continue
                recommendation.why_this_card = explanation.explanation
                recommendation.explanation_mode = "rag"
                recommendation.evidence = [self._citation(chunk) for chunk in cited]

            if not all(rec.explanation_mode == "rag" for rec in output.recommendations):
                raise ValueError("Gemini did not return grounded explanations for every recommendation")
            output.portfolio_strategy = generated.portfolio_strategy
            output.ai_status = "generated"
        except Exception as exc:
            LOGGER.warning("RAG explanation failed; using deterministic fallback: %s", exc)
            output.ai_status = "fallback"
            for recommendation in output.recommendations:
                recommendation.explanation_mode = "deterministic_fallback"
                recommendation.evidence = []
        return output

    def _retrieve_evidence(
        self,
        user_profile: UserProfile,
        output: RecommendationOutput,
    ) -> Dict[str, List[EvidenceChunk]]:
        spending = user_profile.monthly_spending.model_dump()
        categories = " ".join(
            category for category, amount in spending.items() if amount and amount > 0
        )
        evidence = {}
        for recommendation in output.recommendations:
            query = (
                f"{recommendation.card_id.replace('_', ' ')} {categories} rewards annual fee signup bonus "
                "credits benefits limitations foreign transaction fee"
            )
            evidence[recommendation.card_id] = self.retriever.retrieve(
                query=query,
                k=4,
                allowed_card_ids=[recommendation.card_id],
            )
        return evidence

    @staticmethod
    def _citation(chunk: EvidenceChunk) -> EvidenceCitation:
        return EvidenceCitation(
            evidence_id=chunk.evidence_id,
            title=chunk.title,
            source_url=chunk.source_url,
            source_last_checked=chunk.source_last_checked,
        )
