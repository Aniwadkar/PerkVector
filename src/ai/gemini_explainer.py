"""Generate structured, evidence-grounded recommendation explanations."""
from __future__ import annotations

import json
from typing import Dict, List

from pydantic import BaseModel, Field

from src.api.gemini_client import GeminiClient
from src.models.agent_outputs import RecommendationOutput
from src.models.user_input import UserProfile
from src.rag.evidence_retriever import EvidenceChunk


class CardExplanation(BaseModel):
    card_id: str
    explanation: str = Field(min_length=20, max_length=700)
    evidence_ids: List[str] = Field(min_length=1, max_length=4)


class GroundedExplanationResponse(BaseModel):
    explanations: List[CardExplanation]
    portfolio_strategy: str = Field(min_length=20, max_length=700)


class GeminiRAGExplainer:
    """Ask Gemini to explain deterministic results using retrieved evidence only."""

    def __init__(self, client: GeminiClient | None = None):
        self.client = client or GeminiClient()

    def explain(
        self,
        user_profile: UserProfile,
        recommendations: RecommendationOutput,
        evidence_by_card: Dict[str, List[EvidenceChunk]],
    ) -> GroundedExplanationResponse:
        payload = {
            "user_profile": user_profile.model_dump(),
            "ranked_results": [
                {
                    "card_id": rec.card_id,
                    "card_name": rec.card_name,
                    "rank": rec.rank,
                    "financial_summary": rec.financial_summary,
                    "deterministic_explanation": rec.why_this_card,
                }
                for rec in recommendations.recommendations
            ],
            "retrieved_evidence": {
                card_id: [chunk.model_dump() for chunk in chunks]
                for card_id, chunks in evidence_by_card.items()
            },
        }
        system_prompt = (
            "You explain credit-card recommendations. The ranking and financial_summary are trusted, "
            "precalculated inputs and must never be changed. Use only retrieved_evidence for card facts. "
            "Write two concise sentences per card: first connect the user's spending to the calculated value; "
            "second mention a useful benefit or limitation. Cite 1-4 exact evidence_id values supplied for that "
            "card. Do not give application, legal, or financial advice and do not invent rates, credits, or terms. "
            "Return one explanation for every ranked card and a concise portfolio strategy."
        )
        return self.client.generate_structured(
            system_prompt=system_prompt,
            user_message=json.dumps(payload, separators=(",", ":"), default=str),
            response_schema=GroundedExplanationResponse,
            max_tokens=1800,
        )
