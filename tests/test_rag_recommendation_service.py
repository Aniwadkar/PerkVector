from src.ai.gemini_explainer import CardExplanation, GroundedExplanationResponse
from src.models.user_input import MonthlySpending, UserProfile
from src.rag.evidence_retriever import CardEvidenceRetriever
from src.repositories import CardRepository
from src.services.rag_recommendation_service import RAGRecommendationService
from src.services.recommendation_service import RecommendationService


def _profile():
    return UserProfile(
        monthly_spending=MonthlySpending(
            dining=600,
            groceries=500,
            travel=200,
            gas=100,
            streaming=50,
            other=300,
            flights=0,
            hotels=0,
            transit=0,
        ),
        credit_score="good",
        max_annual_fee=700,
    )


class _GroundedExplainer:
    def explain(self, user_profile, recommendations, evidence_by_card):
        return GroundedExplanationResponse(
            explanations=[
                CardExplanation(
                    card_id=rec.card_id,
                    explanation=f"Grounded explanation for {rec.card_name} based on retrieved catalog evidence.",
                    evidence_ids=[evidence_by_card[rec.card_id][0].evidence_id],
                )
                for rec in recommendations.recommendations
            ],
            portfolio_strategy="Use the first ranked card for its strongest matched categories and verified value.",
        )


class _FailingExplainer:
    def explain(self, user_profile, recommendations, evidence_by_card):
        raise RuntimeError("Vertex unavailable")


def _rag_service(explainer):
    repository = CardRepository()
    return RAGRecommendationService(
        base_service=RecommendationService(card_repository=repository),
        retriever=CardEvidenceRetriever(repository.list_cards()),
        explainer=explainer,
    )


def test_retriever_returns_source_linked_evidence_for_allowed_card():
    repository = CardRepository()
    retriever = CardEvidenceRetriever(repository.list_cards())

    evidence = retriever.retrieve(
        "American Express Gold grocery dining rewards annual fee",
        k=3,
        allowed_card_ids=["american_express_gold"],
    )

    assert len(evidence) == 3
    assert all(item.card_id == "american_express_gold" for item in evidence)
    assert all(item.source_url.startswith("https://") for item in evidence)
    assert evidence[0].retrieval_score > 0


def test_rag_service_adds_only_validated_evidence_citations():
    output = _rag_service(_GroundedExplainer()).recommend(_profile())

    assert output.ai_status == "generated"
    assert all(rec.explanation_mode == "rag" for rec in output.recommendations)
    assert all(rec.evidence for rec in output.recommendations)
    assert all(
        citation.evidence_id.startswith(rec.card_id)
        for rec in output.recommendations
        for citation in rec.evidence
    )


def test_rag_service_preserves_results_when_gemini_fails():
    output = _rag_service(_FailingExplainer()).recommend(_profile())

    assert output.ai_status == "fallback"
    assert len(output.recommendations) == 3
    assert all(rec.explanation_mode == "deterministic_fallback" for rec in output.recommendations)
    assert all(not rec.evidence for rec in output.recommendations)
