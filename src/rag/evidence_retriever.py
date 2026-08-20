"""Retrieve compact, source-linked evidence from the processed card catalog."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence

from pydantic import BaseModel


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class EvidenceChunk(BaseModel):
    evidence_id: str
    card_id: str
    title: str
    text: str
    source_url: str
    source_last_checked: str
    retrieval_score: float = 0.0


class CardEvidenceRetriever:
    """In-memory lexical retriever designed for a small, validated catalog."""

    def __init__(self, cards: Sequence[Dict]):
        self.chunks = self._build_chunks(cards)
        self._token_counts = [
            Counter(self._tokenize(f"{chunk.title} {chunk.text}"))
            for chunk in self.chunks
        ]
        self._document_frequency = Counter(
            token for counts in self._token_counts for token in counts
        )

    def retrieve(
        self,
        query: str,
        k: int = 6,
        allowed_card_ids: Iterable[str] | None = None,
    ) -> List[EvidenceChunk]:
        query_tokens = Counter(self._tokenize(query))
        allowed = set(allowed_card_ids) if allowed_card_ids is not None else None
        scored = []

        for chunk, counts in zip(self.chunks, self._token_counts):
            if allowed is not None and chunk.card_id not in allowed:
                continue
            score = self._score(query_tokens, counts)
            if chunk.card_id.replace("_", " ") in query.lower():
                score += 2.0
            scored.append(chunk.model_copy(update={"retrieval_score": round(score, 4)}))

        scored.sort(key=lambda chunk: (-chunk.retrieval_score, chunk.evidence_id))
        return scored[: min(k, len(scored))]

    def _score(self, query: Counter, document: Counter) -> float:
        if not query or not document:
            return 0.0
        document_length = sum(document.values())
        total_documents = len(self.chunks)
        score = 0.0
        for token, query_count in query.items():
            if token not in document:
                continue
            idf = math.log((total_documents + 1) / (self._document_frequency[token] + 1)) + 1
            score += query_count * idf * (document[token] / document_length) * 10
        return score

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return TOKEN_PATTERN.findall(text.lower())

    def _build_chunks(self, cards: Sequence[Dict]) -> List[EvidenceChunk]:
        chunks: List[EvidenceChunk] = []
        for card in cards:
            common = {
                "card_id": card["card_id"],
                "source_url": card["source_url"],
                "source_last_checked": card["source_last_checked"],
            }
            eligibility = card.get("eligibility", {})
            chunks.append(EvidenceChunk(
                evidence_id=f"{card['card_id']}:overview",
                title=f"{card['card_name']} overview",
                text=(
                    f"{card['card_name']} is issued by {card['issuer']}. {card.get('description', '')} "
                    f"Annual fee ${card['annual_fee']}. Rewards type {card['rewards_type']}. "
                    f"Eligibility tier {eligibility.get('credit_tier', 'not listed')} with minimum score "
                    f"{eligibility.get('min_credit_score', 'not listed')}."
                ),
                **common,
            ))
            reward_text = ", ".join(
                f"{category} {rate}x" for category, rate in card.get("rewards", {}).items()
            )
            chunks.append(EvidenceChunk(
                evidence_id=f"{card['card_id']}:rewards",
                title=f"{card['card_name']} reward rates",
                text=f"Reward rates: {reward_text}. Point value ${card.get('point_value', 0):.4f}.",
                **common,
            ))
            bonus = card.get("signup_bonus", {})
            credits = ", ".join(
                f"{credit.get('name')} ${credit.get('amount', 0)}"
                for credit in card.get("annual_credits", [])
            ) or "No annual credits listed"
            chunks.append(EvidenceChunk(
                evidence_id=f"{card['card_id']}:offer",
                title=f"{card['card_name']} offer and credits",
                text=(
                    f"Signup value ${bonus.get('estimated_value', bonus.get('amount', 0))} after "
                    f"${bonus.get('spend_requirement', 0)} spend in {bonus.get('timeframe_months', 0)} months. "
                    f"Credits: {credits}."
                ),
                **common,
            ))
            features = ", ".join(card.get("special_features", [])) or "No special features listed"
            chunks.append(EvidenceChunk(
                evidence_id=f"{card['card_id']}:features",
                title=f"{card['card_name']} features and limitations",
                text=(
                    f"Best for {', '.join(card.get('best_for', [])) or 'general use'}. "
                    f"Foreign transaction fee {card.get('foreign_transaction_fee', 0)}%. Features: {features}."
                ),
                **common,
            ))
        return chunks
