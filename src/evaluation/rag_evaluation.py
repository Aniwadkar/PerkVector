"""Small, reproducible retrieval benchmark for the PerkVector evidence corpus."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from src.config.settings import PROJECT_ROOT
from src.rag.evidence_retriever import CardEvidenceRetriever
from src.repositories import CardRepository


RAG_QUERIES = [
    {
        "query": "American Express Gold dining and grocery reward rates",
        "expected_evidence_id": "american_express_gold:rewards",
    },
    {
        "query": "Capital One Venture X signup offer and annual travel credits",
        "expected_evidence_id": "capital_one_venture_x:offer",
    },
    {
        "query": "Blue Cash Everyday annual fee and credit score eligibility",
        "expected_evidence_id": "american_express_blue_cash_everyday:overview",
    },
    {
        "query": "Chase Sapphire Preferred travel protections and foreign transaction fee",
        "expected_evidence_id": "chase_sapphire_preferred_card:features",
    },
    {
        "query": "Citi DoubleCash cashback rewards rates",
        "expected_evidence_id": "citi_doublecash:rewards",
    },
    {
        "query": "Wells Fargo Active Cash card issuer annual fee",
        "expected_evidence_id": "wells_fargo_active_cash:overview",
    },
]


def evaluate_retrieval(k: int = 3) -> Dict:
    retriever = CardEvidenceRetriever(CardRepository().list_cards())
    results: List[Dict] = []
    for case in RAG_QUERIES:
        retrieved = retriever.retrieve(case["query"], k=k)
        ids = [item.evidence_id for item in retrieved]
        results.append({
            **case,
            "retrieved_evidence_ids": ids,
            "hit": case["expected_evidence_id"] in ids,
            "all_sources_present": all(bool(item.source_url) for item in retrieved),
        })

    hits = sum(result["hit"] for result in results)
    return {
        "query_count": len(results),
        "k": k,
        "hits": hits,
        "hit_rate": round(hits / len(results), 4),
        "source_coverage": round(
            sum(result["all_sources_present"] for result in results) / len(results), 4
        ),
        "results": results,
    }


def write_retrieval_evaluation(
    path: Path = PROJECT_ROOT / "outputs" / "evaluation" / "rag_retrieval_report.json",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evaluate_retrieval(), indent=2) + "\n", encoding="utf-8")
    return path
