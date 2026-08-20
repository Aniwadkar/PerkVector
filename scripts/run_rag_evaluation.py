"""Run the PerkVector RAG retrieval benchmark."""
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.rag_evaluation import evaluate_retrieval, write_retrieval_evaluation


if __name__ == "__main__":
    report = evaluate_retrieval()
    path = write_retrieval_evaluation()
    print(
        f"RAG retrieval: {report['hits']}/{report['query_count']} Hit@{report['k']} "
        f"({report['hit_rate']:.0%}), source coverage {report['source_coverage']:.0%}."
    )
    print(f"Report: {path}")
