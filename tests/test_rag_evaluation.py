from src.evaluation.rag_evaluation import evaluate_retrieval


def test_retrieval_benchmark_has_full_hit_and_source_coverage():
    report = evaluate_retrieval(k=3)

    assert report["query_count"] == 6
    assert report["hit_rate"] == 1.0
    assert report["source_coverage"] == 1.0
