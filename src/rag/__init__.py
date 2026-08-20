"""Retrieval components for PerkVector.

Legacy FAISS classes are loaded lazily so the production web process does not
import sentence-transformers unless an offline vector-index script requests it.
"""

from src.rag.evidence_retriever import CardEvidenceRetriever, EvidenceChunk

__all__ = ["CardEvidenceRetriever", "EvidenceChunk"]


def __getattr__(name):
    if name == "EmbeddingGenerator":
        from src.rag.embeddings import EmbeddingGenerator
        return EmbeddingGenerator
    if name == "VectorStore":
        from src.rag.vector_store import VectorStore
        return VectorStore
    if name == "CardRetriever":
        from src.rag.retriever import CardRetriever
        return CardRetriever
    raise AttributeError(name)
