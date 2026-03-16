"""
Search Evaluation Metrics
==========================
Standard information retrieval metrics: Precision@K, Recall@K,
MRR, and NDCG@K.
"""
import numpy as np


def precision_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return sum(1 for item in top_k if item in relevant) / len(top_k)


def recall_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    return sum(1 for item in top_k if item in relevant) / len(relevant)


def mrr(retrieved: list[int], relevant: set[int]) -> float:
    """Mean Reciprocal Rank."""
    for i, item in enumerate(retrieved):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, item in enumerate(retrieved[:k])
        if item in relevant
    )
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0
