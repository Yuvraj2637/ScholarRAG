"""Unit tests for IR retrieval metrics."""

import math
from src.evaluation.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    hit_rate_at_k,
    reciprocal_rank,
    dcg_at_k,
    ndcg_at_k,
    deduplicate_preserving_order,
)

def test_retrieval_metrics():
    targets = ["paper_a", "paper_b"]
    retrieved = ["paper_c", "paper_a", "paper_d", "paper_b", "paper_e"]

    # Precision@K
    assert precision_at_k(retrieved, targets, k=1) == 0.0
    assert precision_at_k(retrieved, targets, k=2) == 0.5
    assert precision_at_k(retrieved, targets, k=4) == 0.5
    assert precision_at_k(retrieved, targets, k=5) == 0.4

    # Recall@K
    assert recall_at_k(retrieved, targets, k=1) == 0.0
    assert recall_at_k(retrieved, targets, k=2) == 0.5
    assert recall_at_k(retrieved, targets, k=4) == 1.0

    # Hit Rate / Success@K
    assert hit_rate_at_k(retrieved, targets, k=1) == 0.0
    assert hit_rate_at_k(retrieved, targets, k=2) == 1.0

    # Reciprocal Rank
    assert reciprocal_rank(retrieved, targets) == 0.5

    # nDCG@K
    ndcg_2 = ndcg_at_k(retrieved, targets, k=2)
    assert 0.0 <= ndcg_2 <= 1.0

    # Empty cases
    assert precision_at_k([], targets, k=5) == 0.0
    assert reciprocal_rank(["paper_x", "paper_y"], targets) == 0.0

def test_duplicate_chunks_ndcg_bound():
    """Verify that multiple chunks from the same relevant paper never cause nDCG > 1.0."""
    targets = ["paper_a"]
    # All 5 retrieved chunks belong to the same target paper
    retrieved_with_duplicates = ["paper_a", "paper_a", "paper_a", "paper_a", "paper_a"]

    p1 = precision_at_k(retrieved_with_duplicates, targets, k=1)
    p5 = precision_at_k(retrieved_with_duplicates, targets, k=5)
    r1 = recall_at_k(retrieved_with_duplicates, targets, k=1)
    r5 = recall_at_k(retrieved_with_duplicates, targets, k=5)
    rr = reciprocal_rank(retrieved_with_duplicates, targets)
    ndcg5 = ndcg_at_k(retrieved_with_duplicates, targets, k=5)

    assert p1 == 1.0
    assert p5 == 0.2  # 1 unique target item in top 5
    assert r1 == 1.0
    assert r5 == 1.0
    assert rr == 1.0
    # CRITICAL: nDCG must be exactly 1.0, NEVER > 1.0
    assert ndcg5 == 1.0
    assert 0.0 <= ndcg5 <= 1.0

def test_deduplicate_preserving_order():
    items = ["b", "a", "b", "c", "a", "d"]
    assert deduplicate_preserving_order(items) == ["b", "a", "c", "d"]
