"""Mathematical implementation of standard IR evaluation metrics from first principles."""

import math
from typing import List, Sequence, Set

def deduplicate_preserving_order(items: Sequence[str]) -> List[str]:
    """Deduplicate items while preserving their first-occurrence ranking."""
    seen: Set[str] = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def precision_at_k(retrieved_ids: Sequence[str], target_ids: Sequence[str], k: int, deduplicate: bool = True) -> float:
    """
    Precision@K: Proportion of retrieved items in top-K that are relevant.
    P@K = |Unique Relevant Items in Top-K| / K
    """
    if k <= 0 or not retrieved_ids or not target_ids:
        return 0.0
    cutoff = deduplicate_preserving_order(retrieved_ids[:k]) if deduplicate else list(retrieved_ids[:k])
    targets_set: Set[str] = set(target_ids)
    relevant_retrieved = sum(1 for item in cutoff if item in targets_set)
    return relevant_retrieved / k

def recall_at_k(retrieved_ids: Sequence[str], target_ids: Sequence[str], k: int) -> float:
    """
    Recall@K: Proportion of total relevant items retrieved within top-K.
    R@K = |Unique Relevant Items in Top-K ∩ Targets| / |Targets|
    """
    if not target_ids or k <= 0 or not retrieved_ids:
        return 0.0
    cutoff_unique: Set[str] = set(retrieved_ids[:k])
    targets_set: Set[str] = set(target_ids)
    relevant_retrieved = len(cutoff_unique.intersection(targets_set))
    return min(1.0, relevant_retrieved / len(targets_set))

def hit_rate_at_k(retrieved_ids: Sequence[str], target_ids: Sequence[str], k: int) -> float:
    """
    Hit@K (Success@K): 1.0 if at least one target appears in top-K, else 0.0.
    """
    if k <= 0 or not retrieved_ids or not target_ids:
        return 0.0
    cutoff_set: Set[str] = set(retrieved_ids[:k])
    targets_set: Set[str] = set(target_ids)
    return 1.0 if any(item in targets_set for item in cutoff_set) else 0.0

def reciprocal_rank(retrieved_ids: Sequence[str], target_ids: Sequence[str]) -> float:
    """
    Reciprocal Rank (RR): 1 / rank of the first relevant document in the retrieved list.
    Returns 0.0 if no target document is found.
    """
    targets_set: Set[str] = set(target_ids)
    for rank, item in enumerate(retrieved_ids, start=1):
        if item in targets_set:
            return 1.0 / rank
    return 0.0

def dcg_at_k(retrieved_ids: Sequence[str], target_ids: Sequence[str], k: int) -> float:
    """
    Discounted Cumulative Gain (DCG@K) with binary first-hit relevance.
    Each unique target entity can contribute relevance at most once at its highest rank.
    
    Formula:
        DCG@K = sum_{i=1}^K (rel_i / log2(i + 1))
    where rel_i = 1 if item_i is a target and hasn't appeared earlier in retrieved_ids[:i].
    """
    if k <= 0 or not retrieved_ids or not target_ids:
        return 0.0

    targets_set: Set[str] = set(target_ids)
    seen_relevant: Set[str] = set()
    dcg = 0.0

    for idx, item in enumerate(retrieved_ids[:k], start=1):
        if item in targets_set and item not in seen_relevant:
            seen_relevant.add(item)
            dcg += 1.0 / math.log2(idx + 1)

    return dcg

def ndcg_at_k(retrieved_ids: Sequence[str], target_ids: Sequence[str], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain (nDCG@K):
    nDCG@K = DCG@K / IDCG@K
    
    Guaranteed strictly in range [0.0, 1.0].
    """
    if not target_ids or k <= 0 or not retrieved_ids:
        return 0.0

    actual_dcg = dcg_at_k(retrieved_ids, target_ids, k)
    if actual_dcg == 0.0:
        return 0.0

    # Ideal DCG: all distinct target items placed in top rank positions
    num_ideal_targets = min(len(set(target_ids)), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, num_ideal_targets + 1))

    if idcg <= 0.0:
        return 0.0

    return min(1.0, actual_dcg / idcg)
