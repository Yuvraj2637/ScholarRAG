"""Reciprocal Rank Fusion (RRF) hybrid retrieval algorithm."""

from typing import List, Dict, Any
from src.config import settings

def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    k: int = settings.RRF_K,
    top_n: int = 20
) -> List[Dict[str, Any]]:
    """
    Fuse multiple ranked result lists using Reciprocal Rank Fusion (RRF).
    
    Formula:
        RRF_score(d) = sum_{m in M} 1 / (k + rank_m(d))
    where rank_m(d) is 1-indexed.
    """
    rrf_scores: Dict[str, float] = {}
    doc_registry: Dict[str, Dict[str, Any]] = {}
    rank_details: Dict[str, Dict[str, int]] = {}

    for retriever_results in ranked_lists:
        for rank_idx, doc in enumerate(retriever_results, start=1):
            doc_id = doc["chunk_id"]
            if doc_id not in doc_registry:
                doc_registry[doc_id] = doc
                rank_details[doc_id] = {}

            retriever_name = doc.get("retriever", "unknown")
            rank_details[doc_id][retriever_name] = rank_idx
            
            # Reciprocal rank contribution
            contribution = 1.0 / (k + rank_idx)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + contribution

    # Sort items by fused RRF score descending
    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda d_id: rrf_scores[d_id], reverse=True)[:top_n]

    fused_results = []
    for d_id in sorted_doc_ids:
        item = dict(doc_registry[d_id])
        item["score"] = float(rrf_scores[d_id])
        item["rrf_score"] = float(rrf_scores[d_id])
        item["individual_ranks"] = rank_details[d_id]
        item["retriever"] = "hybrid_rrf"
        fused_results.append(item)

    return fused_results
