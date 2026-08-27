from typing import List, Dict, Any
from src.config import settings

class CrossEncoderReranker:
    def __init__(self, model_name: str = settings.RERANKER_MODEL_NAME):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = settings.RERANK_TOP_K
    ) -> List[Dict[str, Any]]:
        """Rerank candidate chunks using Cross-Encoder pairwise relevance scoring."""
        if not candidates:
            return []

        pairs = [[query, c["text"]] for c in candidates]
        scores = self.model.predict(pairs)

        reranked = []
        for idx, score in enumerate(scores):
            item = dict(candidates[idx])
            item["rerank_score"] = float(score)
            item["score"] = float(score)
            item["retriever"] = "cross_encoder"
            reranked.append(item)

        # Sort descending by cross-encoder score
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]
