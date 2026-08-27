"""Unified retrieval pipeline orchestrating Dense, BM25, Hybrid RRF, and Cross-Encoder Reranking."""

import json
from pathlib import Path
from typing import List, Dict, Any, Literal
from src.config import settings
from src.ingestion.chunker import Chunk
from src.retrieval.dense import DenseRetriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import reciprocal_rank_fusion
from src.retrieval.reranker import CrossEncoderReranker

RetrievalMode = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]

class RetrievalPipeline:
    def __init__(self, chunks_path: Path = settings.CHUNKS_FILE, auto_index: bool = True):
        self.chunks_path = chunks_path
        self.chunks: List[Chunk] = []
        self.dense_retriever: DenseRetriever | None = None
        self.bm25_retriever: BM25Retriever | None = None
        self.reranker: CrossEncoderReranker | None = None

        if auto_index:
            self.initialize()

    def initialize(self, force_reindex_dense: bool = False) -> None:
        """Load chunks and warm up retrievers."""
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found at {self.chunks_path}. Run ingestion first.")

        with open(self.chunks_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.chunks = [Chunk(**item) for item in data]

        print(f"[INFO] Loaded {len(self.chunks)} structured chunks.")

        # Initialize Dense
        self.dense_retriever = DenseRetriever()
        self.dense_retriever.index_chunks(self.chunks, force_reindex=force_reindex_dense)

        # Initialize BM25
        self.bm25_retriever = BM25Retriever()
        self.bm25_retriever.index_chunks(self.chunks)

        # Initialize Reranker
        self.reranker = CrossEncoderReranker()

    def retrieve(
        self,
        query: str,
        mode: RetrievalMode = "hybrid_rerank",
        top_k: int = 5,
        dense_top_k: int = 20,
        bm25_top_k: int = 20,
        rerank_candidate_k: int = 20
    ) -> List[Dict[str, Any]]:
        """Retrieve chunks based on selected mode."""
        if mode == "dense":
            if not self.dense_retriever:
                raise RuntimeError("Dense retriever not initialized.")
            return self.dense_retriever.search(query, top_k=top_k)

        elif mode == "bm25":
            if not self.bm25_retriever:
                raise RuntimeError("BM25 retriever not initialized.")
            return self.bm25_retriever.search(query, top_k=top_k)

        elif mode == "hybrid":
            if not self.dense_retriever or not self.bm25_retriever:
                raise RuntimeError("Retrievers not initialized.")
            dense_res = self.dense_retriever.search(query, top_k=dense_top_k)
            bm25_res = self.bm25_retriever.search(query, top_k=bm25_top_k)
            return reciprocal_rank_fusion([dense_res, bm25_res], top_n=top_k)

        elif mode == "hybrid_rerank":
            if not self.dense_retriever or not self.bm25_retriever or not self.reranker:
                raise RuntimeError("Pipeline components not initialized.")
            dense_res = self.dense_retriever.search(query, top_k=dense_top_k)
            bm25_res = self.bm25_retriever.search(query, top_k=bm25_top_k)
            fused = reciprocal_rank_fusion([dense_res, bm25_res], top_n=rerank_candidate_k)
            return self.reranker.rerank(query, fused, top_k=top_k)

        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")

    def retrieve_all_modes(self, query: str, top_k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """Run all 4 retrieval modes for side-by-side comparison."""
        dense_results = self.retrieve(query, mode="dense", top_k=top_k)
        bm25_results = self.retrieve(query, mode="bm25", top_k=top_k)
        hybrid_results = self.retrieve(query, mode="hybrid", top_k=top_k)
        rerank_results = self.retrieve(query, mode="hybrid_rerank", top_k=top_k)

        return {
            "dense": dense_results,
            "bm25": bm25_results,
            "hybrid": hybrid_results,
            "hybrid_rerank": rerank_results
        }
