"""BM25 Sparse Keyword Retriever implementation."""

import re
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi, BM25Plus
from src.config import settings
from src.ingestion.chunker import Chunk

class BM25Retriever:
    def __init__(self):
        self.bm25: BM25Plus | None = None
        self.chunks: List[Chunk] = []
        self.chunk_map: Dict[str, Chunk] = {}

    def tokenize(self, text: str) -> List[str]:
        """Simple, fast alphanumeric tokenizer for sparse retrieval."""
        tokens = re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())
        return [t for t in tokens if len(t) > 1]

    def index_chunks(self, chunks: List[Chunk]) -> int:
        """Build BM25Plus index over all chunk texts."""
        self.chunks = chunks
        self.chunk_map = {c.chunk_id: c for c in chunks}
        corpus_tokens = [self.tokenize(c.text) for c in chunks]
        self.bm25 = BM25Plus(corpus_tokens)
        print(f"[INFO] Indexed {len(chunks)} chunks in BM25.")
        return len(chunks)

    def search(self, query: str, top_k: int = settings.BM25_TOP_K) -> List[Dict[str, Any]]:
        """Query BM25 index and return top-K scored chunks."""
        if not self.bm25 or not self.chunks:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        # Top-K indices sorted descending
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        hits = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            chunk = self.chunks[idx]
            hits.append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "score": score,
                "paper_id": chunk.paper_id,
                "paper_title": chunk.paper_title,
                "section_title": chunk.section_title,
                "year": chunk.year,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "primary_topic": chunk.primary_topic,
                "retriever": "bm25"
            })

        return hits
