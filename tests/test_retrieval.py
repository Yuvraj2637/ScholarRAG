"""Unit tests for retrieval modules."""

from src.retrieval.hybrid import reciprocal_rank_fusion
from src.retrieval.bm25 import BM25Retriever
from src.ingestion.chunker import Chunk

def test_reciprocal_rank_fusion():
    list1 = [
        {"chunk_id": "c1", "text": "sample 1", "retriever": "dense"},
        {"chunk_id": "c2", "text": "sample 2", "retriever": "dense"},
    ]
    list2 = [
        {"chunk_id": "c2", "text": "sample 2", "retriever": "bm25"},
        {"chunk_id": "c3", "text": "sample 3", "retriever": "bm25"},
    ]

    fused = reciprocal_rank_fusion([list1, list2], k=60, top_n=3)
    # c2 is in top of both (rank 2 in list1, rank 1 in list2)
    # score(c2) = 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.016129 + 0.016393 = 0.03252
    # score(c1) = 1/(60+1) = 0.016393
    # score(c3) = 1/(60+2) = 0.016129
    assert fused[0]["chunk_id"] == "c2"
    assert fused[1]["chunk_id"] == "c1"
    assert fused[2]["chunk_id"] == "c3"

def test_bm25_retriever():
    c1 = Chunk(
        chunk_id="c1",
        paper_id="p1",
        paper_title="LoRA Paper",
        authors=["Author"],
        year=2021,
        section_title="Method",
        section_index=1,
        chunk_index_in_section=0,
        text="Low-rank adaptation decomposes weight updates into matrices A and B.",
        token_count=10
    )
    c2 = Chunk(
        chunk_id="c2",
        paper_id="p2",
        paper_title="FlashAttention Paper",
        authors=["Author"],
        year=2022,
        section_title="Method",
        section_index=1,
        chunk_index_in_section=0,
        text="FlashAttention uses IO-aware tiling in GPU SRAM blocks.",
        token_count=8
    )

    retriever = BM25Retriever()
    retriever.index_chunks([c1, c2])

    hits = retriever.search("low-rank adaptation", top_k=2)
    assert len(hits) >= 1
    assert hits[0]["chunk_id"] == "c1"
