"""Multi-paper comparative analysis engine using RAG-grounded LLM synthesis."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from src.config import settings
from src.ingestion.paper_catalog import PAPER_CATALOG_MAP, PaperMetadata
from src.retrieval.pipeline import RetrievalPipeline
from src.generation.generator import ResearchGenerator, format_context_blocks

class ComparisonDimension(BaseModel):
    dimension_name: str
    paper_analyses: Dict[str, str]

class PaperComparisonReport(BaseModel):
    paper_ids: List[str]
    paper_titles: Dict[str, str]
    summary_matrix: List[ComparisonDimension] = Field(default_factory=list)
    comparative_synthesis: str = ""
    retrieved_evidence: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    model_name: str = ""

COMPARISON_DIMENSIONS = [
    ("Core Motivation & Bottleneck", "What specific bottleneck (memory, IO bandwidth, compute, quantization degradation) is targeted?"),
    ("Algorithmic Innovation", "What is the primary mathematical, kernel, or architectural innovation introduced?"),
    ("Compute & Memory Trade-offs", "What are the computational complexity and memory footprint characteristics?"),
    ("Empirical Highlights", "What are the headline benchmark speedups, memory reductions, or downstream accuracy results?"),
    ("Limitations & Constraints", "What assumptions, hardware prerequisites, or trade-offs are noted?"),
]

COMPARISON_SYSTEM_PROMPT = """You are an expert AI research scientist specializing in transformer efficiency and LLM architecture.
You are comparing 2-3 research papers based strictly on the provided retrieved context passages.

INSTRUCTIONS:
1. Provide a rigorous, technical comparative analysis contrasting the methodologies, trade-offs, and empirical findings of the papers.
2. Include explicit citations for every claim in brackets (e.g. [Paper Title, Section, Page X]).
3. Be specific regarding equations, tensor dimensions, memory complexities, and benchmark datasets.
"""

class PaperComparator:
    def __init__(self, pipeline: Optional[RetrievalPipeline] = None):
        self.pipeline = pipeline or RetrievalPipeline()
        self.generator = ResearchGenerator()

    def compare_papers(self, paper_ids: List[str]) -> PaperComparisonReport:
        """Generate an authentic RAG-grounded comparative synthesis across 2-3 papers."""
        if len(paper_ids) < 2:
            raise ValueError("Comparison requires at least 2 paper IDs.")

        titles = {
            pid: PAPER_CATALOG_MAP[pid].title if pid in PAPER_CATALOG_MAP else pid
            for pid in paper_ids
        }

        # Step 1: Retrieve targeted evidence for each paper across methodology, results, and architecture
        evidence_by_paper: Dict[str, List[Dict[str, Any]]] = {}
        all_retrieved_chunks: List[Dict[str, Any]] = []

        for pid in paper_ids:
            title = titles[pid]
            chunks = self.pipeline.retrieve(
                query=f"{title} algorithm architecture memory complexity experimental results limitations",
                mode="hybrid",
                top_k=6
            )
            # Filter specifically to chunks belonging to this paper
            paper_specific_chunks = [c for c in chunks if c.get("paper_id") == pid]
            if not paper_specific_chunks:
                # Fallback: retrieve by paper_id keyword
                paper_specific_chunks = self.pipeline.retrieve(query=pid, mode="bm25", top_k=4)

            evidence_by_paper[pid] = paper_specific_chunks[:4]
            all_retrieved_chunks.extend(evidence_by_paper[pid])

        # Step 2: Formulate RAG comparison prompt for the LLM
        context_str = format_context_blocks(all_retrieved_chunks)
        papers_str = " vs ".join([f"'{titles[p]}'" for p in paper_ids])

        user_prompt = (
            f"Please conduct a comprehensive architectural and empirical comparison between the following papers:\n"
            f"{papers_str}\n\n"
            f"RETRIEVED CONTEXT EVIDENCE:\n{context_str}\n\n"
            f"Please structure your comparison across:\n"
            f"1. Core Motivation & Problem Formulation\n"
            f"2. Mathematical & Algorithmic Innovations\n"
            f"3. Compute, Memory & IO Trade-offs\n"
            f"4. Key Empirical Speedups / Benchmarks\n"
            f"5. Critical Limitations & Practical Deployment Constraints\n"
        )

        gen_result = self.generator.generate(
            query=f"Compare {papers_str}",
            retrieved_chunks=all_retrieved_chunks,
            custom_prompt=user_prompt,
            custom_system_prompt=COMPARISON_SYSTEM_PROMPT
        )

        # Step 3: Build structured comparison matrix from retrieved evidence
        matrix: List[ComparisonDimension] = []
        for dim_title, query_prompt in COMPARISON_DIMENSIONS:
            dim_analyses: Dict[str, str] = {}
            for pid in paper_ids:
                p_chunks = evidence_by_paper.get(pid, [])
                if p_chunks:
                    top_chunk = p_chunks[0]
                    p_num = f", p. {top_chunk.get('page_start', 1)}" if top_chunk.get('page_start') else ""
                    dim_analyses[pid] = f"{top_chunk.get('text', '')[:240]}... [{titles[pid]}, {top_chunk.get('section_title', 'Method')}{p_num}]"
                else:
                    dim_analyses[pid] = f"See synthesis details for {titles[pid]}."

            matrix.append(ComparisonDimension(
                dimension_name=dim_title,
                paper_analyses=dim_analyses
            ))

        return PaperComparisonReport(
            paper_ids=paper_ids,
            paper_titles=titles,
            summary_matrix=matrix,
            comparative_synthesis=gen_result.answer,
            retrieved_evidence=evidence_by_paper,
            model_name=gen_result.model_name
        )
