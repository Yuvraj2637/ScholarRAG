import sys
from pathlib import Path
import json
from typing import List, Dict, Any, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import settings
from src.ingestion.paper_catalog import CURATED_PAPERS, PAPER_CATALOG_MAP
from src.retrieval.pipeline import RetrievalPipeline, RetrievalMode
from src.generation.generator import ResearchGenerator, GenerationResult
from src.generation.faithfulness import FaithfulnessJudge, FaithfulnessReport
from src.generation.comparator import PaperComparator, PaperComparisonReport

app = FastAPI(
    title="Research Paper RAG Engine API",
    version="1.0.0",
    description="Production RAG backend for Efficient Transformers and LLM Optimization research corpus."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy singleton instances
_pipeline: Optional[RetrievalPipeline] = None
_generator: Optional[ResearchGenerator] = None
_judge: Optional[FaithfulnessJudge] = None
_comparator: Optional[PaperComparator] = None

def get_pipeline() -> RetrievalPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RetrievalPipeline()
    return _pipeline

def get_generator() -> ResearchGenerator:
    global _generator
    if _generator is None:
        _generator = ResearchGenerator()
    return _generator

def get_judge() -> FaithfulnessJudge:
    global _judge
    if _judge is None:
        _judge = FaithfulnessJudge()
    return _judge

def get_comparator() -> PaperComparator:
    global _comparator
    if _comparator is None:
        _comparator = PaperComparator(get_pipeline())
    return _comparator

# --- Request / Response Models ---
class QueryRequest(BaseModel):
    query: str
    retrieval_mode: RetrievalMode = "hybrid_rerank"
    top_k: int = Field(default=5, ge=1, le=20)
    audit_faithfulness: bool = True

class QueryResponse(BaseModel):
    query: str
    retrieval_mode: str
    answer: str
    citations: List[Dict[str, Any]]
    retrieved_chunks: List[Dict[str, Any]]
    faithfulness: Optional[FaithfulnessReport] = None
    model_name: str

class CompareRequest(BaseModel):
    paper_ids: List[str]

# --- Endpoints ---
@app.get("/health")
def health_check():
    return {"status": "ok", "version": "1.0.0", "corpus_size": len(CURATED_PAPERS)}

@app.get("/api/corpus")
def get_corpus():
    """Return all catalog papers and metadata."""
    return [p.model_dump() for p in CURATED_PAPERS]

@app.post("/api/query", response_model=QueryResponse)
def query_rag(req: QueryRequest):
    """Execute end-to-end RAG query: retrieve -> rerank -> generate -> audit."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    pipeline = get_pipeline()
    generator = get_generator()
    judge = get_judge()

    # Step 1: Retrieve
    chunks = pipeline.retrieve(req.query, mode=req.retrieval_mode, top_k=req.top_k)

    # Step 2: Generate citation-grounded response
    gen_result = generator.generate(req.query, chunks)

    # Step 3: Optional faithfulness verification
    faithfulness_report = None
    if req.audit_faithfulness and chunks:
        faithfulness_report = judge.evaluate(gen_result.answer, chunks)

    return QueryResponse(
        query=req.query,
        retrieval_mode=req.retrieval_mode,
        answer=gen_result.answer,
        citations=[c.model_dump() for c in gen_result.citations],
        retrieved_chunks=chunks,
        faithfulness=faithfulness_report,
        model_name=gen_result.model_name
    )

@app.post("/api/compare", response_model=PaperComparisonReport)
def compare_papers_endpoint(req: CompareRequest):
    """Perform multi-paper comparative synthesis."""
    if len(req.paper_ids) < 2:
        raise HTTPException(status_code=400, detail="Select at least 2 papers for comparison.")
    comparator = get_comparator()
    return comparator.compare_papers(req.paper_ids)

@app.get("/api/benchmark")
def get_benchmark_results():
    """Return precomputed benchmark results."""
    json_path = settings.RESULTS_DIR / "retrieval_benchmark.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="Benchmark results not yet generated.")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)
