# ⚡ ScholarRAG: Research Paper RAG & Retrieval Evaluation System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B.svg)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/VectorDB-Chroma-orange.svg)](https://www.trychroma.com)
[![Tests Passing](https://img.shields.io/badge/tests-10%20passed-brightgreen.svg)]()

A modular, citation-grounded Retrieval-Augmented Generation (RAG) system built over a curated corpus of **30 curated research papers (1,242 structured chunks)** in **Efficient LLMs and Transformer Optimization**.

Features **structure-aware PDF parsing**, **hybrid retrieval (Dense + BM25 via Reciprocal Rank Fusion)**, **cross-encoder reranking**, **LLM-as-a-judge faithfulness auditing**, **RAG-grounded multi-paper comparison**, and a rigorous **empirical retrieval benchmark** evaluated with custom metric implementations (Precision@K, Recall@K, MRR, nDCG@K) at both **chunk-level and paper-level granularity**.

---

## 🏛️ System Architecture

```
                                  [ User Query ]
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
     ┌────────────────────────┐                     ┌────────────────────────┐
     │   Dense Semantic IR    │                     │   Sparse Keyword IR    │
     │  (BAAI/bge-base-en)    │                     │      (BM25Plus)        │
     └───────────┬────────────┘                     └────────────┬───────────┘
                 │ (Top-20 Dense Chunks)                         │ (Top-20 Sparse Chunks)
                 └───────────────────────┬───────────────────────┘
                                         ▼
                        ┌─────────────────────────────────┐
                        │   Reciprocal Rank Fusion (RRF)  │
                        │   RRF(d) = Σ 1/(60 + r_m(d))   │
                        └────────────────┬────────────────┘
                                         │ (Top-20 Fused Candidates)
                                         ▼
                        ┌─────────────────────────────────┐
                        │     Cross-Encoder Reranker      │
                        │  (ms-marco-MiniLM-L-6-v2)       │
                        └────────────────┬────────────────┘
                                         │ (Top-K Reranked Chunks)
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
    ┌─────────────────────────┐                     ┌─────────────────────────┐
    │ Grounded Gen + Citation │                     │  LLM-as-a-Judge Audit   │
    │ [Paper Title, Sec, p.X] │                     │   Faithfulness Score    │
    └─────────────────────────┘                     └─────────────────────────┘
```

---

## 📊 Empirical Retrieval Benchmark

Retrieval performance was evaluated on a systematically curated dataset of **30 domain queries** categorized into 6 distinct intent types (*Conceptual, Mechanistic, Comparative, Numerical, Terminology, Trade-off*) with hand-labeled `gold_chunk_ids` and `target_paper_ids`.

### ⚠️ Two Evaluation Granularities

The benchmark evaluates at two distinct levels:

| Granularity | What it measures | Why it matters |
|:---|:---|:---|
| **Chunk-Level (Primary)** | Does the exact supporting passage appear in top-K results? | Tests precision of passage retrieval — the real signal for RAG generation quality |
| **Paper-Level (Secondary)** | Does the correct source paper appear in top-K results? | Tests broad document routing; weaker signal since papers contain many chunks |

---

### 1. Primary: Gold Chunk-Level Retrieval Ablation

*(Measured across 30 queries with verified `gold_chunk_ids`)*

| Retriever Architecture | Chunk Recall@1 | Chunk Recall@3 | Chunk Recall@5 | Chunk MRR | Chunk nDCG@5 | Warm Latency | Cold Start |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Dense (BGE-base) | 7.8% | 21.1% | 31.1% | 0.3806 | 0.2722 | ~53 ms | ~30 s |
| Sparse (BM25Plus) | 6.7% | 28.9% | 41.1% | 0.4167 | 0.3387 | ~5 ms | ~8 ms |
| **Hybrid (Dense + BM25 RRF)** | **13.3%** | **31.1%** | 37.8% | **0.5261** | **0.3683** | **~54 ms** | **~77 ms** |
| Hybrid + Cross-Encoder Reranker | 11.1% | 25.6% | 35.6% | 0.4744 | 0.3297 | ~1,473 ms | ~10 s |

> **Latency Note**: Cold-start times include one-time model weight loading from disk. Warm latency is per-query after models are loaded into memory. Dense retrieval warm latency is ~53 ms once the embedding model is cached. BM25 has no model loading overhead (~5 ms warm, ~8 ms cold).

---

### 2. Secondary: Paper-Level Retrieval Ablation

*(Same 30 queries evaluated at document level)*

| Retriever Architecture | Paper Recall@1 | Paper Recall@5 | Paper MRR | Paper nDCG@5 |
|:---|:---:|:---:|:---:|:---:|
| Dense (BGE-base) | 73.3% | 90.0% | 0.8028 | 0.8281 |
| Sparse (BM25Plus) | 71.7% | 90.0% | 0.8011 | 0.8219 |
| **Hybrid (Dense + BM25 RRF)** | **81.7%** | **93.3%** | **0.8650** | **0.8775** |
| Hybrid + Cross-Encoder Reranker | 81.7% | 90.0% | 0.8567 | 0.8673 |

---

### 3. Cross-Encoder Reranker: Where It Helps and Where It Hurts

The cross-encoder **does not uniformly improve** overall retrieval. It trades off higher CPU latency (~1,473 ms vs ~54 ms warm) for marginal shifts that benefit some query categories and hurt others:

| Query Category | Hybrid RRF Chunk R@1 | Reranker Chunk R@1 | Paper R@1 Hybrid | Paper R@1 Reranker | Verdict |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Comparative** | 16.7% | 16.7% | 75.0% | **100.0%** | ✅ Reranker helps paper routing |
| **Mechanistic** | **16.7%** | 11.9% | **75.0%** | 67.9% | ❌ Reranker hurts |
| **Numerical** | 0.0% | 0.0% | 100.0% | 100.0% | ↔ Neutral |
| **Terminology** | 0.0% | 0.0% | 100.0% | 100.0% | ↔ Neutral |
| **Conceptual** | 8.3% | 8.3% | 75.0% | 75.0% | ↔ Neutral |
| **Trade-off** | **16.7%** | 16.7% | 100.0% | 100.0% | ↔ Neutral |

> **Interpretation**: Cross-encoder reranking boosts paper-level routing for **Comparative** queries (where distinguishing subtle architectural differences between adjacent papers matters) but degrades **Mechanistic** query performance because multi-step mathematical explanations span multiple chunks and the cross-encoder's single-passage scoring misses holistic context. For most deployments, **Hybrid RRF is the better default** given its 27× lower warm latency at equal or better chunk-level metrics.

---

## 🔬 Architectural Design Decisions

### 1. Structure-Aware Chunking vs. Fixed-Token Windows
Academic papers have rigid semantic structures (Abstract → Methods → Experiments → Conclusion). Naive fixed-size chunking cuts across section boundaries, splitting algorithms from their definitions. `PDFStructureExtractor` + `StructureAwareChunker` detect font-size hierarchies, chunk strictly within section boundaries, and attach `paper_id`, `title`, `section_title`, `page_start`, `page_end` to every chunk.

### 2. Embedding Model: `BAAI/bge-base-en-v1.5`
Selected for strong MTEB retrieval benchmark performance, compact 768-dimensional representation, and practical CPU inference without high VRAM requirements.

### 3. Reciprocal Rank Fusion (RRF)
Combines sparse and dense rank lists:

$$RRF(d) = \sum_{m \in \{dense, bm25\}} \frac{1}{k + r_m(d)} \quad (k = 60)$$

**Rationale**: Eliminates fragile score calibration (Min-Max normalization, weighted linear combination) which fails when score distributions drift across query types.

### 4. Two-Stage Retrieval (Bi-Encoder + Cross-Encoder)
- **Stage 1** (Bi-encoder / BM25): Fast candidate filtering over 1,242 chunks → top 20 candidates.
- **Stage 2** (Cross-encoder, optional): High-precision joint query-document scoring on the top 20. Beneficial for comparative queries; not always beneficial overall — see benchmark above.

### 5. Page-Level Citations
Every chunk retains `page_start` and `page_end` extracted from the PDF. The generator formats inline citations as `[Paper Title, Section, p. X]`, making every claim in generated answers directly traceable to a specific page.

### 6. RAG-Grounded Paper Comparison
Given 2–3 selected papers, the comparator retrieves targeted methodology and empirical chunks for each paper, builds a structured prompt across 5 dimensions, and passes the full retrieval context + prompt directly to the LLM for a cited comparative synthesis.

---

## 📚 Indexed Corpus: 30 Curated Research Papers

- **Attention Optimization**: FlashAttention-1, FlashAttention-2, FlashAttention-3, Flash-Decoding, GQA/MQA, RetNet, Longformer, Mamba
- **Parameter-Efficient Fine-Tuning**: LoRA, QLoRA, DoRA, Unsloth Triton Kernels
- **Quantization & Compression**: GPTQ, AWQ, SmoothQuant, SpQR, LLM.int8(), KIVI, DistilBERT
- **LLM Serving & KV Cache**: vLLM (PagedAttention), SGLang (RadixAttention), StreamingLLM, H2O
- **Inference Acceleration & Scaling**: Speculative Decoding, Medusa, EAGLE, DeepSpeed ZeRO, Chinchilla Scaling Laws, LLaMA, Llama 2

---

## 🚀 Quickstart

### 1. Install dependencies
```bash
git clone https://github.com/Yuvraj2637/ScholarRAG.git
cd ScholarRAG
pip install -r requirements.txt
```

### 2. Configure environment (optional)
```bash
# Copy and edit the example env file
cp .env.example .env
# LLM_PROVIDER=gemini | openai | mock
# GEMINI_API_KEY=your_key
```
> Without an API key the system falls back to grounded local synthesis using retrieved passages.

### 3. Run automated tests
```bash
pytest tests/ -v
```

### 4. Run retrieval benchmark
```bash
python -m src.evaluation.benchmark_retrieval
```

### 5. Launch interfaces

```bash
# Streamlit Research Dashboard
streamlit run src/ui/app.py

# FastAPI Backend
uvicorn src.api.main:app --reload --port 8000
# Swagger docs → http://localhost:8000/docs
```

---

## 🗂️ Project Structure

```
RAG_Project/
├── .gitignore                   # Excludes cache, venv, .env, Chroma DB, raw PDFs
├── requirements.txt             # All dependencies
├── data/
│   ├── processed/               # Chunks JSON + ChromaDB vector store
│   └── eval/                    # 30-query benchmark dataset with gold_chunk_ids
├── results/
│   ├── retrieval_benchmark.json # Machine-readable evaluation results + latencies
│   └── retrieval_benchmark.md   # Published benchmark report
├── src/
│   ├── config.py
│   ├── ingestion/
│   │   ├── paper_catalog.py     # 30-paper metadata registry
│   │   ├── pdf_extractor.py     # Structure-aware PDF parser (page_start/page_end)
│   │   ├── chunker.py           # Section-bounded chunker
│   │   └── run_ingestion.py
│   ├── retrieval/
│   │   ├── dense.py             # BGE-base ChromaDB retriever (lazy-loaded)
│   │   ├── bm25.py              # BM25Plus sparse retriever
│   │   ├── hybrid.py            # Reciprocal Rank Fusion
│   │   ├── reranker.py          # Cross-Encoder rescorer (lazy-loaded)
│   │   └── pipeline.py          # Unified multi-stage coordinator
│   ├── evaluation/
│   │   ├── retrieval_metrics.py # Custom bounded-nDCG metrics implementation
│   │   └── benchmark_retrieval.py # Chunk-level + paper-level benchmark runner
│   ├── generation/
│   │   ├── generator.py         # Citation-grounded LLM engine
│   │   ├── faithfulness.py      # LLM-as-a-judge hallucination auditor
│   │   └── comparator.py        # RAG-grounded multi-paper comparator
│   ├── api/main.py              # FastAPI REST backend
│   └── ui/app.py                # Streamlit research dashboard
└── tests/
    ├── test_chunker.py
    ├── test_retrieval.py
    ├── test_metrics.py          # Duplicate-chunk nDCG bound tests
    └── test_api.py
```
