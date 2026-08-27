# Project Scope & Architecture Contract

## 1. Domain Lock
- **Domain**: **Efficient LLMs, Attention Mechanisms, and Transformer Optimization**
- **Corpus**: 30–40 peer-reviewed & seminal arXiv papers covering:
  - Memory-efficient attention (e.g., FlashAttention-1/2/3, Multi-Head Latent Attention)
  - Parameter-Efficient Fine-Tuning (e.g., LoRA, QLoRA, DoRA, Prefix-Tuning)
  - Efficient Serving & KV Caching (e.g., vLLM / PagedAttention, SGLang, RadixAttention)
  - Model Compression & Quantization (e.g., GPTQ, AWQ, SmoothQuant, DistilBERT)
  - Speculative & Fast Decoding (e.g., Speculative Decoding, Medusa, Lookahead)

## 2. In-Scope Commitments (v1 Deliverables)
1. **Structure-Aware Ingestion**: Section-aware parsing (Title, Abstract, Method, Results, Conclusion) with rich chunk metadata.
2. **Hybrid Retrieval**: Dense (BGE-base) + BM25 Sparse Search fused via Reciprocal Rank Fusion (RRF).
3. **Cross-Encoder Reranking**: Rescoring top-$N$ candidate chunks to top-$K$ using `ms-marco-MiniLM-L-6-v2`.
4. **Curated Ground Truth Benchmark**: 30–50 hand-crafted question-passage-answer triplets.
5. **Retrieval Evaluation**: Hand-implemented Precision@K, Recall@K, MRR, and nDCG metrics with benchmark table.
6. **Grounded Generation & Citation**: Prompt engineering enforcing verbatim bracketed citations `[Paper, Section]`.
7. **Faithfulness Evaluation**: LLM-as-a-judge claim extraction and passage verification.
8. **Paper Comparison Engine**: Structured comparative analysis across selected papers.
9. **Interactive UI & REST API**: Streamlit web frontend and FastAPI backend.

## 3. Explicitly Out of Scope (Scope-Creep Guardrails)
- ❌ Multi-agent swarm orchestration / autonomous agent loops.
- ❌ Knowledge Graph construction (Neo4j / GraphRAG).
- ❌ Complex multimodal image/figure OCR and diagram vector search.
- ❌ User authentication, billing, or enterprise multi-tenancy.
- ❌ Distributed cloud clusters / complex Kubernetes deployments.
