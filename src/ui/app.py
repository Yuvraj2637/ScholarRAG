import sys
from pathlib import Path
import json
import streamlit as st
import pandas as pd

# Add project root to sys.path so 'src' can be imported when running streamlit from any directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configure page layout & styling
st.set_page_config(
    page_title="ScholarRAG: Transformer Efficiency Research Hub",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.config import settings
from src.ingestion.paper_catalog import CURATED_PAPERS, PAPER_CATALOG_MAP
from src.retrieval.pipeline import RetrievalPipeline
from src.generation.generator import ResearchGenerator
from src.generation.faithfulness import FaithfulnessJudge
from src.generation.comparator import PaperComparator

# Cache heavyweight components
@st.cache_resource(show_spinner="Warming up Vector DB & Neural Models...")
def load_rag_pipeline():
    pipeline = RetrievalPipeline()
    generator = ResearchGenerator()
    judge = FaithfulnessJudge()
    comparator = PaperComparator(pipeline)
    return pipeline, generator, judge, comparator

pipeline, generator, judge, comparator = load_rag_pipeline()

# --- Sidebar ---
st.sidebar.title("⚡ ScholarRAG")
st.sidebar.markdown("**Production RAG on Transformer Optimization**")
st.sidebar.markdown(f"📚 **Corpus Size**: {len(CURATED_PAPERS)} Seminal Papers")
st.sidebar.markdown(f"🧩 **Indexed Chunks**: {len(pipeline.chunks)} Section Chunks")

st.sidebar.divider()
st.sidebar.markdown("### ⚙️ System Specs")
st.sidebar.markdown(f"- **Embedding**: `{settings.EMBEDDING_MODEL_NAME.split('/')[-1]}`")
st.sidebar.markdown(f"- **Reranker**: `{settings.RERANKER_MODEL_NAME.split('/')[-1]}`")
st.sidebar.markdown(f"- **Sparse Engine**: `BM25Plus`")
st.sidebar.markdown(f"- **Fusion**: `Reciprocal Rank Fusion (k={settings.RRF_K})`")

# --- Main Navigation Tabs ---
tab_chat, tab_compare, tab_benchmark, tab_corpus = st.tabs([
    "🔍 Research Q&A",
    "⚖️ Paper Comparator",
    "📊 Retrieval Benchmark",
    "📚 Corpus Explorer"
])

# ==========================================
# TAB 1: RESEARCH Q&A
# ==========================================
with tab_chat:
    st.header("Search & Grounded Question Answering")
    st.caption("Ask architectural, empirical, or theoretical questions about efficient Transformers.")

    col1, col2 = st.columns([3, 1])
    with col2:
        retrieval_mode = st.selectbox(
            "Retrieval Architecture",
            options=["hybrid_rerank", "hybrid", "dense", "bm25"],
            format_func=lambda x: {
                "hybrid_rerank": "🎯 Hybrid + Cross-Encoder Reranker",
                "hybrid": "⚡ Hybrid (Dense + BM25 RRF)",
                "dense": "🧠 Dense Only (BGE-base)",
                "bm25": "🔍 Sparse Only (BM25Plus)"
            }[x],
            index=0
        )
        top_k = st.slider("Top Chunks (K)", min_value=1, max_value=10, value=4)
        audit_faithfulness = st.checkbox("Audit Faithfulness (Judge)", value=True)

    with col1:
        example_questions = [
            "How does FlashAttention avoid materializing the full attention matrix in HBM?",
            "What is NormalFloat4 (NF4) and Double Quantization in QLoRA?",
            "How does PagedAttention eliminate memory fragmentation in vLLM?",
            "Why does AWQ protect the top 1% salient weights based on activation magnitudes?",
            "What compute-optimal scaling relationship was established by Chinchilla?",
            "How does Mamba achieve linear-time sequence modeling with selective state spaces?"
        ]
        selected_example = st.selectbox("💡 Or select a sample query:", [""] + example_questions)
        
        user_query = st.text_input(
            "Enter your research question:",
            value=selected_example if selected_example else "How does FlashAttention avoid materializing the full attention matrix in HBM?"
        )

        search_btn = st.button("🚀 Run Research Query", type="primary")

    if search_btn and user_query.strip():
        with st.spinner("Retrieving relevant passages and synthesizing cited answer..."):
            chunks = pipeline.retrieve(user_query, mode=retrieval_mode, top_k=top_k)
            gen_result = generator.generate(user_query, chunks)
            
            faithfulness_report = None
            if audit_faithfulness and chunks:
                faithfulness_report = judge.evaluate(gen_result.answer, chunks)

        st.subheader("💡 Grounded Research Answer")
        st.markdown(gen_result.answer)

        # Faithfulness Badge
        if faithfulness_report:
            score = faithfulness_report.faithfulness_score * 100
            if score >= 80:
                st.success(f"🛡️ **Faithfulness Score: {score:.0f}%** ({faithfulness_report.supported_claims}/{faithfulness_report.total_claims} claims verified by `{faithfulness_report.judge_model}`)")
            else:
                st.warning(f"⚠️ **Faithfulness Score: {score:.0f}%** ({faithfulness_report.supported_claims}/{faithfulness_report.total_claims} claims verified)")
            
            with st.expander("🔎 View Claim-by-Claim Verification Audit"):
                for audit in faithfulness_report.audits:
                    status_emoji = "✅" if audit.verdict == "SUPPORTED" else "❌"
                    st.markdown(f"{status_emoji} **Claim**: *\"{audit.claim}\"*")
                    st.caption(f"**Verdict**: `{audit.verdict}` — {audit.reasoning}")
                    st.divider()

        # Retrieved Context Chunks
        st.subheader(f"📑 Retrieved Context Evidence (Top {len(chunks)})")
        for idx, c in enumerate(chunks, 1):
            score_type = "Rerank Score" if "rerank_score" in c else "Score"
            score_val = c.get("score", 0.0)
            with st.expander(f"Chunk #{idx} | **{c.get('paper_title')}** ({c.get('year')}) — *{c.get('section_title')}* [{score_type}: {score_val:.4f}]"):
                st.markdown(f"**Paper ID**: `{c.get('paper_id')}` | **Section**: `{c.get('section_title')}`")
                st.markdown(f"**Text Content**:\n{c.get('text')}")


# ==========================================
# TAB 2: PAPER COMPARATOR
# ==========================================
with tab_compare:
    st.header("Multi-Paper Comparative Analysis")
    st.caption("Select 2–3 papers to generate a structured methodology and trade-off comparison matrix.")

    paper_options = list(PAPER_CATALOG_MAP.keys())
    default_selection = ["flashattention_v1", "flashattention_v2", "flash_attention_3"]

    selected_pids = st.multiselect(
        "Select Papers to Compare:",
        options=paper_options,
        default=[p for p in default_selection if p in paper_options],
        format_func=lambda x: f"{PAPER_CATALOG_MAP[x].title} ({PAPER_CATALOG_MAP[x].year})"
    )

    if st.button("⚖️ Compare Papers", type="primary"):
        if len(selected_pids) < 2:
            st.error("Please select at least 2 papers to compare.")
        else:
            with st.spinner("Extracting methodology and compiling comparative matrix..."):
                report = comparator.compare_papers(selected_pids)

            st.markdown(report.comparative_synthesis)

            st.subheader("📋 Architectural & Empirical Comparison Matrix")
            # Build DataFrame for display
            matrix_data = []
            for dim in report.summary_matrix:
                row = {"Dimension": dim.dimension_name}
                for pid in selected_pids:
                    title_short = PAPER_CATALOG_MAP[pid].paper_id
                    row[title_short] = dim.paper_analyses.get(pid, "N/A")
                matrix_data.append(row)

            df_matrix = pd.DataFrame(matrix_data)
            st.table(df_matrix)


# ==========================================
# TAB 3: RETRIEVAL BENCHMARK
# ==========================================
with tab_benchmark:
    st.header("Empirical Retrieval Benchmark")
    st.caption("Rigorous evaluation over 30 hand-curated domain questions across 6 query categories.")

    benchmark_json_path = settings.RESULTS_DIR / "retrieval_benchmark.json"
    if benchmark_json_path.exists():
        with open(benchmark_json_path, "r", encoding="utf-8") as f:
            bench_data = json.load(f)

        chunk_summary = bench_data.get("chunk_summary", bench_data.get("summary", {}))
        paper_summary = bench_data.get("paper_summary", {})

        mode_labels = {
            "dense": "Dense (BGE-base)",
            "bm25": "Sparse (BM25Plus)",
            "hybrid": "Hybrid (Dense + BM25 RRF)",
            "hybrid_rerank": "Hybrid + Cross-Encoder Reranker"
        }

        # Chunk-level DataFrame
        chunk_rows = []
        for mode, m in chunk_summary.items():
            chunk_rows.append({
                "Retriever Architecture": mode_labels.get(mode, mode),
                "Chunk Recall@1": f"{m.get('r@1', 0)*100:.1f}%",
                "Chunk Recall@3": f"{m.get('r@3', 0)*100:.1f}%",
                "Chunk Recall@5": f"{m.get('r@5', 0)*100:.1f}%",
                "Chunk MRR": f"{m.get('mrr', 0):.4f}",
                "Chunk nDCG@5": f"{m.get('ndcg@5', 0):.4f}",
                "Warm Latency": f"{m.get('warm_latency_ms', 0):.1f} ms",
                "Cold Start": f"{m.get('cold_latency_ms', 0):.0f} ms",
                "_raw_r1": m.get('r@1', 0),
                "_raw_mrr": m.get('mrr', 0),
            })

        df_chunk = pd.DataFrame(chunk_rows)
        st.subheader("🎯 Primary Benchmark: Gold Chunk-Level Retrieval")
        st.dataframe(df_chunk.drop(columns=["_raw_r1", "_raw_mrr"]), use_container_width=True)

        # Paper-level DataFrame
        if paper_summary:
            paper_rows = []
            for mode, pm in paper_summary.items():
                paper_rows.append({
                    "Retriever Architecture": mode_labels.get(mode, mode),
                    "Paper Recall@1": f"{pm.get('r@1', 0)*100:.1f}%",
                    "Paper Recall@5": f"{pm.get('r@5', 0)*100:.1f}%",
                    "Paper MRR": f"{pm.get('mrr', 0):.4f}",
                    "Paper nDCG@5": f"{pm.get('ndcg@5', 0):.4f}"
                })
            st.subheader("📚 Secondary Benchmark: Paper-Level Retrieval")
            st.dataframe(pd.DataFrame(paper_rows), use_container_width=True)

        st.subheader("📈 Metric Comparison Charts")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("**Chunk Recall@1 (Top-1 Chunk Accuracy)**")
            chart_df = df_chunk.set_index("Retriever Architecture")[["_raw_r1"]]
            chart_df.columns = ["Chunk Recall@1"]
            st.bar_chart(chart_df)

        with col_c2:
            st.markdown("**Chunk Mean Reciprocal Rank (MRR)**")
            mrr_df = df_chunk.set_index("Retriever Architecture")[["_raw_mrr"]]
            mrr_df.columns = ["Chunk MRR"]
            st.bar_chart(mrr_df)

        st.subheader("🧪 Benchmark Queries Inspector")
        queries = bench_data.get("queries", [])
        if queries:
            selected_q_idx = st.selectbox(
                "Inspect Query Performance:",
                range(len(queries)),
                format_func=lambda i: f"Q{i+1} [{queries[i].get('query_type', '')}]: {queries[i]['query']}"
            )
            q_info = queries[selected_q_idx]
            st.markdown(f"**Query**: {q_info['query']}")
            st.markdown(f"**Target Papers**: `{q_info.get('target_papers', q_info.get('targets'))}` | **Category**: `{q_info.get('query_type')}`")
            st.markdown(f"**Gold Chunks**: `{q_info.get('gold_chunks', [])}`")
            st.json(q_info.get("retriever_hits", {}))

    else:
        st.info("Run `python -m src.evaluation.benchmark_retrieval` to generate live benchmark data.")


# ==========================================
# TAB 4: CORPUS EXPLORER
# ==========================================
with tab_corpus:
    st.header("Corpus Explorer")
    st.caption("Searchable catalog of 30 seminal research papers indexed in ScholarRAG.")

    corpus_data = []
    for p in CURATED_PAPERS:
        corpus_data.append({
            "Paper ID": p.paper_id,
            "Title": p.title,
            "Topic": p.primary_topic,
            "Year": p.year,
            "Authors": ", ".join(p.authors[:2]) + (" et al." if len(p.authors) > 2 else ""),
            "arXiv": f"https://arxiv.org/abs/{p.arxiv_id}",
            "Summary": p.summary
        })

    df_corpus = pd.DataFrame(corpus_data)
    
    topic_filter = st.multiselect(
        "Filter by Topic:",
        options=sorted(list(set(df_corpus["Topic"]))),
        default=sorted(list(set(df_corpus["Topic"])))
    )
    
    filtered_df = df_corpus[df_corpus["Topic"].isin(topic_filter)]
    st.dataframe(filtered_df, use_container_width=True)
