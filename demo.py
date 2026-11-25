"""Run the keyless ScholarRAG prototype with `streamlit run demo.py`."""

import json
from pathlib import Path

import streamlit as st

from demo_corpus import EXAMPLES, PAPERS, StarterCorpus
from src.ingestion.paper_catalog import PAPER_CATALOG_MAP


st.set_page_config(page_title="ScholarRAG", layout="wide")
st.markdown("""
<style>
[data-testid="stMainBlockContainer"] { max-width: 1120px; padding-top: 2.6rem; }
[data-testid="stAppViewContainer"] h1 { letter-spacing: -0.045em; font-weight: 750; }
[data-testid="stAppViewContainer"] h2 { letter-spacing: -0.025em; }
[data-testid="stButton"] button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] {
  color: white; background: #244a43; border-color: #244a43;
}
[data-testid="stButton"] button[kind="primary"]:hover,
[data-testid="stFormSubmitButton"] button[kind="primary"]:hover {
  color: white; background: #183a34; border-color: #183a34;
}
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parent


@st.cache_resource
def load_corpus() -> StarterCorpus:
    return StarterCorpus()


def answer_question() -> None:
    question = st.session_state.question.strip()
    st.session_state.result = (question, load_corpus().answer(question))


def select_example(question: str) -> None:
    st.session_state.question = question
    answer_question()


def citation(paper_id, note) -> str:
    paper = PAPERS[paper_id]
    return f"[{paper.label} · {note.section} · p. {note.page}]({paper.pdf_url(paper_id, note.page)})"


st.title("ScholarRAG")
st.markdown("Ask questions about efficient language models and jump directly to the evidence.")
st.caption(f"Starter corpus · {len(PAPERS)} papers · BM25Plus retrieval over page-linked research notes")

ask, compare, library, benchmark = st.tabs(("Ask a question", "Compare papers", "Paper library", "Benchmark"))

with ask:
    work, examples = st.columns([3, 2], gap="large")
    with work:
        with st.form("ask"):
            st.text_input("Research question", key="question", max_chars=350,
                          placeholder="How does FlashAttention reduce GPU memory traffic?")
            st.form_submit_button("Find evidence", type="primary", on_click=answer_question)
    with examples:
        st.markdown("**Try a question**")
        for label, example in EXAMPLES.items():
            st.button(label, on_click=select_example, args=(example,), width="stretch")

    if "result" in st.session_state:
        question, evidence = st.session_state.result
        if not question:
            st.info("Enter a question or choose one of the examples.")
        elif not evidence:
            st.info("No matching evidence in the starter corpus. Try a paper name or a specific method.")
        else:
            st.subheader("Evidence-backed answer")
            st.caption(question)
            st.caption("Selected research notes · source pages open in the linked paper PDFs")
            for paper_id, note in evidence:
                with st.container(border=True):
                    st.markdown(note.text)
                    st.markdown(citation(paper_id, note))

with compare:
    st.subheader("Compare methods and findings")
    paired = st.columns(2)
    options = list(PAPERS)
    left = paired[0].selectbox("First paper", options, format_func=lambda pid: PAPERS[pid].label)
    right = paired[1].selectbox("Second paper", options, index=1, format_func=lambda pid: PAPERS[pid].label)
    if left == right:
        st.info("Choose two different papers.")
    else:
        for column, paper_id in zip(paired, (left, right)):
            with column.container(border=True):
                st.markdown(f"### {PAPERS[paper_id].label}")
                for note in PAPERS[paper_id].notes:
                    st.markdown(f"**{note.facet}**")
                    st.markdown(note.text)
                    st.markdown(citation(paper_id, note))

with library:
    st.subheader("Papers in the starter corpus")
    for paper_id, paper in PAPERS.items():
        meta = PAPER_CATALOG_MAP[paper_id]
        with st.expander(f"{paper.label} · {meta.year} · {meta.primary_topic}"):
            st.markdown(f"**{meta.title}**")
            st.write(", ".join(meta.authors))
            st.markdown(f"[Open paper PDF]({paper.pdf_url(paper_id)})")

with benchmark:
    st.subheader("Full-corpus retrieval benchmark")
    st.caption("Recorded results from the repository’s 30-query evaluation. These numbers describe the full index.")
    with (ROOT / "results" / "retrieval_benchmark.json").open(encoding="utf-8") as file:
        results = json.load(file)
    labels = {"dense": "Dense", "bm25": "BM25Plus", "hybrid": "Hybrid", "hybrid_rerank": "Hybrid + reranker"}
    rows = [
        {
            "Retriever": labels[mode],
            "Passage recall @1": f"{stats['r@1']:.1%}",
            "Paper recall @1": f"{results['paper_summary'][mode]['r@1']:.1%}",
            "Warm latency": f"{stats['warm_latency_ms']:.0f} ms",
        }
        for mode, stats in results["chunk_summary"].items()
    ]
    st.table(rows)
