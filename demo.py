"""Run the keyless ScholarRAG prototype with `streamlit run demo.py`."""

import json
from pathlib import Path

import streamlit as st

from demo_corpus import EXAMPLES, PAPERS, StarterCorpus
from src.ingestion.paper_catalog import PAPER_CATALOG_MAP


ROOT = Path(__file__).resolve().parent
DEFAULT_QUESTION = EXAMPLES["FlashAttention memory"]

st.set_page_config(page_title="ScholarRAG", layout="wide")
st.html("""
<style>
[data-testid="stAppViewContainer"] {
    --paper: #f8f7f4;
    --ink: #212238;
    --muted: #5b5b72;
    --line: #e0dce8;
    --accent: #5035a4;
    color: var(--ink);
    background: var(--paper);
    color-scheme: light;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] {
    max-width: 1120px;
    padding: 3rem 2.25rem 4rem;
}
.scholar-brand, .scholar-meta {
    display: flex;
    align-items: center;
    gap: .65rem;
}
.scholar-brand { justify-content: space-between; margin: 0 0 1.05rem; }
.scholar-wordmark {
    display: inline-flex;
    align-items: center;
    gap: .65rem;
    color: #242037;
    font-weight: 760;
    letter-spacing: -.025em;
}
.scholar-mark {
    display: grid;
    align-content: center;
    gap: 3px;
    width: 33px;
    height: 33px;
    padding: 7px;
    border-radius: 9px;
    background: #eae3fa;
}
.scholar-mark i { height: 2px; background: #5035a4; border-radius: 2px; }
.scholar-mark i:last-child { width: 65%; }
.scholar-brand a {
    color: #5035a4;
    font-size: .87rem;
    font-weight: 650;
    text-decoration: none;
}
.scholar-brand a:hover { text-decoration: underline; }
.scholar-hero {
    position: relative;
    isolation: isolate;
    overflow: hidden;
    border-radius: 19px;
    background: linear-gradient(105deg, #242040 8%, #34305b 64%, #4b3973);
    padding: clamp(1.4rem, 3.3vw, 2.65rem);
}
.scholar-hero::after {
    position: absolute;
    z-index: -1;
    inset: 0 0 0 auto;
    width: 45%;
    content: "";
    background:
        radial-gradient(ellipse at 84% 105%, #aa8cec65 0%, transparent 64%),
        radial-gradient(circle, #dfd1ff50 1px, transparent 1.4px) 0 0 / 16px 16px;
    mask-image: linear-gradient(90deg, transparent, #000 90%);
    pointer-events: none;
}
[data-testid="stAppViewContainer"] .scholar-hero h1 {
    color: #fff;
    font-family: Georgia, serif;
    font-size: clamp(2.25rem, 5vw, 3.65rem);
    font-weight: 400;
    letter-spacing: -.05em;
    line-height: 1.04;
    padding: 0;
    margin: 0 0 .65rem;
}
.scholar-hero p {
    max-width: 36rem;
    color: #f3ebff;
    font-size: clamp(.94rem, 1.5vw, 1.1rem);
    margin: 0;
}
.scholar-meta { flex-wrap: wrap; margin-top: 1.15rem; }
.scholar-meta span {
    display: inline-block;
    padding: .27rem .65rem;
    border: 1px solid #cbbaf34a;
    border-radius: 999px;
    background: #e4d7ff15;
    color: #f4efff;
    font-size: .8rem;
}
[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] :is(p, h2, h3, h4),
[data-testid="stAppViewContainer"] [data-testid="stHeading"] :is(h2, h3),
[data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] p {
    color: var(--ink);
}
[data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] p { color: var(--muted); }
[data-testid="stAppViewContainer"] [data-testid="stHeading"] :is(h2, h3) {
    letter-spacing: -.025em;
}
[data-testid="stTabs"] [role="tablist"] {
    gap: clamp(.7rem, 3vw, 1.8rem);
    border-bottom: 1px solid var(--line);
}
[data-testid="stTabs"] [role="tablist"]::after { background-color: var(--line); }
[data-testid="stTabs"] [role="tab"] { color: var(--muted); }
[data-testid="stTabs"] [role="tab"] p { color: inherit !important; font-weight: 650; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color: var(--accent); }
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background-color: var(--accent); }
[data-testid="stTabs"] [data-baseweb="tab-border"] { background-color: transparent; }
[data-testid="stTabs"] [role="tabpanel"] { padding-top: .45rem; }
[data-testid="stForm"],
[class*="st-key-evidence-"], [class*="st-key-compare-"] {
    background: #fff;
    border-color: var(--line) !important;
    border-radius: 13px !important;
}
.st-key-examples {
    background: #f0ebfa;
    border: 1px solid #e0d7f3;
    border-radius: 13px;
    padding: .8rem 1rem;
}
.st-key-examples [data-testid="stVerticalBlock"] { gap: .5rem; }
[data-testid="stTextInput"] [data-baseweb="input"],
[data-testid="stTextInputRootElement"],
[data-testid="stSelectbox"] [role="group"],
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    color: var(--ink);
    background: #fff;
    border-color: #d9d2e8;
    border-radius: 8px;
}
[data-testid="stTextInput"] input, [data-testid="stSelectbox"] input {
    color: var(--ink);
    background: transparent;
    caret-color: var(--accent);
}
[data-testid="stTextInput"] input::placeholder { color: #66647b; opacity: 1; }
[data-testid="stSelectboxVirtualDropdown"] {
    color: #212238;
    background: #fff;
    border: 1px solid #e0dce8;
}
[data-testid="stSelectboxVirtualDropdown"] [role="option"] { color: #212238; }
[data-testid="stSelectboxVirtualDropdown"] [role="option"]:is([data-focused], [aria-selected="true"]) {
    background: #f0ebfa;
}
[data-testid="stAppViewContainer"] button:is([kind="primary"], [kind="primaryFormSubmit"]) {
    color: #fff;
    background: #5035a4;
    border: 1px solid #5035a4;
    border-radius: 8px;
}
[data-testid="stAppViewContainer"] button:is([kind="primary"], [kind="primaryFormSubmit"]) p { color: #fff !important; }
[data-testid="stAppViewContainer"] button:is([kind="primary"], [kind="primaryFormSubmit"]):hover { background: #3f268b; }
.st-key-examples button {
    justify-content: flex-start;
    background: #fff;
    border: 1px solid #dfd6ef;
    border-radius: 8px;
    text-align: left;
    color: var(--accent);
}
.st-key-examples button:hover { background: #faf7ff; border-color: var(--accent); }
[data-testid="stAppViewContainer"] :is(a, button, input):focus-visible {
    outline: 2px solid #7555ca;
    outline-offset: 2px;
}
.scholar-result-kicker {
    margin: 0;
    color: #5035a4;
    font-size: .78rem;
    font-weight: 730;
    line-height: 1.25;
}
[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] a {
    color: var(--accent);
    text-underline-offset: .16em;
    overflow-wrap: anywhere;
}
[data-testid="stExpander"] details {
    background: #fff;
    border-color: var(--line);
    border-radius: 10px;
}
[data-testid="stTable"] :is(th, td) { color: var(--ink); border-color: var(--line); }
[data-testid="stTable"] th { background: #eee8f7; }
[data-testid="stTable"] tr:nth-child(even) { background: #f1edf6; }
@media (max-width: 640px) {
    [data-testid="stMainBlockContainer"] { padding: 2.8rem 1rem 2rem; }
    .scholar-hero { border-radius: 13px; }
    [data-testid="stTabs"] [role="tablist"] { gap: .85rem; }
    [data-testid="stTabs"] [data-testid="stHorizontalBlock"] { flex-direction: column; }
    [data-testid="stTabs"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        flex: 0 0 auto !important;
        width: 100% !important;
        min-width: 0;
    }
}
</style>
""")

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
    return f"[{paper.label} · {note.section} · p. {note.page} ↗]({paper.pdf_url(paper_id, note.page)})"


if "question" not in st.session_state:
    select_example(DEFAULT_QUESTION)

st.html(f"""
<div class="scholar-brand">
  <span class="scholar-wordmark"><span class="scholar-mark" aria-hidden="true"><i></i><i></i><i></i></span>ScholarRAG</span>
  <a href="https://github.com/Yuvraj2637/ScholarRAG" target="_blank" rel="noopener noreferrer">GitHub ↗</a>
</div>
<section class="scholar-hero" aria-labelledby="scholar-heading">
  <h1 id="scholar-heading">Search research papers</h1>
  <p>Ask about efficient language models, compare methods, and open the exact pages behind each answer.</p>
  <div class="scholar-meta"><span>{len(PAPERS)} papers in this demo</span></div>
</section>
""")

ask, compare, library, benchmark = st.tabs(("Ask a question", "Compare papers", "Paper library", "Benchmark"))

with ask:
    work, examples = st.columns([2, 1], gap="medium")
    with work:
        with st.form("ask"):
            st.text_input("Research question", key="question", max_chars=350,
                          placeholder="How does FlashAttention reduce GPU memory traffic?")
            st.form_submit_button("Find evidence", type="primary", on_click=answer_question)
    with examples:
        with st.container(key="examples"):
            st.markdown("**Try an example**")
            for label, example in EXAMPLES.items():
                st.button(label, on_click=select_example, args=(example,), width="stretch")
    question, evidence = st.session_state.result
    if not question:
        st.info("Enter a question or choose one of the examples.")
    elif not evidence:
        st.info("No matching evidence in the starter corpus. Try a paper name or a specific method.")
    else:
        st.subheader("Evidence-backed answer")
        st.caption(question)
        for index, (paper_id, note) in enumerate(evidence):
            with st.container(border=True, key=f"evidence-{index}"):
                st.html(f'<p class="scholar-result-kicker">{note.facet}</p>')
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
            with column.container(border=True, key=f"compare-{paper_id}"):
                st.markdown(f"### {PAPERS[paper_id].label}")
                for note in PAPERS[paper_id].notes:
                    st.html(f'<p class="scholar-result-kicker">{note.facet}</p>')
                    st.markdown(note.text)
                    st.markdown(citation(paper_id, note))

with library:
    st.subheader("Papers in the starter corpus")
    for paper_id, paper in PAPERS.items():
        meta = PAPER_CATALOG_MAP[paper_id]
        with st.expander(f"{paper.label} · {meta.year} · {meta.primary_topic}"):
            st.markdown(f"**{meta.title}**")
            st.write(", ".join(meta.authors))
            st.markdown(f"[Open paper PDF ↗]({paper.pdf_url(paper_id)})")

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
