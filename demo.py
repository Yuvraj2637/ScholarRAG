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
    --paper: #faf9f6;
    --ink: #202723;
    --muted: #516059;
    --line: #dce1db;
    --accent: #1c5b43;
    color: var(--ink);
    background: var(--paper);
    color-scheme: light;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] {
    max-width: 1080px;
    padding: 3rem 2.25rem 4rem;
}
.scholar-brand {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
    padding-bottom: .8rem;
    border-bottom: 1px solid var(--line);
}
.scholar-wordmark { font-weight: 730; letter-spacing: -.02em; }
.scholar-wordmark::before {
    content: "";
    display: inline-block;
    width: .55rem;
    height: .55rem;
    margin-right: .5rem;
    background: var(--accent);
    vertical-align: .11em;
}
.scholar-brand a {
    color: var(--accent);
    font-size: .88rem;
    font-weight: 650;
    text-decoration: none;
}
.scholar-brand a:hover { text-decoration: underline; }
.scholar-intro {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 15rem);
    align-items: end;
    gap: .75rem 2rem;
    padding: .8rem 0 .35rem;
}
[data-testid="stAppViewContainer"] .scholar-intro h1 {
    color: var(--ink);
    font-family: Georgia, serif;
    font-size: clamp(2rem, 5vw, 3.3rem);
    font-weight: 400;
    letter-spacing: -.045em;
    line-height: 1.1;
    padding: 0;
    margin: 0 0 .6rem;
}
.scholar-intro p { color: var(--muted); margin: 0; }
.scholar-scope {
    padding: .15rem 0 .15rem .85rem;
    border-left: 2px solid #bfd0c4;
    font-size: .85rem;
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
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color: var(--accent); border-bottom-color: var(--accent); }
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background-color: var(--accent); }
[data-testid="stTabs"] [data-baseweb="tab-border"] { background-color: transparent; }
[data-testid="stTabs"] [role="tabpanel"] { padding-top: .45rem; }
[data-testid="stForm"], [class*="st-key-compare-"] {
    background: #fff;
    border-color: var(--line) !important;
    border-radius: 8px !important;
}
.st-key-examples [data-testid="stHorizontalBlock"] { gap: .55rem; }
.st-key-answer { border-top: 1px solid var(--line); padding-top: .65rem; }
[class*="st-key-evidence-"] { padding-bottom: .25rem; border-bottom: 1px solid var(--line); }
.st-key-sources {
    background: #eef2ed;
    border-left: 2px solid #c6d5ca;
    padding: .65rem .85rem;
}
[data-testid="stTextInput"] [data-baseweb="input"],
[data-testid="stTextInputRootElement"],
[data-testid="stSelectbox"] [role="group"],
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    color: var(--ink);
    background: #fff;
    border-color: #c8d3cc;
    border-radius: 8px;
}
[data-testid="stTextInput"] input, [data-testid="stSelectbox"] input {
    color: var(--ink);
    background: transparent;
    caret-color: var(--accent);
}
[data-testid="stTextInput"] input::placeholder { color: #516059; opacity: 1; }
[data-testid="stSelectboxVirtualDropdown"] {
    color: #202723;
    background: #fff;
    border: 1px solid #dce1db;
}
[data-testid="stSelectboxVirtualDropdown"] [role="option"] { color: #202723; }
[data-testid="stSelectboxVirtualDropdown"] [role="option"]:is([data-focused], [aria-selected="true"]) {
    background: #e8f0ea;
}
[data-testid="stAppViewContainer"] button:is([kind="primary"], [kind="primaryFormSubmit"]) {
    color: #fff;
    background: #1c5b43;
    border: 1px solid #1c5b43;
    border-radius: 8px;
}
[data-testid="stAppViewContainer"] button:is([kind="primary"], [kind="primaryFormSubmit"]) p { color: #fff !important; }
[data-testid="stAppViewContainer"] button:is([kind="primary"], [kind="primaryFormSubmit"]):hover { background: #144633; }
.st-key-examples button {
    background: transparent;
    border: 1px solid #cdd8d0;
    border-radius: 8px;
    text-align: left;
    color: var(--accent);
}
.st-key-examples button:hover { background: #eaf0eb; border-color: var(--accent); }
[data-testid="stAppViewContainer"] :is(a, button, input):focus-visible {
    outline: 2px solid #2a7b59;
    outline-offset: 2px;
}
.scholar-result-kicker {
    margin: 0;
    color: #1c5b43;
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
    border-radius: 7px;
}
[data-testid="stTable"] :is(th, td) { color: var(--ink); border-color: var(--line); }
[data-testid="stTable"] th { background: #e8efe9; }
[data-testid="stTable"] tr:nth-child(even) { background: #f1f3ef; }
@media (max-width: 640px) {
    [data-testid="stMainBlockContainer"] { padding: 2.8rem 1rem 2rem; }
    .scholar-intro { grid-template-columns: minmax(0, 1fr); }
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
<header class="scholar-brand">
  <span class="scholar-wordmark">ScholarRAG</span>
  <a href="https://github.com/Yuvraj2637/ScholarRAG" target="_blank" rel="noopener noreferrer">Source on GitHub ↗</a>
</header>
<section class="scholar-intro" aria-labelledby="scholar-heading">
  <div>
    <h1 id="scholar-heading">Search research papers</h1>
    <p>Ask a question about efficient language models and follow the answer to the source.</p>
  </div>
  <p class="scholar-scope">This demo searches notes from {len(PAPERS)} papers, with links to the original PDF pages.</p>
</section>
""")

ask, compare, library, benchmark = st.tabs(("Ask a question", "Compare papers", "Paper library", "Benchmark"))

with ask:
    with st.form("ask"):
        field, action = st.columns([5, 1], vertical_alignment="bottom")
        field.text_input("Research question", key="question", max_chars=350,
                         placeholder="How does FlashAttention reduce GPU memory traffic?")
        action.form_submit_button("Find evidence", type="primary", on_click=answer_question, width="stretch")

    with st.container(key="examples"):
        st.caption("Example questions")
        example_columns = st.columns(len(EXAMPLES))
        for column, (label, example) in zip(example_columns, EXAMPLES.items()):
            column.button(label, on_click=select_example, args=(example,), width="stretch")

    question, evidence = st.session_state.result
    if not question:
        st.info("Enter a question or choose one of the examples.")
    elif not evidence:
        st.info("No matching evidence in the starter corpus. Try a paper name or a specific method.")
    else:
        with st.container(key="answer"):
            st.subheader("Evidence-backed answer")
            st.caption(question)
            response, sources = st.columns([2, 1], gap="large")
            with response:
                for index, (_, note) in enumerate(evidence, 1):
                    with st.container(key=f"evidence-{index}"):
                        st.html(f'<p class="scholar-result-kicker">{note.facet} [{index}]</p>')
                        st.markdown(note.text)
            with sources:
                with st.container(key="sources"):
                    st.markdown("**Source pages**")
                    for index, (paper_id, note) in enumerate(evidence, 1):
                        st.markdown(f"{index}. {citation(paper_id, note)}")

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
