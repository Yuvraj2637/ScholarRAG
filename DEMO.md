# ScholarRAG browser demo

From the repository root, in a Python 3.11–3.14 environment:

```bash
python -m pip install -r requirements-demo.txt
python -m streamlit run demo.py
```

Open the local URL shown by Streamlit, usually `http://localhost:8501`.

- **Ask a question** searches the starter notes with the repository's BM25Plus retriever and links each matched note to the exact PDF page.
- **Compare papers** displays cited notes for a paper's problem, method, and results side by side.
- **Paper library** lists the six papers available in the demo.
- **Benchmark** displays the original repository's recorded results for its separate 30-query evaluation.

The compact starter corpus contains paraphrased notes with version-pinned arXiv citations. It needs no API key, model download, or paper-index setup after dependencies are installed. It returns matching notes for the most relevant paper and reports when it cannot find enough evidence.

For full-corpus retrieval and optional provider-backed generation, install `requirements.txt`, run `python -m src.ingestion.run_ingestion` to build the corpus, then follow the full dashboard or FastAPI instructions in [README.md](README.md).
