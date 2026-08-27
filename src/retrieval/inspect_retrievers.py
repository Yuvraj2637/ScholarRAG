"""CLI tool for comparing Dense, BM25, Hybrid, and Reranked retrieval outputs side-by-side."""

import sys
import argparse

# Ensure UTF-8 output encoding on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.retrieval.pipeline import RetrievalPipeline

def clean_console_text(text: str) -> str:
    """Safely escape or normalize Unicode math symbols for terminal display."""
    return text.encode("utf-8", errors="replace").decode("utf-8")

def main():
    parser = argparse.ArgumentParser(description="Side-by-side retriever inspection tool.")
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default="How does FlashAttention avoid materializing the full attention matrix in HBM?",
        help="Search query"
    )
    parser.add_argument(
        "--top_k",
        "-k",
        type=int,
        default=3,
        help="Number of chunks to print per retriever"
    )
    args = parser.parse_args()

    pipeline = RetrievalPipeline()
    print("\n" + "=" * 80)
    print(f"QUERY: {args.query}")
    print("=" * 80)

    all_results = pipeline.retrieve_all_modes(args.query, top_k=args.top_k)

    for mode, results in all_results.items():
        print(f"\n>>> RETRIEVER: {mode.upper()} (Top {len(results)})")
        print("-" * 80)
        for rank, res in enumerate(results, 1):
            score_str = f"Score: {res.get('score', 0.0):.4f}"
            print(f"[{rank}] Paper: {res['paper_id']} | Sec: {clean_console_text(res['section_title'])} | {score_str}")
            preview = clean_console_text(res['text'][:200].replace('\n', ' '))
            print(f"    Text Preview: {preview}...\n")

if __name__ == "__main__":
    main()
