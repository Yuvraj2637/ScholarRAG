"""Ingestion pipeline orchestrator: Downloads PDFs, extracts sections, chunks, and persists structured JSON."""

import json
from pathlib import Path
from typing import List
from tqdm import tqdm
from src.config import settings
from src.ingestion.paper_catalog import CURATED_PAPERS, PAPER_CATALOG_MAP
from src.ingestion.downloader import download_paper
from src.ingestion.pdf_extractor import PDFStructureExtractor
from src.ingestion.chunker import StructureAwareChunker, Chunk
import httpx

def run_pipeline(download_limit: int | None = None) -> List[Chunk]:
    settings.ensure_directories()
    extractor = PDFStructureExtractor()
    chunker = StructureAwareChunker()

    all_chunks: List[Chunk] = []
    processed_papers = 0

    print("=" * 60)
    print("PHASE 1: INGESTION & STRUCTURE-AWARE CHUNKING")
    print("=" * 60)

    with httpx.Client() as client:
        papers_to_process = CURATED_PAPERS[:download_limit] if download_limit else CURATED_PAPERS

        for paper in tqdm(papers_to_process, desc="Processing Papers"):
            # Step 1: Download or locate PDF
            try:
                pdf_path = download_paper(paper, settings.PAPERS_DIR, client)
            except Exception as e:
                print(f"\n[WARN] Could not fetch {paper.paper_id}: {e}")
                # Check if local PDF already exists anyway
                pdf_path = settings.PAPERS_DIR / f"{paper.paper_id}.pdf"
                if not pdf_path.exists():
                    continue

            # Step 2: Extract structured document
            try:
                parsed_doc = extractor.parse_pdf(
                    pdf_path=pdf_path,
                    paper_id=paper.paper_id,
                    title=paper.title,
                    authors=paper.authors,
                    year=paper.year
                )
            except Exception as e:
                print(f"\n[ERROR] Failed to extract structure for {paper.paper_id}: {e}")
                continue

            # Step 3: Chunk with section & sentence awareness
            paper_chunks = chunker.chunk_document(parsed_doc, primary_topic=paper.primary_topic)
            all_chunks.extend(paper_chunks)
            processed_papers += 1

    # Step 4: Persist structured chunks JSON
    chunks_dict = [c.model_dump() for c in all_chunks]
    with open(settings.CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks_dict, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("INGESTION COMPLETED")
    print(f"Total Papers Processed: {processed_papers}")
    print(f"Total Chunks Generated: {len(all_chunks)}")
    if all_chunks:
        avg_tokens = sum(c.token_count for c in all_chunks) / len(all_chunks)
        print(f"Average Tokens / Chunk: {avg_tokens:.1f}")
        print(f"Saved Structured Chunks to: {settings.CHUNKS_FILE}")
    print("=" * 60)

    return all_chunks

if __name__ == "__main__":
    run_pipeline()
