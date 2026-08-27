"""Downloads curated research papers from arXiv into local storage."""

import time
from pathlib import Path
import httpx
from tqdm import tqdm
from src.config import settings
from src.ingestion.paper_catalog import CURATED_PAPERS, PaperMetadata

ARXIV_PDF_BASE_URL = "https://arxiv.org/pdf"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ResearchRAGPipeline/1.0"

def download_paper(paper: PaperMetadata, output_dir: Path, client: httpx.Client) -> Path:
    """Download a single paper PDF from arXiv if not already present."""
    output_path = output_dir / f"{paper.paper_id}.pdf"
    if output_path.exists() and output_path.stat().st_size > 10000:
        return output_path

    url = f"{ARXIV_PDF_BASE_URL}/{paper.arxiv_id}.pdf"
    headers = {"User-Agent": USER_AGENT}

    response = client.get(url, headers=headers, follow_redirects=True, timeout=30.0)
    response.raise_for_status()

    if b"%PDF-" not in response.content[:10]:
        raise ValueError(f"Downloaded content for {paper.paper_id} is not a valid PDF.")

    with open(output_path, "wb") as f:
        f.write(response.content)

    return output_path

def download_all_papers(limit: int | None = None) -> list[Path]:
    """Download all curated papers with progress reporting and polite delays."""
    settings.ensure_directories()
    papers_to_fetch = CURATED_PAPERS[:limit] if limit else CURATED_PAPERS
    downloaded_paths = []

    with httpx.Client() as client:
        for paper in tqdm(papers_to_fetch, desc="Downloading Papers"):
            try:
                path = download_paper(paper, settings.PAPERS_DIR, client)
                downloaded_paths.append(path)
                time.sleep(0.5)  # Polite delay between arXiv requests
            except Exception as e:
                print(f"[WARN] Failed to download {paper.paper_id} ({paper.arxiv_id}): {e}")

    return downloaded_paths

if __name__ == "__main__":
    paths = download_all_papers()
    print(f"Successfully verified/downloaded {len(paths)} papers in {settings.PAPERS_DIR}")
