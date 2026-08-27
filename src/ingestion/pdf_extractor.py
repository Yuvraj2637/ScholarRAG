"""Structure-aware PDF extractor using PyMuPDF (fitz)."""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import pymupdf as fitz
from pydantic import BaseModel

class ParsedSection(BaseModel):
    section_title: str
    section_number: Optional[str] = None
    text: str
    paragraphs: List[str]
    page_start: int = 1
    page_end: int = 1

class ParsedDocument(BaseModel):
    paper_id: str
    title: str
    authors: List[str]
    year: int
    abstract: str
    sections: List[ParsedSection]
    total_pages: int
    abstract_page: int = 1

# Regex patterns for standard academic paper sections
SECTION_HEADER_REGEX = re.compile(
    r"^(?:(?:[0-9]{1,2}(?:\.[0-9]{1,2})*|[I|V|X]+)\.?\s+)?"
    r"(Abstract|Introduction|Background|Related\s+Work|Method(?:ology)?|Approach|Architecture|"
    r"Algorithm|Implementation|Experiments?(?:\s+and\s+Results)?|Results|Discussion|Limitations|"
    r"Conclusion|Broader\s+Impacts?|Ablation\s+Study|System\s+Design|Evaluation)"
    r"(?::|\b|\s.*)?$",
    re.IGNORECASE
)

PAGE_HEADER_FOOTER_PATTERNS = [
    re.compile(r"^arXiv:\d{4}\.\d{4,5}(?:v\d+)?\s+\[.*\]\s+\d+\s+[A-Za-z]+\s+\d{4}$", re.I),
    re.compile(r"^Preprint\. Under review\.", re.I),
    re.compile(r"^Published as a conference paper at.*", re.I),
    re.compile(r"^\d+\s*$", re.I),  # Isolated page number
]

class PDFStructureExtractor:
    def __init__(self, filter_references: bool = True):
        self.filter_references = filter_references

    def is_noise_line(self, line: str, page_num: int, total_pages: int, y_coord: float, page_height: float) -> bool:
        """Filter out running headers, footers, arxiv banners, and page numbers."""
        line_clean = line.strip()
        if not line_clean:
            return True
        # Top 5% or bottom 5% page coordinate check for isolated numbers/headers
        if (y_coord < 0.05 * page_height or y_coord > 0.95 * page_height) and len(line_clean) < 40:
            if line_clean.isdigit() or "arxiv" in line_clean.lower() or "page" in line_clean.lower():
                return True

        for pat in PAGE_HEADER_FOOTER_PATTERNS:
            if pat.match(line_clean):
                return True
        return False

    def extract_blocks_with_fonts(self, doc: fitz.Document) -> List[Dict[str, Any]]:
        """Extract text blocks with font sizes, flags, and bounding boxes."""
        raw_blocks = []
        for page_idx, page in enumerate(doc):
            page_dict = page.get_text("dict")
            page_height = page.rect.height
            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:  # Text blocks only
                    continue
                for line in block.get("lines", []):
                    line_text = ""
                    max_font_size = 0.0
                    is_bold = False
                    y_coord = line["bbox"][1]

                    for span in line.get("spans", []):
                        text = span.get("text", "")
                        line_text += text
                        max_font_size = max(max_font_size, span.get("size", 0.0))
                        flags = span.get("flags", 0)
                        if flags & 2 or "bold" in span.get("font", "").lower():
                            is_bold = True

                    line_text = line_text.strip()
                    if self.is_noise_line(line_text, page_idx, len(doc), y_coord, page_height):
                        continue

                    raw_blocks.append({
                        "page": page_idx + 1,
                        "text": line_text,
                        "font_size": max_font_size,
                        "is_bold": is_bold,
                        "bbox": line["bbox"]
                    })
        return raw_blocks

    def parse_pdf(self, pdf_path: Path, paper_id: str, title: str, authors: List[str], year: int) -> ParsedDocument:
        """Parse a PDF file into structured sections, abstract, and metadata."""
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        blocks = self.extract_blocks_with_fonts(doc)
        doc.close()

        # Group lines into sections based on font size / header regex
        # Baseline body font size is usually the mode / median font size
        font_sizes = [b["font_size"] for b in blocks if b["font_size"] > 0]
        body_font_size = max(set(font_sizes), key=font_sizes.count) if font_sizes else 10.0

        sections: List[ParsedSection] = []
        current_section_title = "Introduction"
        current_section_number = "1"
        current_paragraphs: List[str] = []
        current_buffer: List[str] = []
        section_start_page = 1
        section_end_page = 1
        abstract_text = ""
        in_abstract = False
        in_references = False

        for b in blocks:
            text = b["text"]
            font_sz = b["font_size"]
            is_bold = b["is_bold"]
            page_num = b["page"]

            # Detect abstract
            if text.lower() == "abstract" or text.lower().startswith("abstract:"):
                in_abstract = True
                continue

            # Detect section heading
            is_heading_size = (font_sz >= body_font_size + 1.2) or (is_bold and font_sz >= body_font_size)
            match = SECTION_HEADER_REGEX.match(text)

            if is_heading_size and (match or (len(text) < 60 and is_bold and not text.endswith("."))):
                # Flush previous section
                if current_buffer:
                    para = " ".join(current_buffer).strip()
                    if para:
                        current_paragraphs.append(para)
                    current_buffer = []

                if current_paragraphs:
                    sec_text = "\n\n".join(current_paragraphs)
                    if not in_abstract and not in_references:
                        sections.append(ParsedSection(
                            section_title=current_section_title,
                            section_number=current_section_number,
                            text=sec_text,
                            paragraphs=current_paragraphs,
                            page_start=section_start_page,
                            page_end=section_end_page
                        ))

                new_title = text.strip()
                if "reference" in new_title.lower() or "bibliography" in new_title.lower():
                    if self.filter_references:
                        in_references = True
                        break

                in_abstract = False
                current_section_title = new_title
                current_paragraphs = []
                section_start_page = page_num
                section_end_page = page_num
                continue

            if in_references:
                continue

            section_end_page = page_num

            if in_abstract:
                if is_heading_size and text.lower() != "abstract":
                    in_abstract = False
                    current_section_title = text.strip()
                    section_start_page = page_num
                else:
                    abstract_text += " " + text
                continue

            # Accumulate text into paragraphs
            if not text.endswith("-"):
                current_buffer.append(text)
            else:
                current_buffer.append(text[:-1])

            if len(current_buffer) >= 4 and text.endswith((".", ":", "?", "!")):
                para = " ".join(current_buffer).strip()
                if para:
                    current_paragraphs.append(para)
                current_buffer = []

        if current_buffer:
            para = " ".join(current_buffer).strip()
            if para:
                current_paragraphs.append(para)

        if current_paragraphs and not in_references:
            sections.append(ParsedSection(
                section_title=current_section_title,
                section_number=current_section_number,
                text="\n\n".join(current_paragraphs),
                paragraphs=current_paragraphs,
                page_start=section_start_page,
                page_end=section_end_page
            ))

        if not sections and blocks:
            full_text = " ".join([b["text"] for b in blocks])
            sections.append(ParsedSection(
                section_title="Main Body",
                section_number="1",
                text=full_text,
                paragraphs=[full_text],
                page_start=1,
                page_end=total_pages
            ))

        return ParsedDocument(
            paper_id=paper_id,
            title=title,
            authors=authors,
            year=year,
            abstract=abstract_text.strip(),
            sections=sections,
            total_pages=total_pages,
            abstract_page=1
        )
