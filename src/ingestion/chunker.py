"""Structure-aware chunker with metadata enrichment for academic papers."""

import re
from typing import List
from pydantic import BaseModel, Field
from src.config import settings
from src.ingestion.pdf_extractor import ParsedDocument, ParsedSection

class Chunk(BaseModel):
    chunk_id: str
    paper_id: str
    paper_title: str
    authors: List[str]
    year: int
    primary_topic: str = ""
    section_title: str
    section_index: int
    chunk_index_in_section: int
    text: str
    token_count: int
    page_start: int = 1
    page_end: int = 1
    metadata: dict = Field(default_factory=dict)

class StructureAwareChunker:
    def __init__(
        self,
        max_tokens: int = settings.CHUNK_MAX_TOKENS,
        min_tokens: int = settings.CHUNK_MIN_TOKENS,
        overlap_tokens: int = settings.CHUNK_OVERLAP_TOKENS,
    ):
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_tokens = overlap_tokens
        # Sentence splitting pattern
        self.sentence_regex = re.compile(r"(?<=[.?!])\s+(?=[A-Z0-9])")

    def estimate_tokens(self, text: str) -> int:
        """Heuristic token estimation: ~0.75 words per token or len/4."""
        return max(1, len(text.split()))

    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences while preserving integrity."""
        text = re.sub(r"\s+", " ", text).strip()
        sentences = self.sentence_regex.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_section(
        self,
        section: ParsedSection,
        doc: ParsedDocument,
        sec_idx: int,
        primary_topic: str = ""
    ) -> List[Chunk]:
        """Chunk an individual section respecting sentence boundaries."""
        sentences = self.split_sentences(section.text)
        if not sentences:
            return []

        chunks: List[Chunk] = []
        current_sentences: List[str] = []
        current_tokens = 0
        chunk_sub_idx = 0

        for sent in sentences:
            sent_tokens = self.estimate_tokens(sent)

            if current_tokens + sent_tokens > self.max_tokens and current_sentences:
                chunk_text = " ".join(current_sentences)
                if current_tokens >= self.min_tokens:
                    c_id = f"{doc.paper_id}_s{sec_idx}_c{chunk_sub_idx}"
                    chunks.append(
                        Chunk(
                            chunk_id=c_id,
                            paper_id=doc.paper_id,
                            paper_title=doc.title,
                            authors=doc.authors,
                            year=doc.year,
                            primary_topic=primary_topic,
                            section_title=section.section_title,
                            section_index=sec_idx,
                            chunk_index_in_section=chunk_sub_idx,
                            text=chunk_text,
                            token_count=current_tokens,
                            page_start=section.page_start,
                            page_end=section.page_end,
                            metadata={
                                "paper_id": doc.paper_id,
                                "section_title": section.section_title,
                                "year": doc.year,
                                "page_start": section.page_start,
                                "page_end": section.page_end
                            }
                        )
                    )
                    chunk_sub_idx += 1

                overlap_accum: List[str] = []
                overlap_count = 0
                for prev_sent in reversed(current_sentences):
                    prev_toks = self.estimate_tokens(prev_sent)
                    if overlap_count + prev_toks <= self.overlap_tokens:
                        overlap_accum.insert(0, prev_sent)
                        overlap_count += prev_toks
                    else:
                        break

                current_sentences = overlap_accum + [sent]
                current_tokens = sum(self.estimate_tokens(s) for s in current_sentences)
            else:
                current_sentences.append(sent)
                current_tokens += sent_tokens

        # Flush final remaining sentences
        if current_sentences and current_tokens >= self.min_tokens:
            c_id = f"{doc.paper_id}_s{sec_idx}_c{chunk_sub_idx}"
            chunks.append(
                Chunk(
                    chunk_id=c_id,
                    paper_id=doc.paper_id,
                    paper_title=doc.title,
                    authors=doc.authors,
                    year=doc.year,
                    primary_topic=primary_topic,
                    section_title=section.section_title,
                    section_index=sec_idx,
                    chunk_index_in_section=chunk_sub_idx,
                    text=" ".join(current_sentences),
                    token_count=current_tokens,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    metadata={
                        "paper_id": doc.paper_id,
                        "section_title": section.section_title,
                        "year": doc.year,
                        "page_start": section.page_start,
                        "page_end": section.page_end
                    }
                )
            )

        return chunks

    def chunk_document(self, doc: ParsedDocument, primary_topic: str = "") -> List[Chunk]:
        """Convert a parsed document into structured chunks with page metadata."""
        all_chunks: List[Chunk] = []

        if doc.abstract and len(doc.abstract.split()) >= self.min_tokens:
            abstract_chunk = Chunk(
                chunk_id=f"{doc.paper_id}_s0_c0",
                paper_id=doc.paper_id,
                paper_title=doc.title,
                authors=doc.authors,
                year=doc.year,
                primary_topic=primary_topic,
                section_title="Abstract",
                section_index=0,
                chunk_index_in_section=0,
                text=doc.abstract,
                token_count=self.estimate_tokens(doc.abstract),
                page_start=doc.abstract_page,
                page_end=doc.abstract_page,
                metadata={
                    "paper_id": doc.paper_id,
                    "section_title": "Abstract",
                    "year": doc.year,
                    "page_start": doc.abstract_page,
                    "page_end": doc.abstract_page
                }
            )
            all_chunks.append(abstract_chunk)

        for sec_idx, section in enumerate(doc.sections, start=1):
            sec_chunks = self.chunk_section(section, doc, sec_idx, primary_topic)
            all_chunks.extend(sec_chunks)

        return all_chunks
