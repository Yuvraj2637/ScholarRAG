"""Citation-grounded answer generation engine with multi-provider support."""

import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from src.config import settings

class Citation(BaseModel):
    chunk_id: str
    paper_id: str
    paper_title: str
    section_title: str
    year: int
    page_start: int = 1
    page_end: int = 1
    quote_snippet: str = ""

class GenerationResult(BaseModel):
    query: str
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    model_name: str = ""
    provider: str = ""

SYSTEM_CITATION_PROMPT = """You are an expert AI research assistant specializing in efficient machine learning, attention mechanisms, and Transformer architectures.

Your task is to answer the user's research query strictly based on the provided retrieved research paper chunks.

CRITICAL CITATION RULES:
1. Every factual statement, equation, metric, or architectural detail MUST include an explicit inline citation referencing the paper title, section, and page number, for example: [FlashAttention, Section 3, p. 4] or [LoRA: Low-Rank Adaptation, Section 2.1, p. 3].
2. Do NOT extrapolate or hallucinate facts that cannot be substantiated by the context.
3. If the provided context does not contain sufficient details to answer part of the question, state clearly what is known from the context and what information is missing.
4. Maintain academic rigor, precision, and clarity.
"""

def format_context_blocks(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into numbered context blocks for LLM prompt."""
    blocks = []
    for idx, c in enumerate(chunks, 1):
        p_str = f"Page {c.get('page_start', 1)}" if c.get('page_start') == c.get('page_end') else f"Pages {c.get('page_start', 1)}-{c.get('page_end', 1)}"
        block = (
            f"--- [CHUNK {idx}] ---\n"
            f"Paper Title: {c.get('paper_title', 'Unknown')}\n"
            f"Paper ID: {c.get('paper_id', 'Unknown')}\n"
            f"Section: {c.get('section_title', 'Unknown')}\n"
            f"Page: {p_str}\n"
            f"Year: {c.get('year', '')}\n"
            f"Content:\n{c.get('text', '').strip()}\n"
        )
        blocks.append(block)
    return "\n".join(blocks)

class ResearchGenerator:
    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        self.gemini_key = settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
        self.openai_key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")

    def _generate_gemini(self, prompt: str, system_prompt: str) -> str:
        """Call Google GenAI API."""
        from google import genai
        client = genai.Client(api_key=self.gemini_key)
        response = client.models.generate_content(
            model=settings.GENERATION_MODEL,
            contents=prompt,
            config={"system_instruction": system_prompt}
        )
        return response.text or ""

    def _generate_openai(self, prompt: str, system_prompt: str) -> str:
        """Call OpenAI API."""
        from openai import OpenAI
        client = OpenAI(api_key=self.openai_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content or ""

    def _generate_mock(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Generate high-quality grounded mock response for offline testing."""
        if not chunks:
            return "No relevant context found in the corpus to answer this question."

        top_chunk = chunks[0]
        title = top_chunk.get("paper_title", "Research Paper")
        sec = top_chunk.get("section_title", "Method")

        answer_lines = [
            f"Based on [{title}, {sec}], the analysis highlights the following key points:",
            f"\n1. **Core Mechanism**: {top_chunk.get('text', '')[:280]}... [{title}, {sec}].",
        ]
        if len(chunks) > 1:
            sec_chunk = chunks[1]
            sec_title = sec_chunk.get("paper_title", title)
            sec_sec = sec_chunk.get("section_title", "Results")
            answer_lines.append(
                f"\n2. **Empirical Impact & Architecture**: {sec_chunk.get('text', '')[:240]}... [{sec_title}, {sec_sec}]."
            )
        return "\n".join(answer_lines)

    def generate(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        custom_prompt: Optional[str] = None,
        custom_system_prompt: Optional[str] = None
    ) -> GenerationResult:
        """Generate citation-grounded answer over retrieved chunks."""
        context_str = format_context_blocks(retrieved_chunks)
        sys_prompt = custom_system_prompt or SYSTEM_CITATION_PROMPT

        if custom_prompt:
            user_prompt = custom_prompt
        else:
            user_prompt = f"USER QUERY:\n{query}\n\nRETRIEVED CONTEXT PASSAGES:\n{context_str}\n\nPlease provide a clear, comprehensive, citation-grounded response."

        # Extract citations metadata
        citations = [
            Citation(
                chunk_id=c.get("chunk_id", ""),
                paper_id=c.get("paper_id", ""),
                paper_title=c.get("paper_title", ""),
                section_title=c.get("section_title", ""),
                year=c.get("year", 0),
                page_start=c.get("page_start", 1),
                page_end=c.get("page_end", 1),
                quote_snippet=c.get("text", "")[:120] + "..."
            )
            for c in retrieved_chunks
        ]

        # Determine generation method
        if self.gemini_key and self.provider == "gemini":
            try:
                answer = self._generate_gemini(user_prompt, sys_prompt)
                used_provider = "gemini"
                used_model = settings.GENERATION_MODEL
            except Exception as e:
                print(f"[WARN] Gemini generation failed: {e}. Falling back to grounded synthesis.")
                answer = self._generate_mock(query, retrieved_chunks)
                used_provider = "mock"
                used_model = "local-grounded-synthesizer"
        elif self.openai_key and self.provider == "openai":
            try:
                answer = self._generate_openai(user_prompt, sys_prompt)
                used_provider = "openai"
                used_model = "gpt-4o-mini"
            except Exception as e:
                print(f"[WARN] OpenAI generation failed: {e}. Falling back to grounded synthesis.")
                answer = self._generate_mock(query, retrieved_chunks)
                used_provider = "mock"
                used_model = "local-grounded-synthesizer"
        else:
            answer = self._generate_mock(query, retrieved_chunks)
            used_provider = "mock"
            used_model = "local-grounded-synthesizer"

        return GenerationResult(
            query=query,
            answer=answer,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            model_name=used_model,
            provider=used_provider
        )
