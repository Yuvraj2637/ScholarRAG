"""Page-linked starter evidence for the standalone ScholarRAG demo."""

from dataclasses import dataclass

from src.ingestion.chunker import Chunk
from src.ingestion.paper_catalog import PAPER_CATALOG_MAP
from src.retrieval.bm25 import BM25Retriever


@dataclass(frozen=True)
class Note:
    facet: str
    section: str
    page: int
    text: str


@dataclass(frozen=True)
class Paper:
    label: str
    version: int
    notes: tuple[Note, ...]

    def pdf_url(self, paper_id: str, page: int = 1) -> str:
        arxiv_id = PAPER_CATALOG_MAP[paper_id].arxiv_id
        return f"https://arxiv.org/pdf/{arxiv_id}v{self.version}#page={page}"


# These are original paraphrases, with pages pinned to the cited arXiv versions.
PAPERS = {
    "flashattention_v1": Paper("FlashAttention", 2, (
        Note("Problem", "Abstract", 1, "Standard self-attention writes large intermediate matrices to GPU high-bandwidth memory (HBM). Moving these intermediates between HBM and on-chip SRAM is expensive for long sequences."),
        Note("Method", "§3.1 · FlashAttention", 5, "FlashAttention tiles queries, keys, and values so a block fits in on-chip SRAM. It updates running softmax statistics and partial outputs across tiles, avoiding storage of the full attention matrix in HBM."),
        Note("Result", "§3.1 · Theorem 1", 5, "The tiled algorithm computes exact softmax attention. It still does quadratic arithmetic in sequence length, while the extra storage beyond the inputs and outputs grows linearly."),
    )),
    "flashattention_v2": Paper("FlashAttention-2", 1, (
        Note("Problem", "Abstract", 1, "The first FlashAttention implementation leaves GPU compute underused when thread blocks and warps receive an uneven amount of work or have to exchange data through shared memory."),
        Note("Method", "Abstract", 1, "FlashAttention-2 reduces operations outside matrix multiplication, splits work for a single attention head across thread blocks, and partitions work among warps to reduce shared-memory communication."),
        Note("Result", "Abstract", 1, "On the authors’ A100 tests, FlashAttention-2 was about twice as fast as FlashAttention and reached 50–73% of theoretical peak throughput. The figures describe the paper’s hardware and experiments."),
    )),
    "lora": Paper("LoRA", 2, (
        Note("Problem", "Abstract", 1, "Fully fine-tuning a large language model updates all its parameters. Keeping a separate large checkpoint for every task increases storage and training costs."),
        Note("Method", "Abstract", 1, "LoRA, or Low-Rank Adaptation, freezes pretrained weights and trains low-rank matrices that represent a task-specific weight update for selected Transformer layers."),
        Note("Result", "Abstract", 1, "The paper reports far fewer trainable parameters and lower GPU memory use than full fine-tuning on GPT-3, with no added inference latency."),
    )),
    "qlora": Paper("QLoRA", 1, (
        Note("Problem", "Abstract", 1, "QLoRA targets the memory needed to fine-tune large language models. Keeping their frozen base weights in higher precision can prevent the model from fitting on a single GPU."),
        Note("Method", "Abstract", 1, "QLoRA trains LoRA adapters through a frozen 4-bit base model. NormalFloat4 (NF4) is designed for normally distributed weights, while double quantization further compresses the quantization constants."),
        Note("Result", "Abstract", 1, "The authors report fine-tuning a 65-billion-parameter model on a single 48 GB GPU. They also use paged optimizers to manage memory spikes."),
    )),
    "vllm_pagedattention": Paper("PagedAttention / vLLM", 1, (
        Note("Problem", "§1 · Introduction", 2, "A serving system keeps a key-value (KV) cache for each sequence as it generates tokens. Allocating large contiguous regions wastes memory through fragmentation and redundant copies."),
        Note("Method", "§4.2 · KV Cache Manager", 6, "PagedAttention divides each sequence’s KV cache into fixed-size blocks. A block table maps logical blocks to physical GPU memory, so vLLM can allocate extra space on demand without requiring neighboring blocks."),
        Note("Result", "Abstract", 1, "Smaller KV allocations and cache sharing leave room to batch more requests. The paper reports 2–4× throughput over the evaluated serving baselines at comparable latency."),
    )),
    "mamba": Paper("Mamba", 2, (
        Note("Problem", "Abstract", 1, "Earlier structured state-space models had difficulty deciding which information to retain based on the current input token. This limited their ability to reason about content."),
        Note("Method", "Abstract", 1, "Mamba makes state-space parameters depend on the input, so the recurrent state can selectively retain or discard information. Because these parameters vary by token, the model uses a parallel scan suited to GPU hardware."),
        Note("Result", "Abstract", 1, "The paper reports scaling linear in sequence length and higher inference throughput than the Transformers it evaluated."),
    )),
}

EXAMPLES = {
    "FlashAttention memory": "How does FlashAttention avoid storing the full attention matrix in HBM?",
    "QLoRA and NF4": "What are NF4 and double quantization in QLoRA?",
    "vLLM’s KV cache": "How does PagedAttention manage KV cache fragmentation in vLLM?",
}

STOP_WORDS = frozenset(
    "a about an and are as at be by can could did do does explain for from give had has have how i in is it its "
    "me of on or please s so should tell than that the their this to use using was were what when where which who why "
    "will with work would you your".split()
)


class StarterCorpus:
    def __init__(self):
        self.retriever = BM25Retriever()
        self.notes = {}
        chunks = []
        for paper_id, paper in PAPERS.items():
            metadata = PAPER_CATALOG_MAP[paper_id]
            for index, note in enumerate(paper.notes):
                chunk_id = f"{paper_id}-demo-{index}"
                self.notes[chunk_id] = note
                search_text = f"{paper.label} {note.text}"
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    paper_id=paper_id,
                    paper_title=metadata.title,
                    authors=metadata.authors,
                    year=metadata.year,
                    primary_topic=metadata.primary_topic,
                    section_title=note.section,
                    section_index=index,
                    chunk_index_in_section=0,
                    page_start=note.page,
                    page_end=note.page,
                    text=search_text,
                    token_count=len(search_text.split()),
                ))
        self.retriever.index_chunks(chunks)

    def answer(self, question: str) -> list[tuple[str, Note]]:
        terms = set(self.retriever.tokenize(question)) - STOP_WORDS
        if not terms:
            return []

        # BM25Plus contributes a baseline even without a match in an individual chunk.
        hits = self.retriever.search(" ".join(sorted(terms)), top_k=len(self.notes))
        ranked = [
            hit for hit in hits
            if len(terms.intersection(self.retriever.tokenize(hit["text"]))) >= min(2, len(terms))
        ]
        if not ranked:
            return []

        primary = ranked[0]["paper_id"]
        return [(primary, self.notes[hit["chunk_id"]]) for hit in ranked if hit["paper_id"] == primary][:2]
