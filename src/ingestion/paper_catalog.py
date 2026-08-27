"""Curated catalog of seminal research papers on Efficient Transformers and LLM Optimization."""

from typing import Dict, List
from pydantic import BaseModel

class PaperMetadata(BaseModel):
    paper_id: str
    arxiv_id: str
    title: str
    authors: List[str]
    year: int
    primary_topic: str
    summary: str

CURATED_PAPERS: List[PaperMetadata] = [
    PaperMetadata(
        paper_id="flashattention_v1",
        arxiv_id="2205.14135",
        title="FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
        authors=["Tri Dao", "Daniel Y. Fu", "Stefano Ermon", "Atri Rudra", "Christopher Re"],
        year=2022,
        primary_topic="Attention Optimization",
        summary="Introduces IO-aware exact attention tiling into SRAM blocks to minimize GPU HBM memory accesses."
    ),
    PaperMetadata(
        paper_id="flashattention_v2",
        arxiv_id="2307.08691",
        title="FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning",
        authors=["Tri Dao"],
        year=2023,
        primary_topic="Attention Optimization",
        summary="Optimizes FlashAttention with improved thread block work partitioning, fewer non-matmul FLOPs, and sequence parallelism."
    ),
    PaperMetadata(
        paper_id="lora",
        arxiv_id="2106.09685",
        title="LoRA: Low-Rank Adaptation of Large Language Models",
        authors=["Edward J. Hu", "Yelong Shen", "Phillip Wallis", "Zeyuan Allen-Zhu", "Yuanzhi Li", "Shean Wang", "Lu Wang", "Weizhu Chen"],
        year=2021,
        primary_topic="PEFT",
        summary="Freezes pre-trained model weights and injects trainable rank decomposition matrices into Transformer layers."
    ),
    PaperMetadata(
        paper_id="qlora",
        arxiv_id="2305.14314",
        title="QLoRA: Efficient Finetuning of Quantized LLMs",
        authors=["Tim Dettmers", "Artidoro Pagnoni", "Ari Holtzman", "Luke Zettlemoyer"],
        year=2023,
        primary_topic="PEFT & Quantization",
        summary="Enables 4-bit quantized base models fine-tuned with LoRA using NF4 data type and Double Quantization."
    ),
    PaperMetadata(
        paper_id="vllm_pagedattention",
        arxiv_id="2309.06180",
        title="Efficient Memory Management for Large Language Model Serving with PagedAttention",
        authors=["Woosuk Kwon", "Zhuohan Li", "Siyuan Zhuang", "Ying Sheng", "Lianmin Zheng", "Cody Hao Yu", "Joseph E. Gonzalez", "Hao Zhang", "Ion Stoica"],
        year=2023,
        primary_topic="Serving & KV Cache",
        summary="Introduces PagedAttention to manage KV cache in non-contiguous virtual memory blocks, eliminating internal fragmentation."
    ),
    PaperMetadata(
        paper_id="gptq",
        arxiv_id="2210.17323",
        title="GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers",
        authors=["Elias Frantar", "Saleh Ashkboos", "Torsten Hoefler", "Dan Alistarh"],
        year=2022,
        primary_topic="Quantization",
        summary="One-shot post-training quantization based on Approximate Second-Order Information for 3/4-bit weights."
    ),
    PaperMetadata(
        paper_id="awq",
        arxiv_id="2306.00978",
        title="AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration",
        authors=["Ji Lin", "Jiaming Tang", "Haotian Tang", "Shang Yang", "Xingyu Dang", "Song Han"],
        year=2023,
        primary_topic="Quantization",
        summary="Protects salient 1% weights based on activation magnitudes to perform robust low-bit weight-only quantization."
    ),
    PaperMetadata(
        paper_id="smoothquant",
        arxiv_id="2211.10438",
        title="SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models",
        authors=["Guangxuan Xiao", "Ji Lin", "Mickael Seznec", "Hao Wu", "Julien Demouth", "Song Han"],
        year=2022,
        primary_topic="Quantization",
        summary="Mathematically migrates quantization difficulty from activations to weights via per-channel scaling factors."
    ),
    PaperMetadata(
        paper_id="speculative_decoding",
        arxiv_id="2211.17192",
        title="Fast Inference from Transformers via Speculative Decoding",
        authors=["Charlie Chen", "Sebastian Borgeaud", "Alireza Ghaffarkhah", "Samuel L. Smith", "Laurent Sifre"],
        year=2022,
        primary_topic="Speculative Decoding",
        summary="Accelerates autoregressive inference by using a small draft model and parallel validation by target LLM."
    ),
    PaperMetadata(
        paper_id="distilbert",
        arxiv_id="1910.01108",
        title="DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter",
        authors=["Victor Sanh", "Lysandre Debut", "Julien Chaumond", "Thomas Wolf"],
        year=2019,
        primary_topic="Knowledge Distillation",
        summary="Triple-loss knowledge distillation retaining 97% of BERT language understanding with 40% size reduction."
    ),
    PaperMetadata(
        paper_id="dora",
        arxiv_id="2402.09353",
        title="DoRA: Weight-Decomposed Low-Rank Adaptation",
        authors=["Shih-Yang Liu", "Chien-Yi Wang", "Hongxu Yin", "Pavlo Molchanov", "Yu-Chiang Frank Wang", "Kwang-Ting Cheng", "Min-Hung Chen"],
        year=2024,
        primary_topic="PEFT",
        summary="Decomposes model weights into magnitude and directional components, applying LoRA to directional updates."
    ),
    PaperMetadata(
        paper_id="flashdecoding",
        arxiv_id="2312.04927",
        title="Flash-Decoding for Long-Context LLM Inference",
        authors=["Tri Dao", "Daniel Haziza", "Francisco Massa", "Grigory Sizov"],
        year=2023,
        primary_topic="Attention & Serving",
        summary="Parallelizes attention decoding across the sequence/KV dimension instead of only batch/heads to saturate GPU compute."
    ),
    PaperMetadata(
        paper_id="streaming_llm",
        arxiv_id="2309.17453",
        title="Efficient Streaming Language Models with Attention Sinks",
        authors=["Guangxuan Xiao", "Yuandong Tian", "Beidi Chen", "Song Han", "Mike Lewis"],
        year=2023,
        primary_topic="KV Cache & Long Context",
        summary="Discovers attention sinks in initial tokens and preserves initial KV tokens plus rolling window for infinite streaming."
    ),
    PaperMetadata(
        paper_id="sglang_radix",
        arxiv_id="2312.07104",
        title="SGLang: Efficient Execution of Structured Language Model Programs",
        authors=["Lianmin Zheng", "Liangsheng Yin", "Zhiqiang Xie", "Jeff Huang", "Chuyue Sun", "Cody Hao Yu", "Shiyi Cao", "Christos Kozyrakis", "Ion Stoica", "Joseph E. Gonzalez", "Sheng Shen"],
        year=2023,
        primary_topic="Serving & Cache Reuse",
        summary="Proposes RadixAttention to maintain LRU radix tree of KV caches for automatic multi-turn and prefix cache reuse."
    ),
    PaperMetadata(
        paper_id="medusa",
        arxiv_id="2401.10774",
        title="Medusa: Simple LLM Inference Acceleration with Multiple Decoding Heads",
        authors=["Tianle Cai", "Yuhong Li", "Zhengyang Geng", "Hongwu Peng", "Jason D. Lee", "Deming Chen", "Tri Dao"],
        year=2024,
        primary_topic="Speculative Decoding",
        summary="Augments LLMs with multiple lightweight decoding heads predicting multiple future tokens simultaneously with tree attention."
    ),
    PaperMetadata(
        paper_id="deepspeed_zero",
        arxiv_id="1910.02054",
        title="ZeRO: Memory Optimizations Toward Training Trillion Parameter Models",
        authors=["Samyam Rajbhandari", "Jeff Rasley", "Olatunji Ruwase", "Yuxiong He"],
        year=2019,
        primary_topic="Distributed Training",
        summary="Partitions optimizer states, gradients, and model parameters across data-parallel processes to eliminate memory redundancy."
    ),
    PaperMetadata(
        paper_id="chinchilla_scaling",
        arxiv_id="2203.15556",
        title="Training Compute-Optimal Large Language Models",
        authors=["Jordan Hoffmann", "Sebastian Borgeaud", "Arthur Mensch", "Elena Buchatskaya", "Trevor Cai", "Eliza Rutherford", "Diego de Las Casas", "Lisa Anne Hendricks", "Johannes Welbl", "Aidan Clark", "Tom Hennigan", "Eric Noland", "Katie Millican", "George van den Driessche", "Bogdan Damoc", "Aurelia Guy", "Simon Osindero", "Karen Simonyan", "Erich Elsen", "Jack W. Rae", "Oriol Vinyals", "Laurent Sifre"],
        year=2022,
        primary_topic="Scaling Laws",
        summary="Derives compute-optimal scaling laws showing model size and training tokens should be scaled in equal proportions."
    ),
    PaperMetadata(
        paper_id="mqa_gqa",
        arxiv_id="2305.13245",
        title="GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints",
        authors=["Joshua Ainslie", "James Lee-Thorp", "Michiel de Jong", "Yury Zemlyanskiy", "Federico Lebron", "Sumit Sanghai"],
        year=2023,
        primary_topic="Attention Optimization",
        summary="Interpolates between Multi-Head Attention and Multi-Query Attention using grouped key-value heads."
    ),
    PaperMetadata(
        paper_id="retnet",
        arxiv_id="2307.08621",
        title="Retentive Network: A Successor to the Transformer for Large Language Models",
        authors=["Yutao Sun", "Li Dong", "Shaohan Huang", "Shuming Ma", "Yuqing Xia", "Jilong Xue", "Jianyong Wang", "Furu Wei"],
        year=2023,
        primary_topic="Linear Architecture",
        summary="Introduces multi-scale retention mechanism supporting parallel training, recurrent $O(1)$ inference, and chunkwise representations."
    ),
    PaperMetadata(
        paper_id="mamba",
        arxiv_id="2312.00752",
        title="Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
        authors=["Albert Gu", "Tri Dao"],
        year=2023,
        primary_topic="State Space Models",
        summary="Presents selective state space models (SSM) with hardware-aware parallel scan achieving linear time inference and high throughput."
    ),
    PaperMetadata(
        paper_id="llama_paper",
        arxiv_id="2302.13971",
        title="LLaMA: Open and Efficient Foundation Language Models",
        authors=["Hugo Touvron", "Thibaut Lavril", "Gautier Izacard", "Xavier Martinet", "Marie-Anne Lachaux", "Timothee Lacroix", "Baptiste Roziere", "Naman Goyal", "Eric Hambro", "Faisal Azhar", "Aurelien Rodriguez", "Armand Joulin", "Edouard Grave", "Guillaume Lample"],
        year=2023,
        primary_topic="Model Architecture",
        summary="Details the architectural design of LLaMA (RMSNorm, SwiGLU activations, RoPE rotary positional embeddings)."
    ),
    PaperMetadata(
        paper_id="llama2_paper",
        arxiv_id="2307.09288",
        title="Llama 2: Open Foundation and Fine-Tuned Chat Models",
        authors=["Hugo Touvron", "Louis Martin", "Kevin Stone", "Peter Albert", "Amjad Almahairi", "Yasmine Babaei", "Nikolay Bashlykov", "Soumya Batra", "Prajjwal Bhargava", "Shruti Bhosale", "Dan Bikel", "Lukas Blecher", "Cristian Canton Ferrer", "Moya Chen", "Guillem Cucurull", "David Esiobu", "Jude Fernandes", "Jeremy Fu", "Wenyin Fu", "Brian Fuller", "Cynthia Gao", "Vedanuj Goswami", "Naman Goyal", "Anthony Hartshorn", "Saghar Hosseini", "Rui Hou", "Hakan Inan", "Marcin Kardas", "Viktor Kerkez", "Madian Khabsa", "Isabel Kloumann", "Artem Korenev", "Punit Singh Koura", "Marie-Anne Lachaux", "Thibaut Lavril", "Jenya Lee", "Diana Liskovich", "Yinghai Lu", "Yuning Mao", "Xavier Martinet", "Todor Mihaylov", "Pushkar Mishra", "Igor Molybog", "Yixin Nie", "Andrew Poulton", "Jeremy Reizenstein", "Rashi Rungta", "Kalyan Saladi", "Alan Schelten", "Ruan Silva", "Eric Michael Smith", "Ranjan Subramanian", "Xiaoqing Ellen Tan", "Binh Tang", "Ross Taylor", "Adina Williams", "Jian Xiang Kuan", "Puxin Xu", "Zheng Yan", "Iliyan Zarov", "Yuchen Zhang", "Angela Fan", "Melanie Kambadur", "Sharan Narang", "Aurelien Rodriguez", "Robert Stojnic", "Sergey Edunov", "Thomas Scialom"],
        year=2023,
        primary_topic="Model Architecture & Alignment",
        summary="Scaled open model training with GQA, context window expansion, and iterative RLHF alignment."
    ),
    PaperMetadata(
        paper_id="kv_quant_kivi",
        arxiv_id="2402.02750",
        title="KIVI: A Tuning-Free Asymmetric 2-bit Quantization for KV Cache",
        authors=["Zirui Liu", "Jiayi Yuan", "Hongye Jin", "Shaochen Zhong", "Zhaozhuo Xu", "Vladimir Braverman", "Beidi Chen", "Xia Hu"],
        year=2024,
        primary_topic="KV Cache Quantization",
        summary="Quantizes key cache per-channel and value cache per-token to 2-bit with minimal accuracy degradation."
    ),
    PaperMetadata(
        paper_id="sparse_attention_longformer",
        arxiv_id="2004.05150",
        title="Longformer: The Long-Document Transformer",
        authors=["Iz Beltagy", "Matthew E. Peters", "Arman Cohan"],
        year=2020,
        primary_topic="Sparse Attention",
        summary="Combines local windowed attention with task-specific global attention for linear-time context scaling."
    ),
    PaperMetadata(
        paper_id="flash_attention_3",
        arxiv_id="2407.08608",
        title="FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision",
        authors=["Tri Dao", "Jay Shah", "Makarand Parmar", "Gautam Kamath", "Aakanksha Chowdhery", "Christopher Re"],
        year=2024,
        primary_topic="Attention Optimization",
        summary="Exploits Hopper Tensor Core asynchrony (TMA), warpgroup GEMMs, and FP8 low-precision accumulation."
    ),
    PaperMetadata(
        paper_id="spqr_quantization",
        arxiv_id="2306.03078",
        title="SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression",
        authors=["Tim Dettmers", "Ruslan Svirschevski", "Vage Egiazarian", "Denis Kuznitskiy", "Alexander Borzunov", "Torsten Hoefler", "Dan Alistarh"],
        year=2023,
        primary_topic="Quantization",
        summary="Isolates sensitive outlier weights in higher precision sparse matrix while compressing remainder in 3-bit."
    ),
    PaperMetadata(
        paper_id="kv_cache_compression_h2o",
        arxiv_id="2306.14048",
        title="H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models",
        authors=["Zhenyu Zhang", "Ying Sheng", "Tianyi Zhou", "Tianlong Chen", "Lianmin Zheng", "Ruisi Cai", "Zhao Song", "Yuandong Tian", "Christopher Re", "Clark Barrett", "Zhangyang Wang", "Beidi Chen"],
        year=2023,
        primary_topic="KV Cache Compression",
        summary="Identifies Heavy Hitter (H2) tokens that dominate attention scores and dynamically evicts non-critical KV states."
    ),
    PaperMetadata(
        paper_id="speculative_eagle",
        arxiv_id="2401.15077",
        title="EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty",
        authors=["Yuhui Li", "Fangyun Wei", "Chao Zhang", "Hongyang Zhang"],
        year=2024,
        primary_topic="Speculative Decoding",
        summary="Performs speculative sampling at top-layer feature level rather than token sequence level to remove uncertainty."
    ),
    PaperMetadata(
        paper_id="unsloth_lora",
        arxiv_id="2310.03013",
        title="Fast and Memory-Efficient Finetuning with Manual Backprop and Triton Kernels",
        authors=["Daniel Han", "Michael Han"],
        year=2023,
        primary_topic="PEFT & Kernels",
        summary="Rewrites backpropagation passes in OpenAI Triton to optimize LoRA and cross-entropy memory overhead."
    ),
    PaperMetadata(
        paper_id="quant_int8_llm",
        arxiv_id="2208.07339",
        title="LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale",
        authors=["Tim Dettmers", "Mike Lewis", "Younes Belkada", "Luke Zettlemoyer"],
        year=2022,
        primary_topic="Quantization",
        summary="Vector-wise quantization with mixed-precision decomposition for outlier activation features in emergent LLMs."
    )
]

PAPER_CATALOG_MAP: Dict[str, PaperMetadata] = {p.paper_id: p for p in CURATED_PAPERS}
