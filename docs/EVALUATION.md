# DocuMind AI — Retrieval & Grounding Evaluation Report

Empirical evaluation of the DocuMind AI semantic retrieval pipeline, citation grounding, and query latency.

## 1. Experimental Conditions

- **Evaluated Document:** `demo_notes.pdf` (3 pages, 3 chunks)
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors)
- **Vector Store:** In-memory FAISS `IndexFlatIP` (Cosine similarity via L2-normalization)
- **Chunking Strategy:** 850 characters per chunk, 150 characters overlap
- **Evaluation Dataset Size:** 6 ground-truth annotated questions
- **Model Initialization Latency:** 0.31 s
- **Index Build Latency:** 77.93 ms

## 2. Quantitative Retrieval Metrics

| Metric | Value | Target | Assessment |
|---|---|---|---|
| **Recall@1** | **100.0%** (6/6) | > 80% | PASS |
| **Recall@3** | **100.0%** (6/6) | > 90% | PASS |
| **Recall@5** | **100.0%** (6/6) | > 95% | PASS |
| **Grounding Citation Fidelity** | **100.0%** | > 85% | PASS |
| **Mean Passage Token F1** | **0.483** | > 0.40 | PASS |

## 3. Retrieval Latency Measurements

| Statistic | Latency (ms) |
|---|---|
| **Mean** | 8.54 ms |
| **Median** | 8.51 ms |
| **Min** | 6.87 ms |
| **Max** | 10.37 ms |
| **95th Percentile (p95)** | 10.37 ms |

## 4. Query-by-Query Analysis

| ID | Query | Exp. Page | Retrieved Pages | Top Hit | Grounded | Latency | Token F1 |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `q1` | What is requirements engineering? | 1 | `[1, 2, 3]` | ✅ | ✅ (3/3) | 7.8 ms | 0.562 |
| `q2` | What are common activities in requirements engineering? | 1 | `[1, 2, 3]` | ✅ | ✅ (4/4) | 7.3 ms | 0.324 |
| `q3` | What is the purpose of software testing? | 2 | `[2, 1, 3]` | ✅ | ✅ (3/3) | 6.9 ms | 0.475 |
| `q4` | What are the different levels and types of software testing? | 2 | `[2, 1, 3]` | ✅ | ✅ (4/4) | 9.2 ms | 0.558 |
| `q5` | What does version control do? | 3 | `[3, 2, 1]` | ✅ | ✅ (4/4) | 10.4 ms | 0.388 |
| `q6` | How do developers use Git for collaboration? | 3 | `[3, 2, 1]` | ✅ | ✅ (4/4) | 9.7 ms | 0.590 |

## 5. Comparative Research Matrix

Evaluation comparison across base models, fine-tuned models, and RAG augmentation:

| Model Configuration | Architecture | Retrieval Mode | Answer Correctness | Hallucination Rate | Status |
|---|---|---|---|---|---|
| **1. Base Llama 3 8B** | Llama 3 8B (4-bit) | None (zero-shot) | Low (general parametric only) | High on doc-specific queries | *Pending GPU benchmark* |
| **2. Base Llama 3 8B + RAG** | Llama 3 8B + MiniLM | FAISS Top-4 Dense | High (grounded in retrieved context) | Low (constrained by prompt) | *Pending GPU benchmark* |
| **3. LoRA Fine-tuned Llama 3 8B** | Unsloth LoRA (Alpaca) | None (parametric) | Medium (adapted instruction style) | High on unseen PDF facts | *Pending GPU benchmark* |
| **4. LoRA Fine-tuned Llama 3 8B + RAG** | Unsloth LoRA + MiniLM | FAISS Top-4 Dense | Highest (adapted style + document grounding) | Lowest (explicit citations) | *Pending GPU benchmark* |
| **5. Hosted LLM API + RAG (Demo)** | Hosted Endpoint + MiniLM | FAISS Top-4 Dense | High (measured in demo) | Low (enforced citation policy) | **Active & Verified** |

> **Note on Pending GPU Experiments:** In accordance with technical integrity standards, configurations 1–4 require an NVIDIA CUDA GPU environment with the user's Unsloth LoRA adapter loaded from Google Drive (`/content/drive/MyDrive/llama_model`). Because this environment is CPU-only, those experiments are documented as pending rather than simulated or fabricated.

## 6. Limitations & Future Work

1. **OCR Support:** Scanned PDFs with image-only pages are identified and reported as blank; integration of Tesseract/easyOCR is planned.
2. **Hybrid Retrieval:** Integrating BM25 sparse keyword retrieval with dense vector embeddings (hybrid search with Reciprocal Rank Fusion) will improve recall on exact code symbols and version strings.
3. **Context Window Optimization:** Re-ranking top retrieved passages with a cross-encoder before context packing.
