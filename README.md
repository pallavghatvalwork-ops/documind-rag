# DocuMind AI — RAG-Based PDF Question Answering System

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.67+-FF4B4B.svg)](https://streamlit.io/)
[![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS%20Dense-green.svg)](https://github.com/facebookresearch/faiss)
[![Sentence Transformers](https://img.shields.io/badge/Embeddings-all--MiniLM--L6--v2-orange.svg)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **DocuMind AI** is an end-to-end, reproducible Retrieval-Augmented Generation (RAG) system for question answering over multi-page PDF documents. It combines dense semantic retrieval (Sentence Transformers + FAISS Flat Inner-Product) with grounded prompt synthesis, citation attribution, and dual-backend execution (Hosted LLM API vs. local fine-tuned LoRA Llama 3 8B).

---

## 🏗️ Architecture Overview

```text
[User PDF Uploads]
        │
        ▼
[PyMuPDF (fitz)] ──► Page Extraction & Character Normalization
        │
        ▼
[Chunking Engine] ──► 850 chars/chunk (150 chars overlap) with Word Boundary Snapping
        │
        ▼
[Sentence Transformers] ──► 384-dim Dense Embeddings (L2-Normalized)
        │
        ▼
[In-Memory FAISS Index] ──► Inner-Product Similarity Search (IndexFlatIP)
        │
        ├─────────────────────────────────────────────────┐
        ▼                                                 ▼
[Query: Natural Language]                          [Top-k Passages]
        │                                                 │
        └─────────────────┬───────────────────────────────┘
                          │
                          ▼
             [Grounded Prompt Synthesizer]
            (Strict Citation Budget & Rules)
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
[Hosted LLM API (Groq/OpenAI)]   [Optional Local LoRA GPU]
(Cloud CPU-friendly demo)        (Colab Llama 3 8B Adapter)
            │                           │
            └─────────────┬─────────────┘
                          │
                          ▼
            [Streamlit Chat Interface]
      (Generated Answer + Expandable Page Citations)
```

---

## 🔬 Key Engineering Highlights

1. **Deterministic Dense Retrieval:** Uses `sentence-transformers/all-MiniLM-L6-v2` with L2-normalized vectors and `faiss.IndexFlatIP` to provide exact cosine-similarity ranking in sub-10 ms query times.
2. **Context Grounding & Prompt Injection Defense:** Context snippets are strictly formatted as untrusted evidence `[1] SOURCE: filename, page N` with explicit instructions forbidding fabrication and prompt hijacking.
3. **Dual Execution Backends:**
   - **Hosted LLM API (Production/Cloud):** Connects to OpenAI-compatible hosted inference endpoints (e.g., Groq with `openai/gpt-oss-20b` or Llama 3.3) for fast, zero-GPU cloud deployment.
   - **Fine-Tuned LoRA Llama 3 8B (CUDA GPU):** Preserves the exact inference pathway for the LoRA adapter fine-tuned on `yahma/alpaca-cleaned` using Unsloth in Google Colab.
4. **Retrieval-Only Mode:** Provides zero-credential demonstration capability for inspecting vector retrieval and citation matches without API keys.
5. **Session Isolation & Memory Safety:** Avoids unsafe pickle deserialization (`allow_dangerous_deserialization=True`) by building lightweight in-memory indexes per session.

---

## 📊 Empirical Evaluation Results

Evaluated on the included ground-truth benchmark (`sample_docs/evaluation_questions.json`) using `sample_docs/demo_notes.pdf`:

| Metric | Measured Value | Target | Status |
|---|---|---|---|
| **Recall@1** | **100.0%** (6/6) | > 80% | **PASS** |
| **Recall@3** | **100.0%** (6/6) | > 90% | **PASS** |
| **Recall@5** | **100.0%** (6/6) | > 95% | **PASS** |
| **Grounding Citation Fidelity** | **100.0%** | > 85% | **PASS** |
| **Mean Passage Token F1** | **0.483** | > 0.40 | **PASS** |
| **Mean Retrieval Latency** | **8.54 ms** | < 50 ms | **PASS** |
| **p95 Retrieval Latency** | **10.37 ms** | < 100 ms | **PASS** |

See [docs/EVALUATION.md](file:///c:/Development/documind-rag/docs/EVALUATION.md) for detailed per-query breakdowns and the complete comparative research matrix.

---

## 📁 Repository Structure

```text
documind-rag/
├── app.py                        # Streamlit web application & chat UI
├── rag_core.py                   # Document parsing, chunking, embeddings, FAISS retrieval
├── text_utils.py                 # Pure-Python text chunker with boundary snapping
├── llm_backends.py               # Hosted LLM inference & optional LoRA GPU backend
├── evaluate.py                   # Quantitative retrieval & grounding benchmark suite
├── requirements.txt              # Production CPU dependencies
├── requirements-gpu.txt          # Optional NVIDIA CUDA Unsloth/GPU requirements
├── .env.example                  # Environment variable template
├── .gitignore                    # Comprehensive secrets & artifact exclusion rules
├── .streamlit/
│   ├── config.toml               # Streamlit theme & file size limits
│   └── secrets.toml.example      # Example configuration for Streamlit Cloud
├── docs/
│   └── EVALUATION.md             # Empirical benchmark report and model matrix
├── notebooks/
│   └── original_colab.ipynb      # Original Unsloth LoRA fine-tuning & RAG research notebook
├── sample_docs/
│   ├── demo_notes.pdf            # 3-page software engineering sample notes
│   └── evaluation_questions.json # Labeled ground-truth questions and expected facts
└── tests/
    ├── test_text_utils.py        # Unit tests for text chunking & normalization
    ├── test_rag_core.py          # Unit tests for PyMuPDF parsing, FAISS, context budget
    ├── test_llm_backends.py      # Unit tests for hosted LLM & LoRA CUDA guard
    └── test_evaluate.py          # Unit tests for evaluation metrics & F1 scoring
```

---

## 🚀 Quickstart & Local Installation

### Prerequisites
- Python **3.11, 3.12, or 3.14**
- Git

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/pallaviramdas/documind-rag.git
cd documind-rag

# Create virtual environment
python -m venv .venv

# Activate environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment (Optional for Hosted LLM)

Create `.streamlit/secrets.toml` or export environment variables:

```toml
LLM_API_KEY = "your-groq-or-openai-api-key"
LLM_BASE_URL = "https://api.groq.com/openai/v1"
LLM_MODEL = "openai/gpt-oss-20b"
```

*(You can also run without an API key by selecting **Retrieval only** in the application sidebar).*

### 3. Run the Application

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. Upload `sample_docs/demo_notes.pdf` to test immediate question answering.

---

## 🧪 Running the Test Suite & Evaluation

### Run Unit Tests
```bash
python -m unittest discover -s tests -v
```
*Executes all 19 unit tests across text processing, document extraction, vector indexing, LLM backends, and evaluation.*

### Run Quantitative Evaluation Benchmark
```bash
python evaluate.py sample_docs/demo_notes.pdf sample_docs/evaluation_questions.json --output docs/EVALUATION.md
```

---

## ☁️ Deployment Guide (Streamlit Community Cloud)

1. Push this repository to GitHub as `documind-rag`.
2. Navigate to [share.streamlit.io](https://share.streamlit.io) and sign in.
3. Click **New app** and select:
   - **Repository:** `your-username/documind-rag`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Expand **Advanced settings → Secrets** and input:
   ```toml
   LLM_API_KEY = "your_actual_api_key"
   LLM_BASE_URL = "https://api.groq.com/openai/v1"
   LLM_MODEL = "openai/gpt-oss-20b"
   ```
5. Click **Deploy**. Streamlit Cloud builds the environment automatically from `requirements.txt`.

---

## 📄 Resume Description (Google Student Researcher Portfolio)

**DocuMind AI | Python, Llama 3 8B, Unsloth, LoRA, FAISS, Sentence Transformers, Streamlit**
- Architected an end-to-end RAG question answering system with FAISS dense vector search and Sentence Transformers embeddings (`all-MiniLM-L6-v2`), achieving **100% Recall@1** and sub-10 ms retrieval latency.
- Fine-tuned Llama 3 8B with QLoRA (4-bit quantization, 100 steps) on instruction datasets via Unsloth; engineered a production Streamlit application featuring dual-backend execution (Hosted cloud API vs. local GPU LoRA).
- Implemented context-grounded prompt synthesis with page-level citation attribution, token overlap F1 evaluation, and a comprehensive 19-test automated testing suite.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.
