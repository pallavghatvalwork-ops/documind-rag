"""DocuMind AI — RAG-Based PDF Question Answering System.

Run locally:
    streamlit run app.py
"""
from __future__ import annotations

import hashlib
import os
import time

# Ensure offline cache availability
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import streamlit as st

from llm_backends import generate_hosted, generate_lora, load_lora_adapter
from rag_core import Retriever, pdf_to_chunks

# Page configuration
st.set_page_config(
    page_title="DocuMind AI — RAG Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for polished, academic/portfolio presentation
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E1B4B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .badge-pill {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        background-color: #EEF2FF;
        color: #4F46E5;
        border: 1px solid #C7D2FE;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .source-card {
        border-left: 3px solid #6366F1;
        padding: 0.6rem 0.9rem;
        background-color: #F8FAFC;
        border-radius: 0 6px 6px 0;
        margin-bottom: 0.75rem;
    }
    .stAlert {
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading Sentence Transformer embeddings...")
def load_embeddings():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Loading LoRA adapter into GPU memory...")
def cached_lora(adapter_path: str):
    return load_lora_adapter(adapter_path)


def get_config(name: str, fallback: str = "") -> str:
    """Retrieve setting from environment variables, Streamlit Secrets, or local .streamlit/secrets.toml."""
    # 1. Environment variables
    env = os.environ.get(name)
    if env and env.strip():
        return env.strip()

    # 2. Native Streamlit secrets (Streamlit Community Cloud or local)
    try:
        if name in st.secrets and str(st.secrets[name]).strip():
            return str(st.secrets[name]).strip()
    except Exception:
        pass

    # 3. Direct local .streamlit/secrets.toml fallback (handles sandbox/OS permissions)
    local_path = os.path.join(".streamlit", "secrets.toml")
    if os.path.exists(local_path):
        try:
            import tomllib

            with open(local_path, "rb") as f:
                data = tomllib.load(f)
            if name in data and str(data[name]).strip():
                return str(data[name]).strip()
        except Exception:
            pass

    return fallback


def compute_file_signature(uploaded_files) -> str:
    digest = hashlib.sha256()
    for item in uploaded_files:
        digest.update(item.name.encode("utf-8"))
        digest.update(item.getvalue())
    return digest.hexdigest()


# Sidebar Configuration
with st.sidebar:
    st.markdown("### 📚 DocuMind AI")
    st.caption("Document Grounding & Retrieval Augmented Generation")
    st.markdown("---")

    st.subheader("1. Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or multiple text-based PDFs (up to 10 files, max 10MB each).",
    )

    st.subheader("2. Answering Engine")
    mode = st.radio(
        "Execution Mode",
        ["Hosted LLM API", "Retrieval only", "My fine-tuned LoRA (GPU)"],
        index=0,
        help=(
            "Hosted LLM API connects to Groq / OpenAI-compatible endpoint. "
            "Retrieval only requires no API key. "
            "LoRA GPU executes notebook-trained weights locally on an NVIDIA CUDA device."
        ),
    )

    api_key = get_config("LLM_API_KEY", "")
    base_url = get_config("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    model_name = get_config("LLM_MODEL", "openai/gpt-oss-20b")
    adapter_path = ""

    if mode == "Hosted LLM API":
        if api_key:
            st.success("API key loaded from secrets / environment.", icon="🔑")
            user_override = st.text_input(
                "Override API Key (Optional)",
                type="password",
                placeholder="Leave blank to use configured secret",
                help="Enter an alternate API key here if you wish to override the configured secret.",
            )
            if user_override and user_override.strip():
                api_key = user_override.strip()
        else:
            api_key = st.text_input(
                "API Key",
                type="password",
                placeholder="gsk_... or sk-...",
                help="Set LLM_API_KEY in environment or .streamlit/secrets.toml to avoid typing here.",
            )
        st.text_input("Base URL", value=base_url, key="base_url_input", disabled=True)
        st.text_input("Model ID", value=model_name, key="model_name_input", disabled=True)
        st.info(
            "💡 **Architecture Note:** The hosted demo connects to an OpenAI-compatible cloud model. "
            "It does **not** run your local 8B LoRA fine-tune. To run the fine-tuned model, select the GPU mode.",
            icon="ℹ️",
        )
    elif mode == "My fine-tuned LoRA (GPU)":
        adapter_path = st.text_input(
            "Adapter Path",
            value=get_config("LORA_ADAPTER_PATH", ""),
            placeholder="e.g., /path/to/llama_model",
            help="Directory containing adapter_config.json and adapter_model.safetensors.",
        )
        st.caption("Requires CUDA-enabled GPU and compatible Unsloth/PyTorch environment.")
    else:
        st.success("✅ **Offline Mode:** Inspects semantic retrieval and citations without external API calls.")

    st.subheader("3. Retrieval Parameters")
    top_k = st.slider(
        "Top-k Passages",
        min_value=1,
        max_value=8,
        value=4,
        help="Number of most relevant text chunks to retrieve via FAISS.",
    )

    with st.expander("⚙️ System Architecture"):
        st.markdown(
            """
            - **Embedding Model:** `all-MiniLM-L6-v2` (384-dim)
            - **Vector Store:** FAISS Flat Inner-Product (`IndexFlatIP`)
            - **Chunking:** 850 chars / 150 char overlap
            - **Normalization:** Cosine similarity via L2-normalized vectors
            - **Extraction:** PyMuPDF (fitz)
            """
        )

    st.markdown("---")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

# Main Application Body
st.markdown('<div class="main-title">DocuMind AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Document-grounded question answering with FAISS dense vector search and citation attribution.</div>',
    unsafe_allow_html=True,
)

# Badges
st.markdown(
    """
    <div>
        <span class="badge-pill">RAG Architecture</span>
        <span class="badge-pill">FAISS Vector Index</span>
        <span class="badge-pill">Sentence-Transformers</span>
        <span class="badge-pill">Citation Grounding</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Validate file upload constraints
if uploaded_files:
    if len(uploaded_files) > 10:
        st.error("⚠️ Maximum 10 PDFs allowed per session. Please remove extra files.")
        st.stop()
    for item in uploaded_files:
        if len(item.getvalue()) > 10 * 1024 * 1024:
            st.error(f"⚠️ File '{item.name}' exceeds the 10 MB limit.")
            st.stop()

# Indexing Logic
if uploaded_files:
    signature = compute_file_signature(uploaded_files)
    if st.session_state.get("file_signature") != signature:
        progress_bar = st.progress(0, text="Reading and extracting text from PDFs...")
        all_chunks = []
        total_pages = 0
        total_blank = 0
        file_details = []

        try:
            embed_model = load_embeddings()
            for idx, item in enumerate(uploaded_files):
                file_pct = int(((idx + 0.3) / len(uploaded_files)) * 80)
                progress_bar.progress(file_pct, text=f"Extracting {item.name}...")
                chunks, pages, blank = pdf_to_chunks(item.getvalue(), item.name)
                all_chunks.extend(chunks)
                total_pages += pages
                total_blank += blank
                file_details.append({"name": item.name, "pages": pages, "chunks": len(chunks)})

            progress_bar.progress(85, text="Building FAISS dense vector index...")
            retriever = Retriever(embed_model, all_chunks)

            st.session_state["retriever"] = retriever
            st.session_state["page_count"] = total_pages
            st.session_state["blank_pages"] = total_blank
            st.session_state["file_details"] = file_details
            st.session_state["file_signature"] = signature
            st.session_state["messages"] = []
            progress_bar.progress(100, text="Indexing complete!")
            time.sleep(0.4)
            progress_bar.empty()
        except Exception as exc:
            progress_bar.empty()
            st.session_state.pop("retriever", None)
            st.error(f"❌ Indexing failed: {exc}")
            st.stop()

    # Document Overview Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Documents", len(uploaded_files))
    with col2:
        st.metric("Total Pages", st.session_state.get("page_count", 0))
    with col3:
        st.metric("Vector Chunks", len(st.session_state["retriever"].chunks))
    with col4:
        st.metric("Active Mode", "Hosted LLM" if mode == "Hosted LLM API" else ("LoRA GPU" if mode.startswith("My") else "Retrieval"))

    if st.session_state.get("blank_pages", 0) > 0:
        st.warning(
            f"ℹ️ {st.session_state['blank_pages']} page(s) had no selectable text (scanned or image-only). "
            "Text-based PDFs are recommended.",
            icon="⚠️",
        )
else:
    st.info(
        "👈 **Get Started:** Upload one or more PDF files using the sidebar to build your in-memory vector index. "
        "Try `sample_docs/demo_notes.pdf` for a quick demonstration.",
        icon="📖",
    )
    st.stop()

# Sample prompt helper
sample_questions = [
    "What is requirements engineering?",
    "What is the purpose of software testing?",
    "What does version control do?",
]
st.markdown("**Suggested Questions:**")
q_cols = st.columns(len(sample_questions))
chosen_prompt = None
for q_idx, q_text in enumerate(sample_questions):
    with q_cols[q_idx]:
        if st.button(f"💡 {q_text}", key=f"sq_{q_idx}", use_container_width=True):
            chosen_prompt = q_text

# Render conversation history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("hits"):
            with st.expander(f"📑 View Supporting Sources ({len(msg['hits'])} passages)"):
                for rank, hit in enumerate(msg["hits"], start=1):
                    st.markdown(
                        f"""
                        <div class="source-card">
                            <strong>[{rank}] {hit.chunk.filename} — Page {hit.chunk.page}</strong><br/>
                            <small>Cosine Similarity: <code>{hit.similarity:.4f}</code></small><br/>
                            <p style="margin-top: 0.4rem; font-size: 0.92rem;">{hit.chunk.text}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# Handle user input
user_input = st.chat_input("Ask a question about your documents...")
prompt_to_process = chosen_prompt or user_input

if prompt_to_process:
    clean_question = prompt_to_process.strip()[:1000]
    st.session_state["messages"].append({"role": "user", "content": clean_question})

    with st.chat_message("user"):
        st.markdown(clean_question)

    with st.chat_message("assistant"):
        with st.spinner("Searching document index and synthesizing response..."):
            retriever: Retriever = st.session_state["retriever"]
            t0 = time.perf_counter()
            hits = retriever.search(clean_question, k=top_k)
            retrieval_ms = (time.perf_counter() - t0) * 1000.0

            try:
                if mode == "Retrieval only":
                    answer = (
                        f"**Relevant evidence retrieved in {retrieval_ms:.1f} ms.**\n\n"
                        "Passages are displayed below. To generate natural-language answers, "
                        "select **Hosted LLM API** in the sidebar."
                    )
                elif mode == "Hosted LLM API":
                    answer = generate_hosted(clean_question, hits, api_key, base_url, model_name)
                else:
                    if not adapter_path:
                        raise ValueError("Please provide a local path or ID for the LoRA adapter.")
                    answer = generate_lora(clean_question, hits, cached_lora(adapter_path))
            except Exception as exc:
                answer = (
                    f"⚠️ **Inference Warning:** {exc}\n\n"
                    "Semantic search succeeded, and relevant passages are available below for inspection."
                )

            st.markdown(answer)

            if hits:
                with st.expander(
                    f"📑 View Supporting Sources ({len(hits)} passages, retrieved in {retrieval_ms:.1f} ms)",
                    expanded=(mode == "Retrieval only"),
                ):
                    for rank, hit in enumerate(hits, start=1):
                        st.markdown(
                            f"""
                            <div class="source-card">
                                <strong>[{rank}] {hit.chunk.filename} — Page {hit.chunk.page}</strong><br/>
                                <small>Cosine Similarity: <code>{hit.similarity:.4f}</code></small><br/>
                                <p style="margin-top: 0.4rem; font-size: 0.92rem;">{hit.chunk.text}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    st.session_state["messages"].append({"role": "assistant", "content": answer, "hits": hits})

st.markdown("---")
st.caption(
    "DocuMind AI Portfolio Project | Evaluated on Sentence Transformers & FAISS | "
    "Answers are grounded on extracted PDF context. Verify critical information against primary sources."
)
