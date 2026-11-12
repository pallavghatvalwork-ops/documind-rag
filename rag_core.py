"""PDF parsing, local embedding, and FAISS similarity search.

No pickle loading: we create an in-memory index from the files uploaded in the
current Streamlit session. This avoids unsafe deserialization of FAISS metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from text_utils import split_text

try:
    import pymupdf as fitz
except ImportError:
    import fitz  # PyMuPDF fallback


@dataclass(frozen=True)
class Chunk:
    filename: str
    page: int  # 1-indexed
    text: str


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    similarity: float  # inner product of L2-normalized sentence vectors


def pdf_to_chunks(
    pdf_data: bytes,
    filename: str,
    chunk_size: int = 850,
    overlap: int = 150,
) -> tuple[list[Chunk], int, int]:
    """Return chunks, page count, and number of image/blank pages without selectable text."""
    if not pdf_data:
        raise ValueError(f"{filename} is empty")
    if b"%PDF-" not in pdf_data[:1024]:
        raise ValueError(f"{filename} is not a valid PDF file")

    chunks: list[Chunk] = []
    blank_pages = 0
    try:
        with fitz.open(stream=pdf_data, filetype="pdf") as document:
            pages = len(document)
            for page_no, page in enumerate(document, start=1):
                text = page.get_text("text")
                if not text.strip():
                    blank_pages += 1
                    continue
                chunks.extend(
                    Chunk(filename=filename, page=page_no, text=part)
                    for part in split_text(text, chunk_size=chunk_size, overlap=overlap)
                )
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"Could not read PDF {filename}: {exc}") from exc

    return chunks, pages, blank_pages


class Retriever:
    """In-memory dense retrieval with cosine-equivalent normalized vectors."""

    def __init__(self, model, chunks: Iterable[Chunk]):
        import faiss

        self.chunks = list(chunks)
        if not self.chunks:
            raise ValueError("No extractable text found in the uploaded PDFs")
        self.model = model
        vectors = model.encode(
            [chunk.text for chunk in self.chunks],
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        vectors = np.asarray(vectors, dtype="float32")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search(self, question: str, k: int = 4) -> list[Hit]:
        if not question.strip() or not self.chunks:
            return []
        k = max(1, k)
        vector = self.model.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vector = np.asarray(vector, dtype="float32")
        top_k = min(k, len(self.chunks))
        scores, positions = self.index.search(vector, top_k)
        return [
            Hit(chunk=self.chunks[int(position)], similarity=float(score))
            for score, position in zip(scores[0], positions[0])
            if position >= 0
        ]


def build_context(hits: list[Hit], max_chars: int = 6500) -> str:
    """Prepare numbered evidence snippets for a grounded generation prompt."""
    items = []
    remaining = max_chars
    for number, hit in enumerate(hits, start=1):
        label = f"[{number}] SOURCE: {hit.chunk.filename}, page {hit.chunk.page}\n"
        prefix_len = len(label) + 2
        if remaining <= prefix_len:
            break
        available = remaining - prefix_len
        chunk_text = hit.chunk.text.strip()
        if len(chunk_text) <= available:
            excerpt = chunk_text
        else:
            cut = chunk_text[:available].rsplit(" ", 1)[0]
            excerpt = (cut if cut else chunk_text[:available]) + "..."
        if not excerpt:
            break
        items.append(label + excerpt)
        remaining -= len(label) + len(excerpt) + 2
        if remaining <= 0:
            break
    return "\n\n".join(items)
