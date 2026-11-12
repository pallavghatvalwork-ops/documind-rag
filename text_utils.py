"""Small dependency-free helpers used by the retrieval pipeline."""

from __future__ import annotations
import re


def normalize_text(text: str) -> str:
    """Collapse whitespace from extracted PDF text."""
    return re.sub(r"\s+", " ", text).strip()


def split_text(text: str, chunk_size: int = 850, overlap: int = 150) -> list[str]:
    """Split text into overlapping near-word-boundary chunks.

    Character based for predictable size; unlike token splitting, approximate
    retrieval size depends on the language and embedding model tokenizer.
    """
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller")
    cleaned = normalize_text(text)
    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        if end < len(cleaned):
            # Prefer a word boundary near the end, but never shorten too far.
            boundary = cleaned.rfind(" ", start + int(chunk_size * 0.75), end)
            if boundary != -1:
                end = boundary
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)
        if end == len(cleaned):
            break
        start = max(start + 1, end - overlap)
    return chunks
