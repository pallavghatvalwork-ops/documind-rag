import unittest
import numpy as np

from rag_core import Chunk, Hit, Retriever, build_context, pdf_to_chunks


class DummyEmbeddingModel:
    """Deterministic dummy embedding model for fast unit testing without downloading weights."""

    def __init__(self, dim: int = 16):
        self.dim = dim

    def encode(self, texts, **kwargs):
        arr = []
        for t in texts:
            # Deterministic pseudo-vector based on character values
            vec = np.zeros(self.dim, dtype="float32")
            for i, ch in enumerate(t[:self.dim]):
                vec[i] = ord(ch) % 10
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            arr.append(vec)
        return np.array(arr, dtype="float32")


class TestRAGCore(unittest.TestCase):
    def test_pdf_to_chunks_valid(self):
        with open("sample_docs/demo_notes.pdf", "rb") as f:
            pdf_bytes = f.read()
        chunks, pages, blank = pdf_to_chunks(pdf_bytes, "demo_notes.pdf")
        self.assertEqual(pages, 3)
        self.assertEqual(blank, 0)
        self.assertGreaterEqual(len(chunks), 3)
        self.assertEqual(chunks[0].filename, "demo_notes.pdf")
        self.assertEqual(chunks[0].page, 1)

    def test_pdf_to_chunks_invalid_header(self):
        with self.assertRaises(ValueError):
            pdf_to_chunks(b"Not a PDF header", "bad.pdf")

    def test_pdf_to_chunks_empty(self):
        with self.assertRaises(ValueError):
            pdf_to_chunks(b"", "empty.pdf")

    def test_retriever_initialization_empty_chunks(self):
        model = DummyEmbeddingModel()
        with self.assertRaises(ValueError):
            Retriever(model, [])

    def test_retriever_search(self):
        model = DummyEmbeddingModel()
        chunks = [
            Chunk("file1.pdf", 1, "Requirements engineering process and analysis."),
            Chunk("file1.pdf", 2, "Software testing and integration suites."),
            Chunk("file1.pdf", 3, "Version control system using Git and branches."),
        ]
        retriever = Retriever(model, chunks)

        # Normal search
        hits = retriever.search("Requirements engineering", k=2)
        self.assertEqual(len(hits), 2)
        self.assertIsInstance(hits[0], Hit)
        self.assertGreaterEqual(hits[0].similarity, -1.0)

        # Empty question returns empty list
        self.assertEqual(retriever.search("   "), [])

        # Search k larger than chunk count
        hits_large_k = retriever.search("Git", k=10)
        self.assertEqual(len(hits_large_k), 3)

    def test_build_context(self):
        chunks = [
            Chunk("file.pdf", 1, "First paragraph about software architectures."),
            Chunk("file.pdf", 2, "Second paragraph discussing database normalization."),
        ]
        hits = [Hit(chunk=chunks[0], similarity=0.9), Hit(chunk=chunks[1], similarity=0.8)]

        context = build_context(hits, max_chars=1000)
        self.assertIn("[1] SOURCE: file.pdf, page 1", context)
        self.assertIn("First paragraph", context)
        self.assertIn("[2] SOURCE: file.pdf, page 2", context)
        self.assertIn("Second paragraph", context)

        # Empty hits returns empty string
        self.assertEqual(build_context([]), "")

        # Small budget caps length without throwing
        short_context = build_context(hits, max_chars=50)
        self.assertLessEqual(len(short_context), 50)


if __name__ == "__main__":
    unittest.main()

