import unittest
from text_utils import normalize_text, split_text


class TestChunking(unittest.TestCase):
    def test_whitespace(self):
        self.assertEqual(normalize_text("Hello\n\tworld  !"), "Hello world !")

    def test_overlap(self):
        text = " ".join(f"token{i}" for i in range(100))
        chunks = split_text(text, chunk_size=85, overlap=20)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(0 < len(chunk) <= 85 for chunk in chunks))
        self.assertTrue(any(word in chunks[1] for word in chunks[0].split()[-3:]))

    def test_blank(self):
        self.assertEqual(split_text(" \n "), [])

    def test_invalid_settings(self):
        with self.assertRaises(ValueError):
            split_text("abcd", chunk_size=10, overlap=10)


if __name__ == "__main__":
    unittest.main()
