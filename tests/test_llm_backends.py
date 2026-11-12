import unittest
from unittest.mock import MagicMock, patch

from llm_backends import LORA_PROMPT, generate_hosted, load_lora_adapter
from rag_core import Chunk, Hit


class TestLLMBackends(unittest.TestCase):
    def setUp(self):
        self.chunk = Chunk("sample.pdf", 1, "Sample evidence text for testing.")
        self.hits = [Hit(chunk=self.chunk, similarity=0.95)]

    def test_generate_hosted_missing_api_key(self):
        with self.assertRaises(ValueError):
            generate_hosted("What is this?", self.hits, "", "http://example.com", "dummy-model")

    def test_generate_hosted_empty_hits(self):
        result = generate_hosted("What is this?", [], "mock-key", "http://example.com", "dummy-model")
        self.assertIn("No relevant passages", result)

    @patch("openai.OpenAI")
    def test_generate_hosted_mock_call(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = "Requirements engineering is the elicitation of stakeholder requirements [1]."
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        answer = generate_hosted(
            question="What is requirements engineering?",
            hits=self.hits,
            api_key="test-api-key",
            base_url="https://api.groq.com/openai/v1",
            model_name="openai/gpt-oss-20b",
        )
        self.assertEqual(answer, "Requirements engineering is the elicitation of stakeholder requirements [1].")
        mock_client.chat.completions.create.assert_called_once()

    @patch("torch.cuda.is_available", return_value=False)
    def test_load_lora_adapter_cuda_guard(self, _mock_cuda):
        with self.assertRaises(RuntimeError) as ctx:
            load_lora_adapter("/dummy/path")
        self.assertIn("requires an NVIDIA CUDA GPU", str(ctx.exception))

    def test_lora_prompt_formatting(self):
        query = "What is Git?"
        data = "[1] Git is a distributed VCS"
        formatted = LORA_PROMPT.format(query, data, "")
        self.assertIn("### Query:\nWhat is Git?", formatted)
        self.assertIn("### Data:\n[1] Git is a distributed VCS", formatted)


if __name__ == "__main__":
    unittest.main()

