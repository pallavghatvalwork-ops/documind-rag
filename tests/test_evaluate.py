import unittest
from evaluate import compute_token_f1, run_evaluation


class TestEvaluate(unittest.TestCase):
    def test_compute_token_f1_exact(self):
        text = "Requirements engineering is the process of eliciting requirements."
        metrics = compute_token_f1(text, text)
        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["f1"], 1.0)

    def test_compute_token_f1_partial(self):
        pred = "Requirements engineering eliciting requirements"
        ref = "Requirements engineering analyzing requirements"
        metrics = compute_token_f1(pred, ref)
        self.assertGreater(metrics["f1"], 0.0)
        self.assertLess(metrics["f1"], 1.0)

    def test_compute_token_f1_empty(self):
        metrics = compute_token_f1("", "reference")
        self.assertEqual(metrics["f1"], 0.0)

    def test_run_evaluation_pipeline(self):
        results = run_evaluation(
            pdf_path="sample_docs/demo_notes.pdf",
            questions_path="sample_docs/evaluation_questions.json",
            max_k=3,
        )
        self.assertIn("summary", results)
        self.assertIn("queries", results)
        summary = results["summary"]
        self.assertEqual(summary["dataset_size"], 6)
        self.assertGreaterEqual(summary["recall@1"], 0.8)
        self.assertGreaterEqual(summary["recall@3"], 0.9)
        self.assertGreater(summary["latency_mean_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()

