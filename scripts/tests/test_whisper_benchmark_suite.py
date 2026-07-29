import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/run-whisper-benchmark-suite.py"
SPEC = importlib.util.spec_from_file_location("whisper_benchmark_suite", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SUITE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SUITE
SPEC.loader.exec_module(SUITE)


class WhisperBenchmarkSuiteTest(unittest.TestCase):
    def test_configuration_set_is_unique_and_excludes_single_worker(self):
        identifiers = [item.id for item in SUITE.CONFIGURATIONS]
        self.assertEqual(len(identifiers), 6)
        self.assertEqual(len(set(identifiers)), len(identifiers))
        self.assertNotIn("litert-int8-cpu-1x3", identifiers)
        four_worker = next(
            item
            for item in SUITE.CONFIGURATIONS
            if item.id == "litert-int8-cpu-4x3"
        )
        self.assertEqual(four_worker.expected_mode, "cpu-4x3")
        self.assertEqual(four_worker.instrumentation["cpuWorkers"], "4")
        self.assertEqual(four_worker.instrumentation["cpuThreads"], "3")

    def test_comparison_uses_words_and_tokens(self):
        result = SUITE.comparison(
            {"text": "Hello, world.", "tokens": [1, 2]},
            {"text": "hello brave world", "tokens": [1, 3, 2]},
        )
        self.assertEqual(result["reference_words"], 2)
        self.assertEqual(result["candidate_words"], 3)
        self.assertEqual(result["word_edits"], 1)
        self.assertEqual(result["token_edits"], 1)
        self.assertFalse(result["exact_text_match"])
        self.assertFalse(result["exact_token_match"])

    def test_aggregate_preserves_repeat_distribution(self):
        result = SUITE.aggregate([3.0, 1.0, 2.0])
        self.assertEqual(result["min"], 1.0)
        self.assertEqual(result["median"], 2.0)
        self.assertEqual(result["mean"], 2.0)
        self.assertEqual(result["max"], 3.0)

    def test_instrumentation_run_id_removes_underscores(self):
        result = SUITE.instrumentation_run_id(
            "controlled.3x",
            "whispercpp-q8_0-vulkan",
            1,
        )
        self.assertEqual(result, "controlled-3x-whispercpp-q8-0-vulkan-r1")
        self.assertRegex(result, r"^[A-Za-z0-9-]+$")


if __name__ == "__main__":
    unittest.main()
