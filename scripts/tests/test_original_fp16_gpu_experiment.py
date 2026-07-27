from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "run-original-fp16-gpu-experiment.py"
)
SPEC = importlib.util.spec_from_file_location("original_fp16_gpu", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)


class OriginalFp16GpuExperimentTest(unittest.TestCase):
    def test_audio_windows_use_28_seconds_with_two_second_overlap(self) -> None:
        samples = [0.0] * (55 * experiment.SAMPLE_RATE)
        windows = experiment.audio_windows(samples)

        self.assertEqual([start for start, _ in windows], [0, 416_000, 832_000])
        self.assertEqual(len(windows[0][1]), 448_000)
        self.assertEqual(len(windows[1][1]), 448_000)
        self.assertEqual(len(windows[2][1]), 48_000)

    def test_token_overlap_uses_longest_exact_boundary(self) -> None:
        self.assertEqual(
            experiment.merge_token_overlap([1, 2, 3, 4], [3, 4, 5]),
            [1, 2, 3, 4, 5],
        )
        self.assertEqual(
            experiment.merge_token_overlap([1, 2], [3, 4]),
            [1, 2, 3, 4],
        )

    def test_comparison_reports_word_and_token_error_rates(self) -> None:
        result = experiment.comparison(
            {"text": "Hello quiet world", "tokens": [1, 2, 3]},
            {"text": "hello world", "tokens": [1, 3]},
        )

        self.assertEqual(result["word_edits"], 1)
        self.assertAlmostEqual(result["word_error_rate"], 1 / 3)
        self.assertEqual(result["token_edits"], 1)
        self.assertAlmostEqual(result["token_error_rate"], 1 / 3)

    def test_free_space_guard_fails_before_large_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(experiment.ExperimentError):
                experiment.require_free_space(Path(directory), 1_000_000)

    def test_parser_requires_audio_and_supports_all_pipeline(self) -> None:
        args = experiment.parser().parse_args(
            [
                "all",
                "--audio",
                "memo.wav",
                "--language",
                "de",
                "--skip-build",
            ]
        )

        self.assertEqual(args.command, "all")
        self.assertEqual(args.language, "de")
        self.assertTrue(args.skip_build)

    def test_convert_stage_does_not_require_audio(self) -> None:
        args = experiment.parser().parse_args(["convert"])

        self.assertEqual(args.command, "convert")
        self.assertIsNone(args.audio)
        self.assertIsNone(args.memo_id)


if __name__ == "__main__":
    unittest.main()
