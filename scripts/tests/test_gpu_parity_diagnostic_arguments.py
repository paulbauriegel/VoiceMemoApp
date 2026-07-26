from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1] / "run-gpu-parity-diagnostic.sh"
)


def run_script(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(
        {
            "ADB": "/usr/bin/true",
            "SQLITE3": "/usr/bin/true",
        }
    )
    return subprocess.run(
        [str(SCRIPT), *arguments],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


class ReverseDecoderSplitArgumentsTest(unittest.TestCase):
    def test_help_describes_reverse_split(self) -> None:
        result = run_script("run", "--help")

        self.assertEqual(0, result.returncode)
        self.assertIn("--reverse-decoder-split", result.stdout)
        self.assertIn("prefix on CPU", result.stdout)

    def test_reverse_split_requires_decoder_split(self) -> None:
        result = run_script(
            "run",
            "--memo-id",
            "memo",
            "--reference-job",
            "job",
            "--reverse-decoder-split",
        )

        self.assertNotEqual(0, result.returncode)
        self.assertIn(
            "--reverse-decoder-split requires --decoder-split",
            result.stderr,
        )

    def test_reverse_split_rejects_non_boundary_stage(self) -> None:
        result = run_script(
            "run",
            "--memo-id",
            "memo",
            "--reference-job",
            "job",
            "--decoder-split",
            "--reverse-decoder-split",
            "--decoder-split-stage",
            "pre-layernorm",
        )

        self.assertNotEqual(0, result.returncode)
        self.assertIn(
            "requires a generated after-* decoder boundary stage",
            result.stderr,
        )

    def test_reverse_after_one_block_passes_argument_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing_apk = Path(temporary) / "missing.apk"
            result = run_script(
                "run",
                "--memo-id",
                "memo",
                "--reference-job",
                "job",
                "--decoder-split",
                "--reverse-decoder-split",
                "--decoder-split-stage",
                "after-1-block",
                "--speculative-block-size",
                "32",
                "--app-apk",
                str(missing_apk),
                "--test-apk",
                str(missing_apk),
            )

        self.assertNotEqual(0, result.returncode)
        self.assertIn("App APK not found", result.stderr)
        self.assertNotIn(
            "--reverse-decoder-split requires",
            result.stderr,
        )


class PromotionGateArgumentsTest(unittest.TestCase):
    def test_help_describes_promotion_gate(self) -> None:
        result = run_script("whole", "--help")

        self.assertEqual(0, result.returncode)
        self.assertIn("--promotion-gate", result.stdout)
        self.assertIn("full-GPU/no-source-quant", result.stdout)

    def test_promotion_gate_requires_complete_decoder(self) -> None:
        result = run_script(
            "whole",
            "--memo-id",
            "memo",
            "--promotion-gate",
        )

        self.assertNotEqual(0, result.returncode)
        self.assertIn(
            "--promotion-gate requires --full-float-decoder",
            result.stderr,
        )


if __name__ == "__main__":
    unittest.main()
