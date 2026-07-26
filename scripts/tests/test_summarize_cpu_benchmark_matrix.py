from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path


HELPER_PATH = (
    Path(__file__).resolve().parents[1] / "summarize_cpu_benchmark_matrix.py"
)
SPEC = importlib.util.spec_from_file_location("cpu_benchmark_summary", HELPER_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def sample_report(
    workers: int = 2,
    threads: int = 3,
    transcription_wall_ms: int = 50_000,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "timestamp_millis": 1_750_000_000_000 + workers * 100 + threads,
        "outcome": "SUCCESS",
        "error_code": None,
        "backend": "CPU_XNNPACK",
        "gpu_delegate": False,
        "xnnpack": True,
        "workers": workers,
        "threads_per_worker": threads,
        "maximum_inference_threads": workers * threads,
        "device": "Google Pixel 7 Pro",
        "soc": "Tensor G2",
        "android_sdk": 36,
        "audio_duration_ms": 120_000,
        "segment_count": 5,
        "precompleted_segments": 0,
        "fresh_full_run": True,
        "completed_segments": 5,
        "vad_skipped_segments": 1,
        "model_load_wall_ms": 3_000,
        "model_load_process_cpu_ms": 2_500,
        "transcription_wall_ms": transcription_wall_ms,
        "transcription_process_cpu_ms": 91_000,
        "summed_segment_inference_ms": 80_000,
        "total_job_wall_ms": transcription_wall_ms + 4_000,
        "total_job_process_cpu_ms": 94_000,
        "start_pss_kb": 1_000_000,
        "end_pss_kb": 1_500_000,
        "start_thermal_status": "NONE",
        "maximum_thermal_status": "LIGHT",
        "end_thermal_status": "NONE",
        "real_time_factor": transcription_wall_ms / 120_000,
        "audio_speed_x": 120_000 / transcription_wall_ms,
        "mean_active_inference_lanes": 1.6,
    }


class ValidateReportTest(unittest.TestCase):
    def test_accepts_fresh_cpu_report_and_matching_memo_fingerprint(self) -> None:
        baseline = sample_report(2, 2)
        candidate = sample_report(1, 6)

        MODULE.validate_report(candidate, 1, 6, baseline)

    def test_rejects_resumed_report(self) -> None:
        report = sample_report()
        report["fresh_full_run"] = False
        report["precompleted_segments"] = 2

        with self.assertRaisesRegex(
            MODULE.ReportValidationError,
            "fresh_full_run",
        ):
            MODULE.validate_report(report, 2, 3)

    def test_rejects_different_audio_fingerprint(self) -> None:
        baseline = sample_report(2, 2)
        candidate = sample_report(2, 4)
        candidate["audio_duration_ms"] = 120_001

        with self.assertRaisesRegex(
            MODULE.ReportValidationError,
            "audio_duration_ms",
        ):
            MODULE.validate_report(candidate, 2, 4, baseline)

    def test_rejects_wrong_build_configuration(self) -> None:
        report = sample_report(2, 3)

        with self.assertRaisesRegex(MODULE.ReportValidationError, "workers"):
            MODULE.validate_report(report, 1, 3)

    def test_rejects_missing_thermal_summary_field(self) -> None:
        report = sample_report()
        del report["maximum_thermal_status"]

        with self.assertRaisesRegex(
            MODULE.ReportValidationError,
            "maximum_thermal_status",
        ):
            MODULE.validate_report(report, 2, 3)

    def test_rejects_non_finite_metric(self) -> None:
        report = sample_report()
        report["real_time_factor"] = math.nan

        with self.assertRaisesRegex(
            MODULE.ReportValidationError,
            "real_time_factor",
        ):
            MODULE.validate_report(report, 2, 3)


class SummarizeTest(unittest.TestCase):
    def test_writes_content_free_csv_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first_path = directory / "2x3.json"
            second_path = directory / "1x6.json"
            first_path.write_text(
                json.dumps(sample_report(2, 3, 50_000)),
                encoding="utf-8",
            )
            second_path.write_text(
                json.dumps(sample_report(1, 6, 40_000)),
                encoding="utf-8",
            )

            rows = MODULE.summarize([first_path, second_path], directory)

            self.assertEqual(["2x3", "1x6"], [row["config"] for row in rows])
            csv_text = (directory / "summary.csv").read_text(encoding="utf-8")
            markdown = (directory / "summary.md").read_text(encoding="utf-8")
            self.assertIn("config,workers,threads_per_worker", csv_text)
            self.assertIn("Fastest accepted run: **1x6**", markdown)
            self.assertIn("2x3", markdown)
            self.assertNotIn("memo title", csv_text.lower())


if __name__ == "__main__":
    unittest.main()
