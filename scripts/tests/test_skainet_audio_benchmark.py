from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/run-skainet-audio-benchmark.py"


def load_script():
    spec = importlib.util.spec_from_file_location("skainet_audio_benchmark", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SkainetAudioBenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.benchmark = load_script()

    def test_discovers_exactly_five_pinned_fixtures(self) -> None:
        manifest = self.benchmark.validate_samples()

        self.assertEqual(5, len(manifest))
        self.assertEqual({sample.id for sample in self.benchmark.SAMPLES}, set(manifest))
        self.assertEqual({"en", "de"}, {item["language"] for item in manifest.values()})
        self.assertTrue(all(item["sample_count"] > 0 for item in manifest.values()))

    def test_word_error_reports_substitution_deletion_and_insertion(self) -> None:
        substitution_and_insertion = self.benchmark.word_error("one two three", "one too three four")
        deletion = self.benchmark.word_error("one two three", "one three")

        self.assertEqual(2, substitution_and_insertion["word_edits"])
        self.assertEqual(1, substitution_and_insertion["substitutions"])
        self.assertEqual(1, substitution_and_insertion["insertions"])
        self.assertEqual(1, deletion["word_edits"])
        self.assertEqual(1, deletion["deletions"])

    def test_instrumentation_command_targets_only_isolated_packages(self) -> None:
        command = self.benchmark.instrumentation_command(
            "adb",
            "serial",
            "sample-skainet-r1",
            "skainet",
            "sample.wav",
            "en",
        )

        self.assertIn(
            "com.example.voicememo.benchmark.test/androidx.test.runner.AndroidJUnitRunner",
            command,
        )
        self.assertNotIn("com.example.voicememo.test", command)

    def test_cleanup_rejects_any_real_app_or_directory_path(self) -> None:
        with mock.patch.object(self.benchmark, "run") as run:
            self.benchmark.safe_remove(
                "adb",
                "serial",
                "files/skainet-audio-benchmark/sample.wav",
            )
            self.assertEqual(1, run.call_count)

        for unsafe in (
            "databases/voice-memo.db",
            "files/skainet-audio-benchmark",
            "files/skainet-audio-benchmark/../recording.wav",
        ):
            with self.assertRaises(self.benchmark.BenchmarkError):
                self.benchmark.safe_remove("adb", "serial", unsafe)

    def test_resume_accepts_only_matching_completed_record(self) -> None:
        sample = self.benchmark.SAMPLES[0]
        value = {
            "sample_id": sample.id,
            "frontend": "voicememo",
            "repeat": 1,
            "audio_sha256": sample.audio_sha256,
            "raw_report": {
                "status": "completed",
                "package_name": self.benchmark.PACKAGE,
                "result": {"frontend": "voicememo"},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            self.assertTrue(
                self.benchmark.run_file_valid(path, sample, "voicememo", 1)
            )
            self.assertFalse(
                self.benchmark.run_file_valid(path, sample, "skainet", 1)
            )
            path.write_text("[]", encoding="utf-8")
            self.assertFalse(
                self.benchmark.run_file_valid(path, sample, "voicememo", 1)
            )

    def test_parity_resume_requires_matching_isolated_result(self) -> None:
        sample = self.benchmark.SAMPLES[0]
        value = {
            "sample_id": sample.id,
            "audio_sha256": sample.audio_sha256,
            "raw_report": {
                "status": "completed",
                "package_name": self.benchmark.PACKAGE,
                "parity": {"pcm": {}, "mel": {}},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parity.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            self.assertTrue(self.benchmark.parity_file_valid(path, sample))
            value["raw_report"]["package_name"] = self.benchmark.REAL_PACKAGE
            path.write_text(json.dumps(value), encoding="utf-8")
            self.assertFalse(self.benchmark.parity_file_valid(path, sample))
            path.write_text("[]", encoding="utf-8")
            self.assertFalse(self.benchmark.parity_file_valid(path, sample))

    def test_real_package_snapshot_guard_fails_closed(self) -> None:
        snapshot = {"installed": True, "last_update_time": "before", "uid": "123"}
        self.benchmark.assert_package_unchanged(snapshot, dict(snapshot))

        changed = dict(snapshot, last_update_time="after")
        with self.assertRaisesRegex(self.benchmark.BenchmarkError, "package changed"):
            self.benchmark.assert_package_unchanged(snapshot, changed)

    def test_apk_package_guard_rejects_real_voice_memo(self) -> None:
        self.assertEqual(
            self.benchmark.PACKAGE,
            self.benchmark.parse_apk_application_id(
                "package: name='com.example.voicememo.benchmark' versionCode='1'\n"
            ),
        )
        completed = mock.Mock(stdout="package: name='com.example.voicememo' versionCode='1'\n")
        with mock.patch.object(self.benchmark, "run", return_value=completed):
            with self.assertRaisesRegex(self.benchmark.BenchmarkError, "Refusing to install"):
                self.benchmark.assert_apk_package(
                    Path("/sdk/aapt"),
                    Path("app-debug.apk"),
                    self.benchmark.PACKAGE,
                )

    def test_aggregate_keeps_repeat_distribution(self) -> None:
        result = self.benchmark.aggregate([3.0, 1.0, 2.0])

        self.assertEqual(1.0, result["min"])
        self.assertEqual(2.0, result["median"])
        self.assertEqual(3.0, result["max"])

    def test_complete_suite_writes_all_reports_and_raw_transcripts(self) -> None:
        manifest = self.benchmark.validate_samples()
        runs = []
        for sample in self.benchmark.SAMPLES:
            for frontend in self.benchmark.FRONTENDS:
                for repeat in range(1, self.benchmark.REPEATS + 1):
                    result = {
                        "text": f"unedited {sample.id} {frontend}",
                        "pipeline_rtf": 0.5,
                        "feature_extraction_rtf": 0.1,
                        "frontend_initialization_millis": 1.0,
                        "model_load_wall_millis": 2.0,
                        "model_load_cpu_millis": 2.0,
                        "input_preparation_wall_millis": 3.0,
                        "input_preparation_cpu_millis": 3.0,
                        "feature_extraction_wall_millis": 4.0,
                        "feature_extraction_cpu_millis": 4.0,
                        "inference_wall_millis": 5.0,
                        "inference_cpu_millis": 5.0,
                        "pipeline_wall_millis": 8.0,
                        "pipeline_cpu_millis": 8.0,
                    }
                    runs.append(
                        {
                            "sample_id": sample.id,
                            "frontend": frontend,
                            "repeat": repeat,
                            "peak_pss_kib": 1024,
                            "word_error": {
                                "reference_words": 10,
                                "word_edits": 3,
                                "substitutions": 1,
                                "deletions": 1,
                                "insertions": 1,
                                "word_error_percent": 30.0,
                            },
                            "raw_report": {"status": "completed", "result": result},
                        }
                    )
        parity = {
            sample.id: {"raw_report": {"status": "completed"}}
            for sample in self.benchmark.SAMPLES
        }
        package = {"installed": True, "last_update_time": "unchanged"}

        summary = self.benchmark.summarize(
            "test-suite",
            manifest,
            runs,
            parity,
            {"model": "Pixel 7 Pro"},
            package,
            dict(package),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.benchmark.write_outputs(output, summary, runs)

            self.assertEqual(
                {"summary.json", "summary.csv", "summary.md", "transcripts.md"},
                {path.name for path in output.iterdir() if path.is_file()},
            )
            transcripts = list((output / "raw").glob("*/*.txt"))
            self.assertEqual(30, len(transcripts))
            self.assertTrue(all(path.read_text(encoding="utf-8").startswith("unedited ") for path in transcripts))
            self.assertEqual(1.0, summary["per_sample"][0]["substitutions"]["median"])


if __name__ == "__main__":
    unittest.main()
