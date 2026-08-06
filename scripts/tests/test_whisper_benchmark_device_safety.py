from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/run-whisper-benchmark-suite.py"


def load_script():
    spec = importlib.util.spec_from_file_location("whisper_benchmark_suite", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load Whisper benchmark suite")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class WhisperBenchmarkDeviceSafetyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.suite = load_script()

    def test_idle_check_rejects_non_debuggable_app_before_database_snapshot(self) -> None:
        failure = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="run-as: package not debuggable",
        )
        with mock.patch.object(self.suite, "run", return_value=failure) as run:
            with self.assertRaisesRegex(
                self.suite.SuiteError,
                "signature-compatible debuggable APK",
            ):
                self.suite.assert_device_idle("adb", "device")

        command = run.call_args.args[0]
        self.assertIn("shell", command)
        self.assertNotIn("exec-out", command)
        self.assertEqual(1, run.call_count)


if __name__ == "__main__":
    unittest.main()
