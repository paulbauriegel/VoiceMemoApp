"""Checks for source preservation, context boundaries, and resumable local runs."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "cleanup", Path(__file__).resolve().parents[1] / "run-transcript-cleanup-benchmark.py")
cleanup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cleanup)


class CleanupBenchmarkTest(unittest.TestCase):
    def test_context_contains_original_neighbors_only(self):
        items = cleanup.segments("# Title\n**00:00:01**\nVorher.\n**00:01:00**\nZiel.\n**00:02:00**\nDanach.")
        request = cleanup.messages(items, 1, {"people": ["Nora"]}, "context")
        payload = json.loads(request[1]["content"])
        self.assertEqual(payload["TARGET"], "Ziel.")
        self.assertEqual(payload["PREVIOUS_CONTEXT_DO_NOT_OUTPUT"], "Vorher.")
        self.assertEqual(payload["NEXT_CONTEXT_DO_NOT_OUTPUT"], "Danach.")

    def test_flags_damage_without_claiming_accuracy(self):
        result = cleanup.diagnostics("Ich habe nicht 15 Euro bezahlt.", "Ich habe 50 Euro bezahlt.", "length")
        self.assertIn("numeric_tokens_changed", result["flags"])
        self.assertIn("negation_tokens_changed", result["flags"])
        self.assertIn("generation_not_stopped_normally", result["flags"])
        self.assertNotIn("wer", result)
        self.assertEqual(cleanup.diagnostics("Hallo, Welt!", "Hallo Welt.", "stop")["word_change_rate_not_wer"], 0)

    def test_prepare_run_resume_preserves_source_and_raw_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, context = root / "source.md", root / "context.json"
            original = "# Title\n**00:00:01**\nich habe nicht zugesagt\n"
            source.write_text(original)
            context.write_text('{"people": ["Nora"]}')
            run = root / "run"
            cleanup.prepare(SimpleNamespace(transcript=source, context=context, run=run, models=["test:local"]))
            manifest = json.loads((run / "manifest.json").read_text())
            self.assertEqual(len(manifest["jobs"]), 2)
            calls = []

            def fake_api(endpoint, method, payload=None):
                calls.append(endpoint)
                if endpoint == "tags":
                    return {"models": [{"name": "test:local", "digest": "abc"}]}
                if endpoint == "version":
                    return {"version": "test"}
                if endpoint == "show":
                    return {"details": {"quantization_level": "Q4_K_M"}}
                return {"done": True, "done_reason": "stop", "message": {"content": "Ich habe nicht zugesagt."}}

            args = SimpleNamespace(run=run, model=None, limit=0)
            with patch.object(cleanup, "api", side_effect=fake_api):
                cleanup.run(args)
                cleanup.run(args)
            self.assertEqual(calls.count("chat"), 2)
            self.assertEqual(source.read_text(), original)
            self.assertEqual(json.loads((run / "status.json").read_text())["state"], "complete")
            self.assertIsNone(json.loads((run / "status.json").read_text())["accuracy"])
            request_path = next((run / "requests").glob("*.json"))
            request_path.write_text("{}")
            # Completed output is retained on resume, not regenerated silently.
            self.assertEqual(len(list((run / "responses").glob("*.json"))), 2)

    def test_redirect_cannot_send_private_transcript_elsewhere(self):
        with self.assertRaises(ValueError):
            cleanup.NoRedirect().redirect_request(None, None, None, None, None, None)

    def test_changed_source_is_rejected_before_any_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, context = root / "source.md", root / "context.json"
            source.write_text("**00:00:01**\nOriginal.\n")
            context.write_text("{}")
            run = root / "run"
            cleanup.prepare(SimpleNamespace(transcript=source, context=context, run=run, models=["test:local"]))
            source.write_text("**00:00:01**\nChanged.\n")
            with patch.object(cleanup, "api") as mock:
                with self.assertRaisesRegex(ValueError, "Source transcript changed"):
                    cleanup.run(SimpleNamespace(run=run, model=None, limit=0))
                mock.assert_not_called()

    def test_remote_backed_local_tag_is_rejected_before_chat(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, context = root / "source.md", root / "context.json"
            source.write_text("**00:00:01**\nOriginal.\n")
            context.write_text("{}")
            run = root / "run"
            cleanup.prepare(SimpleNamespace(transcript=source, context=context, run=run, models=["test:local"]))
            calls = []

            def fake_api(endpoint, method, payload=None):
                calls.append(endpoint)
                if endpoint == "tags":
                    return {"models": [{"name": "test:local"}]}
                if endpoint == "version":
                    return {"version": "test"}
                return {"remote_host": "https://example.com"}

            with patch.object(cleanup, "api", side_effect=fake_api):
                with self.assertRaisesRegex(ValueError, "Remote-backed"):
                    cleanup.run(SimpleNamespace(run=run, model=None, limit=0))
            self.assertNotIn("chat", calls)


if __name__ == "__main__":
    unittest.main()
