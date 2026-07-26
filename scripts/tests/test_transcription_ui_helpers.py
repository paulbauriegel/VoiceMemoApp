from __future__ import annotations

import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1]
NODE_HELPER = SCRIPTS / "find-uiautomator-node.py"
PARITY_HELPER = SCRIPTS / "compare-transcription-parity.py"
CPU_FINGERPRINT = (
    "model=model-sha|tokenizer=tokenizer-revision|pipeline=pipeline-revision|"
    "runtime=cpu-xnnpack-2x3"
)
GPU_FINGERPRINT = (
    "model=model-sha|tokenizer=tokenizer-revision|pipeline=pipeline-revision|"
    "runtime=gpu-parity-guard_cpu-xnnpack-2x3"
)


def create_result_database(
    path: Path,
    *,
    job_id: str,
    memo_id: str,
    token_ids: str,
    raw_text: str,
    generated_text: str,
    status: str = "SUCCEEDED",
    total_chunks: int = 1,
    completed_chunks: int = 1,
    fingerprint: str | None = CPU_FINGERPRINT,
    updated_at_millis: int = 1_000,
    generated_at_millis: int | None = None,
    window_start_millis: int = 0,
    window_end_millis: int = 28_000,
    chunk_status: str = "COMPLETED",
) -> None:
    if generated_at_millis is None:
        generated_at_millis = updated_at_millis
    database = sqlite3.connect(path)
    try:
        database.executescript(
            """
            CREATE TABLE transcription_jobs (
                id TEXT PRIMARY KEY,
                memoId TEXT NOT NULL,
                status TEXT NOT NULL,
                totalChunks INTEGER NOT NULL,
                completedChunks INTEGER NOT NULL,
                inferenceFingerprint TEXT,
                updatedAtMillis INTEGER NOT NULL
            );
            CREATE TABLE transcript_chunks (
                jobId TEXT NOT NULL,
                chunkIndex INTEGER NOT NULL,
                memoId TEXT NOT NULL,
                windowStartMillis INTEGER NOT NULL,
                windowEndMillis INTEGER NOT NULL,
                status TEXT NOT NULL,
                tokenIds TEXT NOT NULL,
                rawText TEXT NOT NULL,
                detectedLanguage TEXT,
                truncated INTEGER NOT NULL
            );
            CREATE TABLE transcripts (
                memoId TEXT PRIMARY KEY,
                generatedText TEXT,
                generatedAtMillis INTEGER
            );
            """
        )
        database.execute(
            """
            INSERT INTO transcription_jobs(
                id, memoId, status, totalChunks, completedChunks,
                inferenceFingerprint, updatedAtMillis
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                memo_id,
                status,
                total_chunks,
                completed_chunks,
                fingerprint,
                updated_at_millis,
            ),
        )
        database.execute(
            """
            INSERT INTO transcript_chunks(
                jobId, chunkIndex, memoId, windowStartMillis, windowEndMillis,
                status, tokenIds, rawText, detectedLanguage, truncated
            ) VALUES (?, 0, ?, ?, ?, ?, ?, ?, 'en', 0)
            """,
            (
                job_id,
                memo_id,
                window_start_millis,
                window_end_millis,
                chunk_status,
                token_ids,
                raw_text,
            ),
        )
        database.execute(
            """
            INSERT INTO transcripts(memoId, generatedText, generatedAtMillis)
            VALUES (?, ?, ?)
            """,
            (memo_id, generated_text, generated_at_millis),
        )
        database.commit()
    finally:
        database.close()


def compare(reference: Path, actual: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(PARITY_HELPER),
            "--reference-db",
            str(reference),
            "--reference-job",
            "cpu",
            "--actual-db",
            str(actual),
            "--actual-job",
            "candidate",
        ],
        check=False,
        capture_output=True,
        text=True,
    )


class UiNodeHelperTest(unittest.TestCase):
    def test_finds_compose_resource_id_and_prints_center(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            hierarchy = Path(temporary) / "window.xml"
            hierarchy.write_text(
                """
                <hierarchy>
                  <node resource-id="memo_card_abc"
                        text="" content-desc=""
                        bounds="[20,100][220,300]" />
                </hierarchy>
                """,
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(NODE_HELPER),
                    str(hierarchy),
                    "--resource-id",
                    "memo_card_abc",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(0, result.returncode)
            self.assertEqual("120 200", result.stdout.strip())


class ParityHelperTest(unittest.TestCase):
    def test_exact_tokens_and_text_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            reference = directory / "reference.db"
            actual = directory / "actual.db"
            create_result_database(
                reference,
                job_id="cpu",
                memo_id="memo-a",
                token_ids="10,20,30",
                raw_text="example",
                generated_text="example",
            )
            create_result_database(
                actual,
                job_id="candidate",
                memo_id="memo-a",
                token_ids="10,20,30",
                raw_text="example",
                generated_text="example",
            )

            result = compare(reference, actual)

            self.assertEqual(0, result.returncode)
            self.assertIn("parity_passed=True", result.stdout)
            self.assertNotIn("example", result.stdout)

    def test_first_token_mismatch_fails_without_printing_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            reference = directory / "reference.db"
            actual = directory / "actual.db"
            create_result_database(
                reference,
                job_id="cpu",
                memo_id="memo-a",
                token_ids="10,20,30",
                raw_text="private reference",
                generated_text="private reference",
            )
            create_result_database(
                actual,
                job_id="candidate",
                memo_id="memo-a",
                token_ids="10,99,30",
                raw_text="private candidate",
                generated_text="private candidate",
            )

            result = compare(reference, actual)

            self.assertEqual(1, result.returncode)
            self.assertIn("chunk_0_first_token_mismatch=1", result.stdout)
            self.assertIn("parity_passed=False", result.stdout)
            self.assertNotIn("private", result.stdout)

    def test_rejects_job_that_is_not_succeeded_or_complete(self) -> None:
        invalid_jobs = (
            ("reference running", {"status": "RUNNING"}, {}),
            ("candidate running", {}, {"status": "RUNNING"}),
            ("reference incomplete", {"completed_chunks": 0}, {}),
            ("candidate incomplete", {}, {"completed_chunks": 0}),
        )
        for label, reference_overrides, actual_overrides in invalid_jobs:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                reference = directory / "reference.db"
                actual = directory / "actual.db"
                create_result_database(
                    reference,
                    job_id="cpu",
                    memo_id="memo-a",
                    token_ids="10,20,30",
                    raw_text="private payload",
                    generated_text="private payload",
                    **reference_overrides,
                )
                actual_parameters = {
                    "job_id": "candidate",
                    "memo_id": "memo-a",
                    "token_ids": "10,20,30",
                    "raw_text": "private payload",
                    "generated_text": "private payload",
                    **actual_overrides,
                }
                create_result_database(actual, **actual_parameters)

                result = compare(reference, actual)

                self.assertEqual(1, result.returncode)
                self.assertIn("chunk_0_exact_match=True", result.stdout)
                self.assertIn("generated_text_exact_match=True", result.stdout)
                self.assertIn("parity_passed=False", result.stdout)
                self.assertNotIn("private", result.stdout)

    def test_rejects_different_memos_or_chunk_windows(self) -> None:
        invalid_pairs = (
            ("different memo", {}, {"memo_id": "memo-b"}, "same_memo_id=False"),
            (
                "different window",
                {},
                {"window_start_millis": 1_000, "window_end_millis": 29_000},
                "chunk_windows_match=False",
            ),
        )
        for label, reference_overrides, actual_overrides, expected_flag in invalid_pairs:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                reference = directory / "reference.db"
                actual = directory / "actual.db"
                create_result_database(
                    reference,
                    job_id="cpu",
                    memo_id="memo-a",
                    token_ids="10,20,30",
                    raw_text="private payload",
                    generated_text="private payload",
                    **reference_overrides,
                )
                actual_parameters = {
                    "job_id": "candidate",
                    "memo_id": "memo-a",
                    "token_ids": "10,20,30",
                    "raw_text": "private payload",
                    "generated_text": "private payload",
                    **actual_overrides,
                }
                create_result_database(actual, **actual_parameters)

                result = compare(reference, actual)

                self.assertEqual(1, result.returncode)
                self.assertIn(expected_flag, result.stdout)
                self.assertIn("generated_text_exact_match=True", result.stdout)
                self.assertIn("parity_passed=False", result.stdout)
                self.assertNotIn("private", result.stdout)

    def test_rejects_non_cpu_or_missing_reference_fingerprint(self) -> None:
        for label, fingerprint in (
            ("parity guard", GPU_FINGERPRINT),
            ("missing", None),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                reference = directory / "reference.db"
                actual = directory / "actual.db"
                create_result_database(
                    reference,
                    job_id="cpu",
                    memo_id="memo-a",
                    token_ids="10,20,30",
                    raw_text="private payload",
                    generated_text="private payload",
                    fingerprint=fingerprint,
                )
                create_result_database(
                    actual,
                    job_id="candidate",
                    memo_id="memo-a",
                    token_ids="10,20,30",
                    raw_text="private payload",
                    generated_text="private payload",
                )

                result = compare(reference, actual)

                self.assertEqual(1, result.returncode)
                self.assertIn(
                    "reference_cpu_xnnpack_fingerprint=False",
                    result.stdout,
                )
                self.assertIn("chunk_0_exact_match=True", result.stdout)
                self.assertIn("generated_text_exact_match=True", result.stdout)
                self.assertIn("parity_passed=False", result.stdout)
                self.assertNotIn("private", result.stdout)

    def test_rejects_transcript_not_generated_by_compared_job(self) -> None:
        stale_transcripts = (
            (
                "reference stale",
                {"generated_at_millis": 999},
                {},
                "reference_transcript_matches_job=False",
            ),
            (
                "candidate stale",
                {},
                {"generated_at_millis": 999},
                "actual_transcript_matches_job=False",
            ),
        )
        for label, reference_overrides, actual_overrides, expected_flag in stale_transcripts:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                reference = directory / "reference.db"
                actual = directory / "actual.db"
                create_result_database(
                    reference,
                    job_id="cpu",
                    memo_id="memo-a",
                    token_ids="10,20,30",
                    raw_text="private payload",
                    generated_text="private payload",
                    **reference_overrides,
                )
                create_result_database(
                    actual,
                    job_id="candidate",
                    memo_id="memo-a",
                    token_ids="10,20,30",
                    raw_text="private payload",
                    generated_text="private payload",
                    **actual_overrides,
                )

                result = compare(reference, actual)

                self.assertEqual(1, result.returncode)
                self.assertIn(expected_flag, result.stdout)
                self.assertIn("chunk_0_exact_match=True", result.stdout)
                self.assertIn("generated_text_exact_match=True", result.stdout)
                self.assertIn("parity_passed=False", result.stdout)
                self.assertNotIn("private", result.stdout)


if __name__ == "__main__":
    unittest.main()
