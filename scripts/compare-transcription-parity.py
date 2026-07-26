#!/usr/bin/env python3
"""Compare persisted Whisper results without printing transcript content."""

from __future__ import annotations

import argparse
import hashlib
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path


CPU_XNNPACK_RUNTIME = re.compile(r"cpu-xnnpack-[1-9][0-9]*x[1-9][0-9]*")


@dataclass(frozen=True)
class JobResult:
    memo_id: str
    status: str
    total_chunks: int
    completed_chunks: int
    inference_fingerprint: str | None
    updated_at_millis: int


@dataclass(frozen=True)
class ChunkResult:
    chunk_index: int
    memo_id: str
    window_start_millis: int
    window_end_millis: int
    status: str
    token_ids: str
    raw_text: str
    detected_language: str | None
    truncated: int

    @property
    def window(self) -> tuple[int, str, int, int]:
        return (
            self.chunk_index,
            self.memo_id,
            self.window_start_millis,
            self.window_end_millis,
        )

    @property
    def metadata(self) -> tuple[str, str | None, int]:
        return self.raw_text, self.detected_language, self.truncated


@dataclass(frozen=True)
class TranscriptResult:
    generated_text: str
    generated_at_millis: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-db", required=True, type=Path)
    parser.add_argument("--reference-job", required=True)
    parser.add_argument("--actual-db", required=True, type=Path)
    parser.add_argument("--actual-job", required=True)
    return parser.parse_args()


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def job(database: sqlite3.Connection, job_id: str) -> JobResult:
    row = database.execute(
        """
        SELECT memoId, status, totalChunks, completedChunks,
               inferenceFingerprint, updatedAtMillis
        FROM transcription_jobs
        WHERE id = ?
        """,
        (job_id,),
    ).fetchone()
    if row is None:
        raise ValueError("The requested transcription job does not exist.")
    return JobResult(*row)


def chunks(database: sqlite3.Connection, job_id: str) -> list[ChunkResult]:
    rows = database.execute(
        """
        SELECT chunkIndex, memoId, windowStartMillis, windowEndMillis, status,
               tokenIds, rawText, detectedLanguage, truncated
        FROM transcript_chunks
        WHERE jobId = ?
        ORDER BY chunkIndex
        """,
        (job_id,),
    ).fetchall()
    return [ChunkResult(*row) for row in rows]


def generated_transcript(
    database: sqlite3.Connection,
    result_job: JobResult,
) -> TranscriptResult:
    row = database.execute(
        """
        SELECT generatedText, generatedAtMillis
        FROM transcripts
        WHERE memoId = ?
        """,
        (result_job.memo_id,),
    ).fetchone()
    if row is None or row[0] is None:
        raise ValueError("No generated transcript exists for the requested job.")
    return TranscriptResult(*row)


def is_cpu_xnnpack_fingerprint(value: str | None) -> bool:
    if value is None:
        return False
    fields: dict[str, str] = {}
    for part in value.split("|"):
        key, separator, field_value = part.partition("=")
        if not separator or not key or not field_value or key in fields:
            return False
        fields[key] = field_value
    if not {"model", "tokenizer", "pipeline", "runtime"}.issubset(fields):
        return False
    return CPU_XNNPACK_RUNTIME.fullmatch(fields["runtime"]) is not None


def chunks_are_complete(result_job: JobResult, result_chunks: list[ChunkResult]) -> bool:
    return (
        result_job.total_chunks > 0
        and result_job.completed_chunks == result_job.total_chunks
        and len(result_chunks) == result_job.total_chunks
        and [chunk.chunk_index for chunk in result_chunks]
        == list(range(result_job.total_chunks))
        and all(chunk.memo_id == result_job.memo_id for chunk in result_chunks)
        and all(chunk.status == "COMPLETED" for chunk in result_chunks)
    )


def token_list(serialized: str) -> list[int]:
    return [int(value) for value in serialized.split(",") if value]


def first_mismatch(left: list[int], right: list[int]) -> int | None:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def main() -> int:
    args = parse_args()
    reference = sqlite3.connect(f"file:{args.reference_db}?mode=ro", uri=True)
    actual = sqlite3.connect(f"file:{args.actual_db}?mode=ro", uri=True)
    try:
        try:
            reference_job = job(reference, args.reference_job)
            actual_job = job(actual, args.actual_job)
            expected_chunks = chunks(reference, args.reference_job)
            actual_chunks = chunks(actual, args.actual_job)
            expected_transcript = generated_transcript(reference, reference_job)
            actual_transcript = generated_transcript(actual, actual_job)
        except (sqlite3.Error, TypeError, ValueError):
            print("comparison_input_valid=False")
            print("parity_passed=False")
            return 1

        reference_succeeded = reference_job.status == "SUCCEEDED"
        actual_succeeded = actual_job.status == "SUCCEEDED"
        reference_chunks_complete = chunks_are_complete(
            reference_job,
            expected_chunks,
        )
        actual_chunks_complete = chunks_are_complete(actual_job, actual_chunks)
        same_memo = reference_job.memo_id == actual_job.memo_id
        reference_is_cpu = is_cpu_xnnpack_fingerprint(
            reference_job.inference_fingerprint,
        )
        expected_transcript_is_current = (
            expected_transcript.generated_at_millis == reference_job.updated_at_millis
        )
        actual_transcript_is_current = (
            actual_transcript.generated_at_millis == actual_job.updated_at_millis
        )
        windows_match = (
            [chunk.window for chunk in expected_chunks]
            == [chunk.window for chunk in actual_chunks]
        )

        print("comparison_input_valid=True")
        print(f"reference_job_succeeded={reference_succeeded}")
        print(f"actual_job_succeeded={actual_succeeded}")
        print(f"reference_chunks_complete={reference_chunks_complete}")
        print(f"actual_chunks_complete={actual_chunks_complete}")
        print(f"same_memo_id={same_memo}")
        print(f"reference_cpu_xnnpack_fingerprint={reference_is_cpu}")
        print(f"reference_transcript_matches_job={expected_transcript_is_current}")
        print(f"actual_transcript_matches_job={actual_transcript_is_current}")
        print(f"chunk_windows_match={windows_match}")
        print(f"reference_chunks={len(expected_chunks)}")
        print(f"actual_chunks={len(actual_chunks)}")
        matched = all(
            (
                reference_succeeded,
                actual_succeeded,
                reference_chunks_complete,
                actual_chunks_complete,
                same_memo,
                reference_is_cpu,
                expected_transcript_is_current,
                actual_transcript_is_current,
                windows_match,
            )
        )

        for index in range(max(len(expected_chunks), len(actual_chunks))):
            if index >= len(expected_chunks) or index >= len(actual_chunks):
                print(f"chunk_{index}_present_in_both=false")
                matched = False
                continue
            expected = expected_chunks[index]
            observed = actual_chunks[index]
            expected_tokens = token_list(expected.token_ids)
            observed_tokens = token_list(observed.token_ids)
            mismatch = first_mismatch(expected_tokens, observed_tokens)
            chunk_matches = (
                expected.window == observed.window
                and mismatch is None
                and expected.metadata == observed.metadata
            )
            matched = matched and chunk_matches
            print(f"chunk_{index}_token_count_reference={len(expected_tokens)}")
            print(f"chunk_{index}_token_count_actual={len(observed_tokens)}")
            print(f"chunk_{index}_first_token_mismatch={mismatch}")
            print(
                f"chunk_{index}_metadata_match="
                f"{expected.metadata == observed.metadata}",
            )
            print(f"chunk_{index}_exact_match={chunk_matches}")

        expected_text = expected_transcript.generated_text
        observed_text = actual_transcript.generated_text
        text_matches = expected_text == observed_text
        matched = matched and text_matches
        print(f"generated_text_chars_reference={len(expected_text)}")
        print(f"generated_text_chars_actual={len(observed_text)}")
        print(f"generated_text_sha256_reference={digest(expected_text)}")
        print(f"generated_text_sha256_actual={digest(observed_text)}")
        print(f"generated_text_exact_match={text_matches}")
        print(f"parity_passed={matched}")
        return 0 if matched else 1
    finally:
        reference.close()
        actual.close()


if __name__ == "__main__":
    raise SystemExit(main())
