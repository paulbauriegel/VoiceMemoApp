package com.example.voicememo.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class TranscriptionResumePolicyTest {
    @Test
    fun matchingFingerprintPreservesCompletedChunkProgress() {
        val current = job(inferenceFingerprint = "model-a/runtime-a")

        val preparation = planTranscriptionResume(
            current = current,
            inferenceFingerprint = "model-a/runtime-a",
            updatedAtMillis = 9_999L,
        )

        assertFalse(preparation.discardDerivedChunks)
        assertSame(current, preparation.job)
        assertEquals(2, preparation.job.completedChunks)
        assertEquals(2, preparation.job.currentChunkIndex)
        assertEquals(37, preparation.job.currentToken)
        assertEquals("de", preparation.job.detectedLanguage)
        assertEquals("quiet speech warning", preparation.job.warning)
        assertEquals(2_000L, preparation.job.updatedAtMillis)
    }

    @Test
    fun changedFingerprintDiscardsDerivedProgressButPreservesDurableJobIdentity() {
        val current = job(inferenceFingerprint = "model-a/runtime-a")

        val preparation = planTranscriptionResume(
            current = current,
            inferenceFingerprint = "model-a/runtime-b",
            updatedAtMillis = 9_999L,
        )

        assertTrue(preparation.discardDerivedChunks)
        assertEquals(current.id, preparation.job.id)
        assertEquals(current.memoId, preparation.job.memoId)
        assertEquals(current.workId, preparation.job.workId)
        assertEquals(current.totalChunks, preparation.job.totalChunks)
        assertEquals(current.requestedLanguage, preparation.job.requestedLanguage)
        assertEquals(current.status, preparation.job.status)
        assertEquals(current.phase, preparation.job.phase)
        assertEquals("model-a/runtime-b", preparation.job.inferenceFingerprint)
        assertEquals(0, preparation.job.completedChunks)
        assertNull(preparation.job.currentChunkIndex)
        assertEquals(0, preparation.job.currentToken)
        assertNull(preparation.job.detectedLanguage)
        assertNull(preparation.job.warning)
        assertEquals(9_999L, preparation.job.updatedAtMillis)
    }

    @Test
    fun legacyNullFingerprintCannotReuseExistingChunks() {
        val preparation = planTranscriptionResume(
            current = job(inferenceFingerprint = null),
            inferenceFingerprint = "model-a/runtime-a",
            updatedAtMillis = 9_999L,
        )

        assertTrue(preparation.discardDerivedChunks)
        assertEquals("model-a/runtime-a", preparation.job.inferenceFingerprint)
        assertEquals(0, preparation.job.completedChunks)
    }

    @Test
    fun blankFingerprintIsRejectedBeforeAnyResumeDecision() {
        assertThrows(IllegalArgumentException::class.java) {
            planTranscriptionResume(
                current = job(inferenceFingerprint = "model-a/runtime-a"),
                inferenceFingerprint = " ",
                updatedAtMillis = 9_999L,
            )
        }
    }

    private fun job(inferenceFingerprint: String?) = TranscriptionJob(
        id = "job-id",
        memoId = "memo-id",
        workId = "work-id",
        status = TranscriptionJobStatus.RUNNING,
        phase = TranscriptionPhase.DECODING,
        requestedLanguage = null,
        detectedLanguage = "de",
        totalChunks = 4,
        completedChunks = 2,
        currentChunkIndex = 2,
        currentToken = 37,
        currentTokenLimit = 128,
        errorCode = null,
        errorMessage = null,
        warning = "quiet speech warning",
        inferenceFingerprint = inferenceFingerprint,
        createdAtMillis = 1_000L,
        updatedAtMillis = 2_000L,
    )
}
