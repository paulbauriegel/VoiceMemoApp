package com.example.voicememo.transcription

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

class WhisperSequentialHybridDiagnosticTest {
    @Test
    fun exactTokensAndTermination_passParityAndKeepOnlyTokenIds() {
        val result = WhisperSequentialHybridDiagnostic.compare(
            referenceTokens = listOf(407, 5_616, 390),
            referenceTruncated = false,
            candidateTokens = listOf(407, 5_616, 390),
            candidateTruncated = false,
            timings = timings(),
        )

        assertEquals(listOf(407, 5_616, 390), result.tokenIds)
        assertEquals(
            "ae17847fecb8471ec5de9e99dbcfe5908c09c3b342d95aa6206599b9f5322338",
            result.tokenSha256,
        )
        assertNull(result.firstTokenMismatch)
        assertTrue(result.exactTokenMatch)
        assertTrue(result.exactTerminationMatch)
        assertTrue(result.exactCpuResultMatch)
        assertEquals(18L, result.timings.inferenceComponentMillis)
        assertEquals(28L, result.timings.sequentialComponentMillis)
    }

    @Test
    fun firstDifferentToken_isReportedWithoutTranscriptText() {
        val result = WhisperSequentialHybridDiagnostic.compare(
            referenceTokens = listOf(407, 5_616, 390),
            referenceTruncated = false,
            candidateTokens = listOf(407, 8_308, 390),
            candidateTruncated = false,
            timings = timings(),
        )

        assertEquals(1, result.firstTokenMismatch)
        assertFalse(result.exactTokenMatch)
        assertTrue(result.exactTerminationMatch)
        assertFalse(result.exactCpuResultMatch)
    }

    @Test
    fun differentLengthAndTermination_bothFailParity() {
        val result = WhisperSequentialHybridDiagnostic.compare(
            referenceTokens = listOf(407, 5_616, 390),
            referenceTruncated = false,
            candidateTokens = listOf(407, 5_616),
            candidateTruncated = true,
            timings = timings(),
        )

        assertEquals(2, result.firstTokenMismatch)
        assertFalse(result.exactTokenMatch)
        assertFalse(result.exactTerminationMatch)
        assertFalse(result.exactCpuResultMatch)
    }

    @Test
    fun negativeTiming_isRejected() {
        try {
            WhisperSequentialHybridDiagnostic.Timings(
                gpuEncodeMillis = -1L,
                gpuModelCloseMillis = 0L,
                cpuDecoderModelLoadMillis = 0L,
                cpuDecodeMillis = 0L,
            )
            fail("Negative diagnostic timings must be rejected.")
        } catch (_: IllegalArgumentException) {
            // Expected.
        }
    }

    private fun timings() = WhisperSequentialHybridDiagnostic.Timings(
        gpuEncodeMillis = 5L,
        gpuModelCloseMillis = 3L,
        cpuDecoderModelLoadMillis = 7L,
        cpuDecodeMillis = 13L,
    )
}
