package com.example.voicememo.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class TranscriptChunkJobSelectionTest {
    @Test
    fun cancelledRetryDoesNotHideChunksFromSavedTranscript() {
        val successful = job(
            id = "successful",
            status = TranscriptionJobStatus.SUCCEEDED,
            updatedAtMillis = 2_000L,
        )
        val cancelledRetry = job(
            id = "cancelled-retry",
            status = TranscriptionJobStatus.CANCELLED,
            updatedAtMillis = 4_000L,
        )

        assertEquals(
            successful,
            selectTranscriptChunkJob(listOf(successful, cancelledRetry)),
        )
    }

    @Test
    fun activeRetryDoesNotHideChunksFromSavedTranscript() {
        val successful = job(
            id = "successful",
            status = TranscriptionJobStatus.SUCCEEDED,
            updatedAtMillis = 2_000L,
        )
        val runningRetry = job(
            id = "running-retry",
            status = TranscriptionJobStatus.RUNNING,
            updatedAtMillis = 4_000L,
        )

        assertEquals(
            successful,
            selectTranscriptChunkJob(listOf(successful, runningRetry)),
        )
    }

    @Test
    fun newerSuccessfulJobReplacesEarlierTimedChunks() {
        val earlier = job(
            id = "earlier",
            status = TranscriptionJobStatus.SUCCEEDED,
            updatedAtMillis = 2_000L,
        )
        val newer = job(
            id = "newer",
            status = TranscriptionJobStatus.SUCCEEDED,
            updatedAtMillis = 4_000L,
        )

        assertEquals(newer, selectTranscriptChunkJob(listOf(earlier, newer)))
    }

    @Test
    fun noSuccessfulJobHasNoSavedTimedChunks() {
        assertNull(
            selectTranscriptChunkJob(
                listOf(
                    job(
                        id = "cancelled",
                        status = TranscriptionJobStatus.CANCELLED,
                        updatedAtMillis = 2_000L,
                    ),
                ),
            ),
        )
    }

    private fun job(
        id: String,
        status: TranscriptionJobStatus,
        updatedAtMillis: Long,
    ) = TranscriptionJob(
        id = id,
        memoId = "memo-id",
        workId = null,
        status = status,
        phase = when (status) {
            TranscriptionJobStatus.SUCCEEDED -> TranscriptionPhase.COMPLETE
            TranscriptionJobStatus.RUNNING -> TranscriptionPhase.DECODING
            else -> TranscriptionPhase.PAUSED
        },
        requestedLanguage = null,
        detectedLanguage = "en",
        totalChunks = 1,
        completedChunks = if (status == TranscriptionJobStatus.SUCCEEDED) 1 else 0,
        currentChunkIndex = null,
        currentToken = 0,
        currentTokenLimit = 128,
        errorCode = null,
        errorMessage = null,
        warning = null,
        createdAtMillis = updatedAtMillis - 1_000L,
        updatedAtMillis = updatedAtMillis,
    )
}
