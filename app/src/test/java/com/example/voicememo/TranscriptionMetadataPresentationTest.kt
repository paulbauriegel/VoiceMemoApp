package com.example.voicememo

import com.example.voicememo.data.TranscriptionJob
import com.example.voicememo.data.TranscriptionJobStatus
import com.example.voicememo.data.TranscriptionPhase
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class TranscriptionMetadataPresentationTest {
    @Test
    fun onlyRunningMetadataPhaseShowsAudioProgress() {
        val running = job(
            status = TranscriptionJobStatus.RUNNING,
            phase = TranscriptionPhase.GENERATING_METADATA,
        )

        assertTrue(running.isGeneratingRecordingMetadata())
        assertFalse(
            running.copy(status = TranscriptionJobStatus.SUCCEEDED)
                .isGeneratingRecordingMetadata(),
        )
        assertFalse(
            running.copy(phase = TranscriptionPhase.DECODING)
                .isGeneratingRecordingMetadata(),
        )
        assertFalse((null as TranscriptionJob?).isGeneratingRecordingMetadata())
    }

    private fun job(
        status: TranscriptionJobStatus,
        phase: TranscriptionPhase,
    ) = TranscriptionJob(
        id = "job-id",
        memoId = "memo-id",
        workId = null,
        status = status,
        phase = phase,
        requestedLanguage = null,
        detectedLanguage = "en",
        totalChunks = 1,
        completedChunks = 1,
        currentChunkIndex = null,
        currentToken = 0,
        currentTokenLimit = 128,
        errorCode = null,
        errorMessage = null,
        warning = null,
        createdAtMillis = 1_000L,
        updatedAtMillis = 2_000L,
    )
}
