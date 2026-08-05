package com.example.voicememo.work

import com.example.voicememo.data.TranscriptionJobStatus
import com.example.voicememo.transcription.TranscriptionModel
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class TranscriptionInterruptionPolicyTest {
    @Test
    fun systemInterruptionRemainsResumable() {
        assertFalse(shouldMarkTranscriptionCancelled(TranscriptionJobStatus.QUEUED))
        assertFalse(shouldMarkTranscriptionCancelled(TranscriptionJobStatus.RUNNING))
        assertFalse(
            shouldMarkTranscriptionCancelled(
                TranscriptionJobStatus.PAUSED_FOR_RECORDING,
            ),
        )
    }

    @Test
    fun explicitCancellationRemainsTerminal() {
        assertTrue(
            shouldMarkTranscriptionCancelled(
                TranscriptionJobStatus.CANCEL_REQUESTED,
            ),
        )
        assertTrue(shouldMarkTranscriptionCancelled(TranscriptionJobStatus.CANCELLED))
    }

    @Test
    fun retryDiscoversStagedWhisperResultsWithoutRequestFlag() {
        assertTrue(
            shouldUsePreTranscribedWindows(
                explicitlyRequested = false,
                transcriptionModel = TranscriptionModel.WHISPER_LARGE_V3_TURBO,
                firstStagedResultExists = true,
            ),
        )
        assertFalse(
            shouldUsePreTranscribedWindows(
                explicitlyRequested = false,
                transcriptionModel = TranscriptionModel.WHISPER_LARGE_V3_TURBO,
                firstStagedResultExists = false,
            ),
        )
        assertFalse(
            shouldUsePreTranscribedWindows(
                explicitlyRequested = true,
                transcriptionModel = TranscriptionModel.PARAKEET_TDT_0_6B_V3,
                firstStagedResultExists = true,
            ),
        )
    }
}
