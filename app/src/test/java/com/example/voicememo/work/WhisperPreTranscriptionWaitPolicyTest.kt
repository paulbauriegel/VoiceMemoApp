package com.example.voicememo.work

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class WhisperPreTranscriptionWaitPolicyTest {
    @Test
    fun recordingStopHandsOffRemainingWindowsToFinalTranscription() {
        assertTrue(
            shouldContinuePreTranscription(
                workerStopped = false,
                recordingFinished = false,
            ),
        )
        assertFalse(
            shouldContinuePreTranscription(
                workerStopped = false,
                recordingFinished = true,
            ),
        )
        assertFalse(
            shouldContinuePreTranscription(
                workerStopped = true,
                recordingFinished = false,
            ),
        )
    }

    @Test
    fun activeMatchingRecordingCanKeepProducingWindows() {
        assertTrue(
            shouldWaitForPreTranscriptionWindow(
                matchingRecordingActive = true,
                missingForMillis =
                    WhisperPreTranscriptionWorker.ORPHAN_RECOVERY_GRACE_MILLIS * 10,
            ),
        )
    }

    @Test
    fun recoveredOrphanReleasesQueueAfterGracePeriod() {
        assertTrue(
            shouldWaitForPreTranscriptionWindow(
                matchingRecordingActive = false,
                missingForMillis =
                    WhisperPreTranscriptionWorker.ORPHAN_RECOVERY_GRACE_MILLIS - 1,
            ),
        )
        assertFalse(
            shouldWaitForPreTranscriptionWindow(
                matchingRecordingActive = false,
                missingForMillis =
                    WhisperPreTranscriptionWorker.ORPHAN_RECOVERY_GRACE_MILLIS,
            ),
        )
    }
}
