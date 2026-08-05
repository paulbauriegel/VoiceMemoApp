package com.example.voicememo.coordination

import kotlinx.coroutines.async
import kotlinx.coroutines.test.advanceTimeBy
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class InferenceGateTest {
    @Test
    fun waitingCanBeCancelledWhileRecordingRemainsActive() = runTest {
        var cancel = false
        InferenceGate.recordingStarted()
        try {
            val waiting = async {
                InferenceGate.awaitRecordingIdle { cancel }
            }

            runCurrent()
            cancel = true
            advanceTimeBy(100)
            runCurrent()

            assertFalse(waiting.await())
            assertTrue(InferenceGate.isRecording)
        } finally {
            InferenceGate.recordingStopped()
        }
    }

    @Test
    fun waitingCompletesWhenRecordingStops() = runTest {
        InferenceGate.recordingStarted()
        try {
            val waiting = async {
                InferenceGate.awaitRecordingIdle()
            }

            runCurrent()
            InferenceGate.recordingStopped()
            runCurrent()

            assertTrue(waiting.await())
        } finally {
            InferenceGate.recordingStopped()
        }
    }

    @Test
    fun activeRecordingCanAuthorizeOnlyItsOwnPreTranscription() = runTest {
        InferenceGate.recordingStarted("current-memo")
        try {
            assertTrue(
                InferenceGate.awaitRecordingIdle(
                    allowedRecordingId = "current-memo",
                ),
            )
            assertTrue(InferenceGate.isRecording("current-memo"))
            assertFalse(InferenceGate.isRecording("another-memo"))
            assertTrue(InferenceGate.isRecordingOtherThan("another-memo"))
            assertFalse(InferenceGate.isRecordingOtherThan("current-memo"))
        } finally {
            InferenceGate.recordingStopped("current-memo")
        }
    }
}
