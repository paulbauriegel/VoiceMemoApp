package com.example.voicememo.coordination

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.filter
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.withTimeoutOrNull

/**
 * Recording has priority over inference. Workers check this gate between Whisper chunks.
 */
object InferenceGate {
    private val recording = MutableStateFlow(false)

    fun recordingStarted() {
        recording.value = true
    }

    fun recordingStopped() {
        recording.value = false
    }

    /**
     * Waits until recording releases the inference gate while periodically
     * checking worker cancellation. The polling matters when this is called
     * from a synchronous inference callback whose `runBlocking` scope is not a
     * child of the WorkManager coroutine.
     *
     * @return false when [shouldCancel] requested cancellation first.
     */
    suspend fun awaitRecordingIdle(
        shouldCancel: () -> Boolean = { false },
    ): Boolean {
        while (recording.value) {
            if (shouldCancel()) return false
            withTimeoutOrNull(CANCELLATION_POLL_MILLIS) {
                recording.filter { active -> !active }.first()
            }
        }
        return !shouldCancel()
    }

    val isRecording: Boolean
        get() = recording.value

    private const val CANCELLATION_POLL_MILLIS = 100L
}
