package com.example.voicememo.coordination

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.filter
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.withTimeoutOrNull

/**
 * Recording has priority over inference. Workers check this gate between Whisper chunks.
 */
object InferenceGate {
    private val recordingId = MutableStateFlow<String?>(null)

    fun recordingStarted(memoId: String = LEGACY_RECORDING_ID) {
        recordingId.value = memoId
    }

    fun recordingStopped(memoId: String? = null) {
        if (memoId == null || recordingId.value == memoId) {
            recordingId.value = null
        }
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
        allowedRecordingId: String? = null,
        shouldCancel: () -> Boolean = { false },
    ): Boolean {
        while (
            recordingId.value != null &&
            recordingId.value != allowedRecordingId
        ) {
            if (shouldCancel()) return false
            withTimeoutOrNull(CANCELLATION_POLL_MILLIS) {
                recordingId.filter { active -> active == null || active == allowedRecordingId }
                    .first()
            }
        }
        return !shouldCancel()
    }

    val isRecording: Boolean
        get() = recordingId.value != null

    fun isRecording(memoId: String): Boolean =
        recordingId.value == memoId

    fun isRecordingOtherThan(memoId: String): Boolean =
        recordingId.value?.let { it != memoId } == true

    private const val CANCELLATION_POLL_MILLIS = 100L
    private const val LEGACY_RECORDING_ID = "recording"
}
