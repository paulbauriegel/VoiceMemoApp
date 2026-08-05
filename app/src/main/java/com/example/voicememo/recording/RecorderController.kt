package com.example.voicememo.recording

import kotlinx.coroutines.flow.StateFlow

/**
 * Process-local control surface for the foreground recording service.
 *
 * The service owns the microphone and WAV file so recording survives activity recreation.
 */
interface RecorderController {
    val state: StateFlow<RecordingState>

    fun start()
    fun pause()
    fun resume()
    fun stop()
}

data class WaveformBucket(
    val minimum: Short,
    val maximum: Short,
)

sealed interface RecordingState {
    data object Idle : RecordingState
    data object Starting : RecordingState

    data class Recording(
        val memoId: String,
        val elapsedMillis: Long,
        val waveform: List<WaveformBucket>,
    ) : RecordingState

    data class Paused(
        val memoId: String,
        val elapsedMillis: Long,
        val waveform: List<WaveformBucket>,
        val interrupted: Boolean,
    ) : RecordingState

    data class Finalizing(
        val memoId: String,
        val elapsedMillis: Long,
    ) : RecordingState

    data class Completed(
        val memoId: String,
        val audioPath: String,
        val waveformPath: String,
        val durationMillis: Long,
        val sampleRate: Int,
        val createdAtEpochMillis: Long,
        val preTranscriptionConfig: RecordingTranscriptionConfig? = null,
    ) : RecordingState

    data class Failed(
        val memoId: String?,
        val message: String,
        val recoverableAudioPath: String? = null,
    ) : RecordingState
}

data class RecordingTranscriptionConfig(
    val timestampsEnabled: Boolean,
    val voiceActivityDetectionEnabled: Boolean,
    val gpuAccelerationEnabled: Boolean,
)

internal object RecordingRuntime {
    val state = kotlinx.coroutines.flow.MutableStateFlow<RecordingState>(RecordingState.Idle)
}

/**
 * Implemented by the Application so a completed memo is persisted even when no activity exists.
 */
interface RecordingCompletionSink {
    suspend fun persistCompletedRecording(recording: RecordingState.Completed)
}
