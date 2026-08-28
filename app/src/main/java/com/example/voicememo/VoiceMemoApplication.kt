package com.example.voicememo

import android.app.Activity
import android.app.Application
import android.os.Bundle
import android.util.Log
import com.example.voicememo.data.NewMemo
import com.example.voicememo.data.WaveformState
import com.example.voicememo.recording.RecordingCompletionSink
import com.example.voicememo.recording.RecordingState
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class VoiceMemoApplication : Application(), RecordingCompletionSink {
    private val containerDelegate = lazy { AppContainer(this) }
    val container: AppContainer by containerDelegate

    private var startedActivityCount = 0

    override fun onCreate() {
        super.onCreate()
        registerActivityLifecycleCallbacks(
            object : ActivityLifecycleCallbacks {
                override fun onActivityStarted(activity: Activity) {
                    startedActivityCount += 1
                    if (startedActivityCount == 1) {
                        container.whisperEngineCache.setAppActive(true)
                    }
                }

                override fun onActivityStopped(activity: Activity) {
                    startedActivityCount = (startedActivityCount - 1).coerceAtLeast(0)
                    if (startedActivityCount == 0 && containerDelegate.isInitialized()) {
                        container.whisperEngineCache.setAppActive(false)
                    }
                }

                override fun onActivityCreated(activity: Activity, state: Bundle?) = Unit
                override fun onActivityResumed(activity: Activity) = Unit
                override fun onActivityPaused(activity: Activity) = Unit
                override fun onActivitySaveInstanceState(activity: Activity, state: Bundle) = Unit
                override fun onActivityDestroyed(activity: Activity) = Unit
            },
        )
    }

    override suspend fun persistCompletedRecording(recording: RecordingState.Completed) {
        val title = SimpleDateFormat(
            "MMM d 'at' h:mm a",
            Locale.getDefault(),
        ).format(Date(recording.createdAtEpochMillis))
        container.memoRepository.createMemo(
            NewMemo(
                id = recording.memoId,
                title = title,
                createdAtMillis = recording.createdAtEpochMillis,
                durationMillis = recording.durationMillis,
                audioPath = recording.audioPath,
                waveformPath = recording.waveformPath,
                waveformState = WaveformState.READY,
                titleEditedByUser = false,
            ),
        )
        recording.preTranscriptionConfig?.let { config ->
            runCatching {
                container.transcriptionCoordinator.enqueue(
                    memoId = recording.memoId,
                    languageCode = null,
                    timestampsEnabled = config.timestampsEnabled,
                    voiceActivityDetectionEnabled =
                        config.voiceActivityDetectionEnabled,
                    gpuAccelerationEnabled = config.gpuAccelerationEnabled,
                    transcriptionModelId =
                        com.example.voicememo.transcription.TranscriptionModel
                            .WHISPER_LARGE_V3_TURBO.id,
                    reuseRecordingStageResults = true,
                )
            }.onFailure { failure ->
                Log.e(
                    TAG,
                    "Recording was saved, but automatic transcription could not be queued.",
                    failure,
                )
            }
        }
    }

    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        if (
            level >= RUNNING_LOW_MEMORY_LEVEL &&
            containerDelegate.isInitialized()
        ) {
            container.whisperEngineCache.evictIdle()
        }
    }

    override fun onLowMemory() {
        if (containerDelegate.isInitialized()) container.whisperEngineCache.evictIdle()
        super.onLowMemory()
    }

    override fun onTerminate() {
        if (containerDelegate.isInitialized()) containerDelegate.value.close()
        super.onTerminate()
    }

    private companion object {
        const val TAG = "VoiceMemoApplication"
        // ComponentCallbacks2.TRIM_MEMORY_RUNNING_LOW without referencing the
        // deprecated constant on newer Android SDKs.
        const val RUNNING_LOW_MEMORY_LEVEL = 10
    }
}
