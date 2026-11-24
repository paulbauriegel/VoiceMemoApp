package com.example.voicememo.work

import android.app.ActivityManager
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.content.pm.ServiceInfo
import android.os.SystemClock
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.work.CoroutineWorker
import androidx.work.ForegroundInfo
import androidx.work.WorkerParameters
import com.example.voicememo.BuildConfig
import com.example.voicememo.VoiceMemoApplication
import com.example.voicememo.coordination.InferenceGate
import com.example.voicememo.transcription.LiteRtWhisperBackendFactory
import com.example.voicememo.transcription.LiteRtFullGpuWhisperEngine
import com.example.voicememo.transcription.LiteRtWhisperEngine
import com.example.voicememo.transcription.LongFormTranscriber
import com.example.voicememo.transcription.ParakeetInferenceFingerprint
import com.example.voicememo.transcription.openParakeetEngineSet
import com.example.voicememo.transcription.SileroVoiceActivityDetector
import com.example.voicememo.transcription.StreamingWavChunkSource
import com.example.voicememo.transcription.TranscriptionModel
import com.example.voicememo.transcription.WhisperBackendRequest
import com.example.voicememo.transcription.WhisperCancelledException
import com.example.voicememo.transcription.WhisperEngine
import com.example.voicememo.transcription.WhisperEngineCache
import com.example.voicememo.transcription.WhisperBackendSelection
import com.example.voicememo.transcription.WhisperInferenceFingerprint
import com.example.voicememo.transcription.WhisperLanguage
import com.example.voicememo.transcription.WhisperAssets
import com.example.voicememo.transcription.windowConfig
import java.io.File
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.delay
import kotlinx.coroutines.runInterruptible
import kotlinx.coroutines.withContext
import kotlinx.coroutines.Dispatchers

/**
 * Serial WorkManager stage that consumes immutable model windows (Whisper or
 * Parakeet) while the recorder continues writing the authoritative memo WAV.
 *
 * It shares the same unique work chain as normal transcription, so two model
 * instances are never opened for separate memos.
 */
class WhisperPreTranscriptionWorker(
    appContext: Context,
    params: WorkerParameters,
) : CoroutineWorker(appContext, params) {
    private val engineCache =
        (appContext.applicationContext as VoiceMemoApplication).container.whisperEngineCache
    private val notificationManager =
        appContext.getSystemService(NotificationManager::class.java)

    override suspend fun doWork(): Result {
        val memoId = inputData.getString(KEY_MEMO_ID) ?: return Result.failure()
        val memoDirectoryPath =
            inputData.getString(KEY_MEMO_DIRECTORY) ?: return Result.failure()
        val directory = WhisperPreTranscriptionStorage.directory(File(memoDirectoryPath))
        val timestampsEnabled = inputData.getBoolean(KEY_TIMESTAMPS_ENABLED, false)
        val vadEnabled = inputData.getBoolean(KEY_VAD_ENABLED, true)
        val gpuEnabled = inputData.getBoolean(
            KEY_GPU_ACCELERATION_ENABLED,
            BuildConfig.WHISPER_BACKEND_REQUESTED.equals("gpu", ignoreCase = true),
        )
        val model = TranscriptionModel.fromId(
            inputData.getString(KEY_TRANSCRIPTION_MODEL_ID)
                ?: TranscriptionModel.WHISPER_LARGE_V3_TURBO.id,
        )
        val parakeet = model == TranscriptionModel.PARAKEET_TDT_0_6B_V3
        if (applicationContext.getSystemService(ActivityManager::class.java).isLowRamDevice) {
            return Result.success()
        }

        createNotificationChannel()
        setForeground(foregroundInfo("Waiting for a completed audio window"))
        val requestedBackend =
            WhisperBackendRequest.fromGpuAccelerationEnabled(gpuEnabled)
        val fingerprint = if (parakeet) {
            ParakeetInferenceFingerprint.forConfiguration(requestedBackend)
        } else WhisperInferenceFingerprint.forConfiguration(
            requestedBackend = requestedBackend,
            gpuDecoderMode = if (requestedBackend == WhisperBackendRequest.GPU) {
                "gpu"
            } else {
                "disabled"
            },
            workerCount = 1,
            threadsPerWorker = BuildConfig.WHISPER_THREADS_PER_WORKER,
            timestampsEnabled = timestampsEnabled,
            voiceActivityDetectionEnabled = vadEnabled,
        )

        var engineLease: WhisperEngineCache.Lease? = null
        var vad: SileroVoiceActivityDetector? = null
        var index = 0
        var forcedLanguageCode: String? = null
        var missingWindowSinceMillis: Long? = null
        val finishedMarker = WhisperPreTranscriptionStorage.finishedMarker(directory)
        try {
            while (
                shouldContinuePreTranscription(
                    workerStopped = isStopped,
                    recordingFinished = finishedMarker.isFile,
                )
            ) {
                val windowFile = WhisperPreTranscriptionStorage.windowFile(directory, index)
                val completed = WhisperPreTranscriptionStorage.readResult(
                    directory,
                    index,
                    fingerprint,
                )
                if (completed != null) {
                    missingWindowSinceMillis = null
                    if (forcedLanguageCode == null && completed.detectedLanguage != "und") {
                        forcedLanguageCode = completed.detectedLanguage
                    }
                    windowFile.delete()
                    index += 1
                    continue
                }
                if (!windowFile.isFile) {
                    val now = SystemClock.elapsedRealtime()
                    val missingSince = missingWindowSinceMillis ?: now.also {
                        missingWindowSinceMillis = it
                    }
                    if (
                        !shouldWaitForPreTranscriptionWindow(
                            matchingRecordingActive = InferenceGate.isRecording(memoId),
                            missingForMillis = now - missingSince,
                        )
                    ) {
                        Log.w(
                            TAG,
                            "Recording $memoId is no longer active and has no completion marker; " +
                                "releasing the transcription queue.",
                        )
                        break
                    }
                    delay(POLL_MILLIS)
                    continue
                }
                missingWindowSinceMillis = null

                if (InferenceGate.isRecordingOtherThan(memoId)) {
                    engineLease?.close()
                    engineLease = null
                    vad?.close()
                    vad = null
                }
                if (!InferenceGate.awaitRecordingIdle(allowedRecordingId = memoId) { isStopped }) {
                    throw WhisperCancelledException()
                }
                if (engineLease == null) {
                    setForeground(foregroundInfo("Loading ${model.displayName} for recording"))
                    engineLease = engineCache.acquire(
                        WhisperEngineCache.Key(
                            model = model,
                            requestedBackend = requestedBackend,
                            workerCount = 1,
                            threadsPerWorker = BuildConfig.WHISPER_THREADS_PER_WORKER,
                            timestampsEnabled = timestampsEnabled,
                        ),
                    ) {
                        if (parakeet) {
                            openParakeetEngineSet(applicationContext, requestedBackend)
                        } else {
                            openEngineSet(requestedBackend, timestampsEnabled)
                        }
                    }
                    if (vadEnabled && !parakeet) {
                        vad = runCatching {
                            SileroVoiceActivityDetector.open(applicationContext)
                        }.getOrNull()
                    }
                }

                setForeground(foregroundInfo("Transcribing recorded window ${index + 1}"))
                val result = withContext(Dispatchers.Default) {
                    StreamingWavChunkSource.open(
                        file = windowFile,
                        windowConfig = model.windowConfig(),
                    ).use { source ->
                        val shouldCancel = { isStopped || finishedMarker.isFile }
                        val audio = source.readChunk(
                            index = 0,
                            shouldCancel = shouldCancel,
                        ).samples
                        val engine = checkNotNull(engineLease).engines.single()
                        if (parakeet) {
                            // Same call the post-recording worker makes per window.
                            engine.transcribeChunk(audio, WhisperLanguage.Auto, shouldCancel)
                        } else {
                            LongFormTranscriber(
                                engine = engine,
                                voiceActivityDetector = vad,
                            ).transcribe(
                                samples16Khz = audio,
                                language = forcedLanguageCode
                                    ?.let(WhisperLanguage::Forced)
                                    ?: WhisperLanguage.Auto,
                                shouldCancel = shouldCancel,
                            ).chunks.single().result
                        }
                    }
                }
                if (forcedLanguageCode == null && result.detectedLanguage != "und") {
                    forcedLanguageCode = result.detectedLanguage
                }
                WhisperPreTranscriptionStorage.writeResult(
                    directory = directory,
                    index = index,
                    fingerprint = fingerprint,
                    result = result,
                    includeTimedWords = parakeet,
                )
                if (!windowFile.delete()) {
                    windowFile.deleteOnExit()
                }
                index += 1
            }
            return Result.success()
        } catch (_: WhisperCancelledException) {
            return Result.success()
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (failure: Throwable) {
            // This stage is only an optimization. Never poison the FIFO chain:
            // the normal post-recording worker can still transcribe every
            // window from the authoritative WAV.
            Log.e(TAG, "Whisper pre-transcription stopped; final work will recover.", failure)
            return Result.success()
        } finally {
            runInterruptible { engineLease?.close() }
            vad?.close()
        }
    }

    private fun openEngineSet(
        requestedBackend: WhisperBackendRequest,
        timestampsEnabled: Boolean,
    ): WhisperEngineCache.EngineSet {
        var selection = if (requestedBackend == WhisperBackendRequest.GPU) {
            WhisperBackendSelection.pending(WhisperBackendRequest.GPU)
        } else {
            WhisperBackendSelection.cpuRequested()
        }
        val engine: WhisperEngine = if (requestedBackend == WhisperBackendRequest.GPU) {
            LiteRtWhisperBackendFactory.openExperimentalGpuWithCpuFallback(
                context = applicationContext,
                cpuThreadCount = BuildConfig.WHISPER_THREADS_PER_WORKER,
                useFullGpuDecoder = true,
                allowSourceQuantizedFcConvOps = false,
                timestampsEnabled = timestampsEnabled,
                onBackendSelectionChanged = { selection = it },
            )
        } else {
            if (WhisperAssets.SPLIT_MODEL_PRODUCTION) {
                LiteRtFullGpuWhisperEngine.openProductionSplit(
                    context = applicationContext,
                    cpuOnly = true,
                    timestampsEnabled = timestampsEnabled,
                )
            } else {
                LiteRtWhisperEngine.open(
                    context = applicationContext,
                    threadCount = BuildConfig.WHISPER_THREADS_PER_WORKER,
                    timestampsEnabled = timestampsEnabled,
                )
            }
        }
        return WhisperEngineCache.EngineSet(listOf(engine), selection)
    }

    private fun foregroundInfo(text: String): ForegroundInfo {
        val notification = NotificationCompat.Builder(applicationContext, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setContentTitle("Preparing transcription")
            .setContentText(text)
            .setOnlyAlertOnce(true)
            .setOngoing(true)
            .build()
        return ForegroundInfo(
            NOTIFICATION_ID,
            notification,
            ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC,
        )
    }

    private fun createNotificationChannel() {
        notificationManager.createNotificationChannel(
            NotificationChannel(
                CHANNEL_ID,
                "Background transcription",
                NotificationManager.IMPORTANCE_LOW,
            ),
        )
    }

    companion object {
        const val KEY_MEMO_ID = "pretranscription_memo_id"
        const val KEY_MEMO_DIRECTORY = "pretranscription_memo_directory"
        const val KEY_TIMESTAMPS_ENABLED = "pretranscription_timestamps"
        const val KEY_VAD_ENABLED = "pretranscription_vad"
        const val KEY_GPU_ACCELERATION_ENABLED = "pretranscription_gpu"
        const val KEY_TRANSCRIPTION_MODEL_ID = "pretranscription_model"

        private const val POLL_MILLIS = 250L
        internal const val ORPHAN_RECOVERY_GRACE_MILLIS = 2_000L
        private const val CHANNEL_ID = "transcription"
        private const val NOTIFICATION_ID = 4_103
        private const val TAG = "WhisperPreTranscription"
    }
}

internal fun shouldWaitForPreTranscriptionWindow(
    matchingRecordingActive: Boolean,
    missingForMillis: Long,
): Boolean =
    matchingRecordingActive ||
        missingForMillis < WhisperPreTranscriptionWorker.ORPHAN_RECOVERY_GRACE_MILLIS

internal fun shouldContinuePreTranscription(
    workerStopped: Boolean,
    recordingFinished: Boolean,
): Boolean =
    !workerStopped && !recordingFinished
