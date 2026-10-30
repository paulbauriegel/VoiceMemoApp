package com.example.voicememo.work

import android.app.ActivityManager
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.PowerManager
import android.os.SystemClock
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.work.CoroutineWorker
import androidx.work.Data
import androidx.work.ForegroundInfo
import androidx.work.WorkerParameters
import com.example.voicememo.BuildConfig
import com.example.voicememo.R
import com.example.voicememo.VoiceMemoApplication
import com.example.voicememo.coordination.InferenceGate
import com.example.voicememo.data.MemoRepository
import com.example.voicememo.data.TranscriptChunk
import com.example.voicememo.data.TranscriptChunkStatus
import com.example.voicememo.data.TranscriptSegments
import com.example.voicememo.data.TranscriptionJob
import com.example.voicememo.data.TranscriptionJobStatus
import com.example.voicememo.data.TranscriptionPhase
import com.example.voicememo.transcription.AudioChunkWindow
import com.example.voicememo.transcription.GraniteRecordingMetadataGenerator
import com.example.voicememo.transcription.LiteRtWhisperBackendFactory
import com.example.voicememo.transcription.LiteRtFullGpuWhisperEngine
import com.example.voicememo.transcription.LiteRtWhisperEngine
import com.example.voicememo.transcription.LiteRtParakeetEngine
import com.example.voicememo.transcription.LongFormChunk
import com.example.voicememo.transcription.LongFormTranscriber
import com.example.voicememo.transcription.SileroVoiceActivityDetector
import com.example.voicememo.transcription.StreamingWavChunkSource
import com.example.voicememo.transcription.WhisperBackendRequest
import com.example.voicememo.transcription.WhisperBackendSelection
import com.example.voicememo.transcription.WhisperCancelledException
import com.example.voicememo.transcription.WhisperChunkResult
import com.example.voicememo.transcription.WhisperEngineCache
import com.example.voicememo.transcription.WhisperEnginePoolLoader
import com.example.voicememo.transcription.WhisperFeatureExtractor
import com.example.voicememo.transcription.WhisperInferenceFingerprint
import com.example.voicememo.transcription.WhisperLanguage
import com.example.voicememo.transcription.WhisperAssets
import com.example.voicememo.transcription.WhisperTokenizer
import com.example.voicememo.transcription.ParakeetInferenceFingerprint
import com.example.voicememo.transcription.TranscriptionModel
import com.example.voicememo.transcription.windowConfig
import java.io.File
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext

class TranscriptionWorker(
    appContext: Context,
    params: WorkerParameters,
) : CoroutineWorker(appContext, params) {
    private val container = (appContext.applicationContext as VoiceMemoApplication).container
    private val repository: MemoRepository = container.memoRepository
    private val notificationManager = appContext.getSystemService(NotificationManager::class.java)

    override suspend fun doWork(): Result {
        val jobId = inputData.getString(KEY_JOB_ID) ?: return Result.failure(
            failureData("MISSING_JOB", "Transcription job ID is missing."),
        )
        val memoId = inputData.getString(KEY_MEMO_ID) ?: return Result.failure(
            failureData("MISSING_MEMO", "Memo ID is missing."),
        )
        val timestampsEnabled = inputData.getBoolean(KEY_TIMESTAMPS_ENABLED, false)
        val voiceActivityDetectionEnabled = inputData.getBoolean(KEY_VAD_ENABLED, true)
        val gpuAccelerationEnabled = inputData.getBoolean(
            KEY_GPU_ACCELERATION_ENABLED,
            BuildConfig.WHISPER_BACKEND_REQUESTED.equals("gpu", ignoreCase = true),
        )
        val recordingStageResultsRequested =
            inputData.getBoolean(KEY_USE_PRETRANSCRIBED_WINDOWS, false)
        var job = repository.getTranscriptionJob(jobId) ?: return Result.failure(
            failureData("UNKNOWN_JOB", "The transcription job no longer exists."),
        )
        val transcriptionModel = TranscriptionModel.fromId(job.transcriptionModelId)
        val memo = repository.getMemo(memoId) ?: return Result.failure(
            failureData("UNKNOWN_MEMO", "The voice memo no longer exists."),
        )
        val stagedPreTranscriptionDirectory = WhisperPreTranscriptionStorage.directory(
            File(memo.audioPath).parentFile
                ?: return Result.failure(
                    failureData("INVALID_AUDIO", "Memo audio has no parent directory."),
                ),
        )
        val usePreTranscribedWindows = shouldUsePreTranscribedWindows(
            reuseRecordingStageResults = recordingStageResultsRequested,
            transcriptionModel = transcriptionModel,
        )
        var benchmark: TranscriptionBenchmark? = null

        if (applicationContext.getSystemService(ActivityManager::class.java).isLowRamDevice) {
            val message =
                "This device is designated low-RAM by Android and cannot safely load the bundled transcription models."
            failJob(job, "LOW_RAM_DEVICE", message)
            return Result.failure(failureData("LOW_RAM_DEVICE", message))
        }

        createNotificationChannel()
        setForeground(foregroundInfo("Preparing audio", progress = null))
        val requestedBackend =
            WhisperBackendRequest.fromGpuAccelerationEnabled(gpuAccelerationEnabled)
        val gpuDecoderMode =
            if (requestedBackend == WhisperBackendRequest.GPU) "gpu" else "disabled"
        val workerCount = transcriptionWorkerCount(
            transcriptionModel = transcriptionModel,
            requestedBackend = requestedBackend,
            usePreTranscribedWindows = usePreTranscribedWindows,
            configuredCpuWorkerCount = BuildConfig.WHISPER_WORKER_COUNT,
        )
        val inferenceFingerprint = when (transcriptionModel) {
            TranscriptionModel.WHISPER_LARGE_V3_TURBO ->
                WhisperInferenceFingerprint.forConfiguration(
                    requestedBackend = requestedBackend,
                    gpuDecoderMode = gpuDecoderMode,
                    workerCount = workerCount,
                    threadsPerWorker = BuildConfig.WHISPER_THREADS_PER_WORKER,
                    timestampsEnabled = timestampsEnabled,
                    voiceActivityDetectionEnabled = voiceActivityDetectionEnabled,
                )
            TranscriptionModel.PARAKEET_TDT_0_6B_V3 ->
                ParakeetInferenceFingerprint.forConfiguration(requestedBackend)
        }

        try {
            job = updateJob(
                job.copy(
                    status = TranscriptionJobStatus.RUNNING,
                    phase = TranscriptionPhase.PREPARING_AUDIO,
                    errorCode = null,
                    errorMessage = null,
                    updatedAtMillis = System.currentTimeMillis(),
                ),
            )
            val audioSource = withContext(Dispatchers.IO) {
                StreamingWavChunkSource.open(
                    file = File(memo.audioPath),
                    windowConfig = transcriptionModel.windowConfig(),
                )
            }
            val transcript = audioSource.use { source ->
                val windows = List(source.chunkCount, source::window)
                if (windows.isEmpty()) {
                    repository.completeTranscriptionJob(
                        jobId = job.id,
                        mergedText = "",
                        detectedLanguage = null,
                        warning = "No audio samples were available.",
                    )
                    return Result.success()
                }
                job = repository.prepareTranscriptionJobForInference(
                    jobId = job.id,
                    inferenceFingerprint = inferenceFingerprint,
                )
                // Replace the enqueue-time placeholder chunk count before any
                // pre-transcribed chunk is persisted; persistChunk raises
                // completedChunks past a stale totalChunks otherwise.
                job = updateJob(
                    job.copy(
                        totalChunks = windows.size,
                        completedChunks = job.completedChunks.coerceAtMost(windows.size),
                        updatedAtMillis = System.currentTimeMillis(),
                    ),
                )
                if (
                    usePreTranscribedWindows &&
                    transcriptionModel == TranscriptionModel.WHISPER_LARGE_V3_TURBO
                ) {
                    windows.forEach { window ->
                        WhisperPreTranscriptionStorage.readResult(
                            directory = stagedPreTranscriptionDirectory,
                            index = window.index,
                            expectedFingerprint = inferenceFingerprint,
                        )?.let { result ->
                            persistChunk(
                                originalJob = job,
                                memoId = memoId,
                                chunk = LongFormChunk(window = window, result = result),
                            )
                        }
                    }
                }
                val precompletedSegmentCount =
                    repository.completedTranscriptChunks(job.id).size.coerceAtMost(windows.size)
                benchmark = TranscriptionBenchmark(
                    context = applicationContext,
                    audioDurationMillis = source.info.durationMillis,
                    segmentCount = windows.size,
                    precompletedSegmentCount = precompletedSegmentCount,
                    workerCount = workerCount,
                    threadsPerWorker = BuildConfig.WHISPER_THREADS_PER_WORKER,
                )
                job = updateJob(
                    job.copy(
                        totalChunks = windows.size,
                        completedChunks = precompletedSegmentCount,
                        phase = TranscriptionPhase.LOADING_MODEL,
                        updatedAtMillis = System.currentTimeMillis(),
                    ),
                )
                setForeground(
                    foregroundInfo(
                        text = if (workerCount == 1) {
                            "Loading local model"
                        } else {
                            "Loading $workerCount local model workers"
                        },
                        progress = null,
                    ),
                )
                if (!InferenceGate.awaitRecordingIdle { isStopped }) {
                    throw WhisperCancelledException()
                }

                val completed = repository.completedTranscriptChunks(job.id)
                    .associate { chunk -> chunk.chunkIndex to chunk.toLongFormChunk() }
                val requestedLanguage = job.requestedLanguage
                    ?.let(WhisperLanguage::Forced)
                    ?: WhisperLanguage.Auto

                withContext(Dispatchers.Default) {
                    var engineLease: WhisperEngineCache.Lease? = null
                    val voiceActivityDetector =
                        if (
                            voiceActivityDetectionEnabled &&
                            transcriptionModel == TranscriptionModel.WHISPER_LARGE_V3_TURBO
                        ) {
                            try {
                                SileroVoiceActivityDetector.open(applicationContext)
                            } catch (failure: Exception) {
                                // VAD only avoids optional Whisper work. If its
                                // small local model cannot start, retain every
                                // audio window instead of risking lost speech.
                                Log.w(
                                    TAG,
                                    "Silero VAD could not start; retaining all windows.",
                                    failure,
                                )
                                null
                            }
                        } else {
                            null
                        }
                    try {
                        benchmark?.modelLoadingStarted()
                        benchmark?.backendSelectionChanged(
                            if (requestedBackend == WhisperBackendRequest.GPU) {
                                WhisperBackendSelection.pending(
                                    WhisperBackendRequest.GPU,
                                )
                            } else {
                                WhisperBackendSelection.cpuRequested()
                            },
                        )
                        val cacheKey = WhisperEngineCache.Key(
                            model = transcriptionModel,
                            requestedBackend = requestedBackend,
                            workerCount = workerCount,
                            threadsPerWorker = BuildConfig.WHISPER_THREADS_PER_WORKER,
                            timestampsEnabled = timestampsEnabled,
                        )
                        engineLease = container.whisperEngineCache.acquire(cacheKey) {
                            var selectedBackend = if (
                                requestedBackend == WhisperBackendRequest.GPU
                            ) {
                                WhisperBackendSelection.pending(WhisperBackendRequest.GPU)
                            } else {
                                WhisperBackendSelection.cpuRequested()
                            }
                            val opened = WhisperEnginePoolLoader.open(workerCount) {
                                if (
                                    transcriptionModel ==
                                    TranscriptionModel.PARAKEET_TDT_0_6B_V3
                                ) {
                                    LiteRtParakeetEngine.open(
                                        context = applicationContext,
                                        backend = requestedBackend,
                                    ).also {
                                        if (requestedBackend == WhisperBackendRequest.GPU) {
                                            selectedBackend = WhisperBackendSelection.gpuSelected(
                                                com.example.voicememo.transcription
                                                    .WhisperRuntimeBackend
                                                    .GPU_COMPILED_MODEL_AUTO,
                                            )
                                        }
                                    }
                                } else if (requestedBackend == WhisperBackendRequest.GPU) {
                                    LiteRtWhisperBackendFactory
                                        .openExperimentalGpuWithCpuFallback(
                                            context = applicationContext,
                                            cpuThreadCount =
                                                BuildConfig.WHISPER_THREADS_PER_WORKER,
                                            useFullGpuDecoder = true,
                                            allowSourceQuantizedFcConvOps = false,
                                            timestampsEnabled = timestampsEnabled,
                                            onBackendSelectionChanged = { selection ->
                                                selectedBackend = selection
                                            },
                                        )
                                } else if (WhisperAssets.SPLIT_MODEL_PRODUCTION) {
                                    LiteRtFullGpuWhisperEngine.openProductionSplit(
                                        context = applicationContext,
                                        cpuOnly = true,
                                        timestampsEnabled = timestampsEnabled,
                                    )
                                } else {
                                    LiteRtWhisperEngine.open(
                                        context = applicationContext,
                                        threadCount =
                                            BuildConfig.WHISPER_THREADS_PER_WORKER,
                                        timestampsEnabled = timestampsEnabled,
                                    )
                                }
                            }
                            WhisperEngineCache.EngineSet(opened, selectedBackend)
                        }
                        benchmark?.backendSelectionChanged(engineLease.backendSelection)
                        benchmark?.modelLoadingCompleted()
                        val engines = engineLease.engines
                        val progressStateLock = Any()
                        val publishedTokens = IntArray(windows.size) { -1 }
                        val chunkProgress = FloatArray(windows.size)
                        completed.keys.forEach { index ->
                            if (index in chunkProgress.indices) chunkProgress[index] = 1f
                        }
                        benchmark?.transcriptionStarted()
                        val transcriptionStartedAt = SystemClock.elapsedRealtime()
                        val result = LongFormTranscriber(
                            engines = engines,
                            voiceActivityDetector = voiceActivityDetector,
                            lockAutoDetectedLanguage =
                                transcriptionModel ==
                                    TranscriptionModel.WHISPER_LARGE_V3_TURBO,
                            trimOverlappingTokensAtMillis =
                                if (
                                    transcriptionModel ==
                                    TranscriptionModel.PARAKEET_TDT_0_6B_V3
                                ) {
                                    2_500L
                                } else {
                                    null
                                },
                        ).transcribeBatches(
                            windows = windows,
                            loadAudio = { window ->
                                source.readChunk(
                                    index = window.index,
                                    shouldCancel = { isStopped },
                                ).also { chunk ->
                                    check(chunk.window == window) {
                                        "Prepared audio did not match Whisper segment ${window.index}."
                                    }
                                }.samples
                            },
                            language = requestedLanguage,
                            completedChunks = completed,
                            shouldCancel = { isStopped },
                            onChunkPreparing = { window, chunkCount ->
                                runBlocking {
                                    pauseAtSafeBoundaryIfRecording(
                                        jobId = job.id,
                                        resumePhase = TranscriptionPhase.PREPARING_AUDIO,
                                    )
                                    val current =
                                        repository.getTranscriptionJob(job.id) ?: return@runBlocking
                                    updateJob(
                                        current.copy(
                                            status = TranscriptionJobStatus.RUNNING,
                                            phase = TranscriptionPhase.PREPARING_AUDIO,
                                            currentChunkIndex = window.index,
                                            currentToken = 0,
                                            updatedAtMillis = System.currentTimeMillis(),
                                        ),
                                    )
                                    setProgress(
                                        Data.Builder()
                                            .putInt(PROGRESS_CHUNK, window.index)
                                            .putInt(PROGRESS_TOTAL_CHUNKS, chunkCount)
                                            .putInt(PROGRESS_TOKEN, 0)
                                            .putInt(PROGRESS_TOKEN_LIMIT, current.currentTokenLimit)
                                            .build(),
                                    )
                                    setForeground(
                                        foregroundInfo(
                                            text =
                                                "Preparing segment ${window.index + 1} of $chunkCount",
                                            progress =
                                                window.index.toFloat() /
                                                    chunkCount.coerceAtLeast(1),
                                        ),
                                    )
                                }
                            },
                            onChunkReady = { window, chunkCount ->
                                runBlocking {
                                    pauseAtSafeBoundaryIfRecording(
                                        jobId = job.id,
                                        resumePhase = TranscriptionPhase.ENCODING,
                                    )
                                    val current =
                                        repository.getTranscriptionJob(job.id) ?: return@runBlocking
                                    updateJob(
                                        current.copy(
                                            status = TranscriptionJobStatus.RUNNING,
                                            phase = TranscriptionPhase.ENCODING,
                                            currentChunkIndex = window.index,
                                            currentToken = 0,
                                            updatedAtMillis = System.currentTimeMillis(),
                                        ),
                                    )
                                    setForeground(
                                        foregroundInfo(
                                            text =
                                                "Analyzing segment ${window.index + 1} of $chunkCount",
                                            progress =
                                                window.index.toFloat() /
                                                    chunkCount.coerceAtLeast(1),
                                        ),
                                    )
                                }
                            },
                            onProgress = { progress ->
                                if (InferenceGate.isRecording) {
                                    runBlocking {
                                        pauseAtSafeBoundaryIfRecording(job.id)
                                    }
                                }
                                val overall = synchronized(progressStateLock) {
                                    val previous = publishedTokens[progress.chunkIndex]
                                    val advancedEnough =
                                        progress.generatedTokens - previous >= PROGRESS_TOKEN_STEP
                                    val reachedLimit =
                                        progress.generatedTokens >= progress.tokenLimit
                                    if (previous < 0 || advancedEnough || reachedLimit) {
                                        publishedTokens[progress.chunkIndex] =
                                            progress.generatedTokens
                                        chunkProgress[progress.chunkIndex] =
                                            progress.generatedTokens.toFloat() /
                                                progress.tokenLimit.coerceAtLeast(1)
                                        chunkProgress.sum() /
                                            progress.chunkCount.coerceAtLeast(1)
                                    } else {
                                        null
                                    }
                                }
                                if (overall != null) {
                                    runBlocking {
                                        val current =
                                            repository.getTranscriptionJob(job.id)
                                                ?: return@runBlocking
                                        updateJob(
                                            current.copy(
                                                status = TranscriptionJobStatus.RUNNING,
                                                phase = TranscriptionPhase.DECODING,
                                                currentChunkIndex = progress.chunkIndex,
                                                currentToken = progress.generatedTokens,
                                                currentTokenLimit = progress.tokenLimit,
                                                updatedAtMillis = System.currentTimeMillis(),
                                            ),
                                        )
                                        setProgress(
                                            Data.Builder()
                                                .putInt(PROGRESS_CHUNK, progress.chunkIndex)
                                                .putInt(PROGRESS_TOTAL_CHUNKS, progress.chunkCount)
                                                .putInt(PROGRESS_TOKEN, progress.generatedTokens)
                                                .putInt(PROGRESS_TOKEN_LIMIT, progress.tokenLimit)
                                                .build(),
                                        )
                                        setForeground(
                                            foregroundInfo(
                                                text =
                                                    "Transcribing segment " +
                                                        "${progress.chunkIndex + 1} " +
                                                        "of ${progress.chunkCount}",
                                                progress = overall,
                                            ),
                                        )
                                    }
                                }
                            },
                            onChunkCompleted = { chunk ->
                                synchronized(progressStateLock) {
                                    chunkProgress[chunk.window.index] = 1f
                                }
                                benchmark?.segmentCompleted(chunk)
                                runBlocking {
                                    persistChunk(job, memoId, chunk)
                                    pauseAtSafeBoundaryIfRecording(
                                        jobId = job.id,
                                        resumePhase = TranscriptionPhase.ENCODING,
                                    )
                                }
                            },
                        )
                        benchmark?.transcriptionCompleted(result)
                        repository.getTranscriptionJob(job.id)?.let { current ->
                            val thermalStatus = currentThermalStatus()
                            job = updateJob(
                                current.copy(
                                    transcriptionElapsedMillis =
                                        SystemClock.elapsedRealtime() - transcriptionStartedAt,
                                    maximumThermalStatus = maximumThermalStatus(
                                        current.maximumThermalStatus,
                                        thermalStatus,
                                    ),
                                    updatedAtMillis = System.currentTimeMillis(),
                                ),
                            )
                        }
                        result
                    } finally {
                        engineLease?.close()
                        voiceActivityDetector?.close()
                    }
                }
            }

            job = repository.getTranscriptionJob(job.id) ?: return Result.failure()
            updateJob(
                job.copy(
                    phase = TranscriptionPhase.GENERATING_METADATA,
                    currentChunkIndex = null,
                    currentToken = 0,
                    updatedAtMillis = System.currentTimeMillis(),
                ),
            )
            setForeground(foregroundInfo("Creating title and summary", progress = 1f))
            var metadata: com.example.voicememo.transcription.RecordingMetadata? = null
            var metadataFailure: Throwable? = null
            if (transcript.text.isNotBlank()) {
                if (!InferenceGate.awaitRecordingIdle { isStopped }) {
                    throw WhisperCancelledException()
                }
                try {
                    withContext(Dispatchers.IO) {
                        metadata = GraniteRecordingMetadataGenerator
                            .create(applicationContext)
                            .generate(transcript.text)
                    }
                } catch (cancelled: CancellationException) {
                    throw cancelled
                } catch (failure: Throwable) {
                    metadataFailure = failure
                }
            }
            val transcriptWarnings = transcript.warnings.toMutableList()
            metadataFailure?.let { failure ->
                Log.w(TAG, "Local title and summary generation failed.", failure)
                transcriptWarnings +=
                    "The transcript finished, but its title and summary could not be generated."
            }
            repository.completeTranscriptionJob(
                jobId = job.id,
                mergedText = transcript.text,
                detectedLanguage = transcript.detectedLanguage.takeUnless { it == "und" },
                warning = transcriptWarnings.takeIf(List<String>::isNotEmpty)?.joinToString("\n"),
                generatedTitle = metadata?.title,
                generatedSummary = metadata?.summary,
                metadataModelId = metadata?.let {
                    GraniteRecordingMetadataGenerator.MODEL_ID
                },
            )
            indexForSemanticSearch(memoId, job.id)
            benchmark?.finish(outcome = "SUCCESS")
            return Result.success()
        } catch (cancelled: CancellationException) {
            benchmark?.finish(outcome = "CANCELLED", errorCode = "COROUTINE_CANCELLED")
            withContext(NonCancellable) { recordInterruption(job) }
            throw cancelled
        } catch (cancelled: WhisperCancelledException) {
            benchmark?.finish(outcome = "CANCELLED", errorCode = "CANCELLED")
            val explicitlyCancelled = withContext(NonCancellable) {
                recordInterruption(job)
            }
            return if (explicitlyCancelled) {
                Result.failure(failureData("CANCELLED", "Transcription was cancelled."))
            } else {
                Result.retry()
            }
        } catch (outOfMemory: OutOfMemoryError) {
            val message =
                "The device could not allocate enough memory for local Whisper inference. " +
                    "Close other apps and retry, or use a lower-concurrency CPU build."
            benchmark?.finish(outcome = "FAILED", errorCode = "ALLOCATION_FAILED")
            failJob(job, "ALLOCATION_FAILED", message)
            return Result.failure(failureData("ALLOCATION_FAILED", message))
        } catch (failure: Throwable) {
            val code = if (failure is IllegalArgumentException) "INVALID_AUDIO" else "INFERENCE_FAILED"
            val message = failure.message ?: "Local transcription failed."
            benchmark?.finish(outcome = "FAILED", errorCode = code)
            failJob(job, code, message)
            return Result.failure(failureData(code, message))
        }
    }

    /**
     * Embeds the finished transcript's segments for semantic find-in-transcript.
     * It is best effort: the detail screen indexes lazily on the first search,
     * so a failure or an active recording only defers the work.
     */
    private suspend fun indexForSemanticSearch(memoId: String, jobId: String) {
        if (!BuildConfig.SEMANTIC_SEARCH_ENABLED || InferenceGate.isRecording) return
        try {
            val tokenizer = applicationContext.assets.open(WhisperAssets.TOKENIZER)
                .use(WhisperTokenizer::load)
            val segments = TranscriptSegments.generated(
                repository.completedTranscriptChunks(jobId),
                tokenizer,
            )
            val result = container.semanticTranscriptIndex.index(memoId, segments)
            Log.i(
                TAG,
                "Semantic index: ${result.embedded} embedded in ${result.embeddingMillis} ms, " +
                    "model load ${result.modelLoadMillis} ms.",
            )
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (failure: Throwable) {
            Log.w(TAG, "Semantic transcript indexing failed; it will run on first search.", failure)
        } finally {
            withContext(NonCancellable) { container.semanticTranscriptIndex.release() }
        }
    }

    private suspend fun persistChunk(
        originalJob: TranscriptionJob,
        memoId: String,
        chunk: LongFormChunk,
    ) {
        val result = chunk.result
        repository.upsertTranscriptChunk(
            TranscriptChunk(
                jobId = originalJob.id,
                chunkIndex = chunk.window.index,
                memoId = memoId,
                windowStartMillis = chunk.window.startSample * 1_000L /
                    WhisperFeatureExtractor.SAMPLE_RATE,
                windowEndMillis = chunk.window.endSample * 1_000L /
                    WhisperFeatureExtractor.SAMPLE_RATE,
                status = TranscriptChunkStatus.COMPLETED,
                rawText = result.text,
                tokenIds = result.tokens,
                detectedLanguage = result.detectedLanguage.takeUnless { it == "und" },
                truncated = result.truncated,
                warning = result.warnings.takeIf(List<String>::isNotEmpty)?.joinToString("\n"),
                completedAtMillis = System.currentTimeMillis(),
                inferenceMillis = result.inferenceMillis,
            ),
        )
        val persisted = repository.getTranscriptionJob(originalJob.id) ?: return
        val thermalStatus = currentThermalStatus()
        repository.upsertTranscriptionJob(
            persisted.copy(
                status = TranscriptionJobStatus.RUNNING,
                phase = TranscriptionPhase.ENCODING,
                completedChunks = maxOf(persisted.completedChunks, chunk.window.index + 1),
                currentChunkIndex = null,
                currentToken = 0,
                maximumThermalStatus = maximumThermalStatus(
                    persisted.maximumThermalStatus,
                    thermalStatus,
                ),
                updatedAtMillis = System.currentTimeMillis(),
            ),
        )
    }

    private suspend fun pauseAtSafeBoundaryIfRecording(
        jobId: String,
        resumePhase: TranscriptionPhase = TranscriptionPhase.DECODING,
    ) {
        if (!InferenceGate.isRecording) return
        repository.getTranscriptionJob(jobId)?.let { current ->
            repository.upsertTranscriptionJob(
                current.copy(
                    status = TranscriptionJobStatus.PAUSED_FOR_RECORDING,
                    phase = TranscriptionPhase.PAUSED,
                    updatedAtMillis = System.currentTimeMillis(),
                ),
            )
        }
        setForeground(foregroundInfo("Paused while recording", progress = null))
        if (!InferenceGate.awaitRecordingIdle { isStopped }) {
            throw WhisperCancelledException()
        }
        repository.getTranscriptionJob(jobId)?.let { current ->
            repository.upsertTranscriptionJob(
                current.copy(
                    status = TranscriptionJobStatus.RUNNING,
                    phase = resumePhase,
                    updatedAtMillis = System.currentTimeMillis(),
                ),
            )
        }
    }

    private fun currentThermalStatus(): Int? =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            applicationContext.getSystemService(PowerManager::class.java)
                ?.currentThermalStatus
        } else {
            null
        }

    private fun TranscriptChunk.toLongFormChunk(): LongFormChunk {
        val startSample =
            (windowStartMillis * WhisperFeatureExtractor.SAMPLE_RATE / 1_000L).toInt()
        val endSample =
            (windowEndMillis * WhisperFeatureExtractor.SAMPLE_RATE / 1_000L).toInt()
        return LongFormChunk(
            window = AudioChunkWindow(chunkIndex, startSample, endSample),
            result = WhisperChunkResult(
                tokens = tokenIds,
                text = rawText,
                detectedLanguage = detectedLanguage ?: "und",
                inferenceMillis = 0L,
                truncated = truncated,
                warnings = warning?.let(::listOf).orEmpty(),
            ),
        )
    }

    private suspend fun updateJob(job: TranscriptionJob): TranscriptionJob {
        repository.upsertTranscriptionJob(job)
        return job
    }

    private suspend fun cancelJob(job: TranscriptionJob) {
        repository.getTranscriptionJob(job.id)?.let { current ->
            repository.upsertTranscriptionJob(
                current.copy(
                    status = TranscriptionJobStatus.CANCELLED,
                    phase = TranscriptionPhase.PAUSED,
                    currentChunkIndex = null,
                    currentToken = 0,
                    updatedAtMillis = System.currentTimeMillis(),
                ),
            )
        }
    }

    /**
     * WorkManager also cancels a running coroutine for resumable system stops
     * such as device idle, thermal pressure, or process replacement. Only a
     * cancellation explicitly requested through the app should become a
     * terminal cancelled job.
     *
     * @return true when the user explicitly cancelled the job.
     */
    private suspend fun recordInterruption(job: TranscriptionJob): Boolean {
        val current = repository.getTranscriptionJob(job.id) ?: return true
        if (shouldMarkTranscriptionCancelled(current.status)) {
            cancelJob(job)
            return true
        }
        repository.upsertTranscriptionJob(
            current.copy(
                status = TranscriptionJobStatus.QUEUED,
                phase = TranscriptionPhase.QUEUED,
                currentChunkIndex = null,
                currentToken = 0,
                updatedAtMillis = System.currentTimeMillis(),
            ),
        )
        return false
    }

    private suspend fun failJob(job: TranscriptionJob, code: String, message: String) {
        repository.getTranscriptionJob(job.id)?.let { current ->
            repository.upsertTranscriptionJob(
                current.copy(
                    status = TranscriptionJobStatus.FAILED,
                    phase = TranscriptionPhase.FAILED,
                    currentChunkIndex = null,
                    currentToken = 0,
                    errorCode = code,
                    errorMessage = message,
                    updatedAtMillis = System.currentTimeMillis(),
                ),
            )
        }
    }

    private fun foregroundInfo(text: String, progress: Float?): ForegroundInfo {
        val builder = NotificationCompat.Builder(applicationContext, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setContentTitle("Transcribing locally")
            .setContentText(text)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setCategory(NotificationCompat.CATEGORY_PROGRESS)
        if (progress == null) {
            builder.setProgress(0, 0, true)
        } else {
            builder.setProgress(100, (progress.coerceIn(0f, 1f) * 100).toInt(), false)
        }
        return ForegroundInfo(
            NOTIFICATION_ID,
            builder.build(),
            ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC,
        )
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        notificationManager.createNotificationChannel(
            NotificationChannel(
                CHANNEL_ID,
                applicationContext.getString(R.string.transcription_channel_name),
                NotificationManager.IMPORTANCE_LOW,
            ).apply {
                description = applicationContext.getString(R.string.transcription_channel_description)
            },
        )
    }

    private fun failureData(code: String, message: String): Data =
        Data.Builder()
            .putString(OUTPUT_ERROR_CODE, code)
            .putString(OUTPUT_ERROR_MESSAGE, message)
            .build()

    companion object {
        private const val TAG = "TranscriptionWorker"
        const val KEY_JOB_ID = "job_id"
        const val KEY_MEMO_ID = "memo_id"
        const val KEY_TIMESTAMPS_ENABLED = "timestamps_enabled"
        const val KEY_VAD_ENABLED = "vad_enabled"
        const val KEY_GPU_ACCELERATION_ENABLED = "gpu_acceleration_enabled"
        const val KEY_TRANSCRIPTION_MODEL_ID = "transcription_model_id"
        const val KEY_USE_PRETRANSCRIBED_WINDOWS = "use_pretranscribed_windows"
        const val PROGRESS_CHUNK = "chunk"
        const val PROGRESS_TOTAL_CHUNKS = "total_chunks"
        const val PROGRESS_TOKEN = "token"
        const val PROGRESS_TOKEN_LIMIT = "token_limit"
        const val OUTPUT_ERROR_CODE = "error_code"
        const val OUTPUT_ERROR_MESSAGE = "error_message"

        private const val CHANNEL_ID = "local_transcription"
        private const val NOTIFICATION_ID = 1002
        private const val PROGRESS_TOKEN_STEP = 4
    }
}

internal fun shouldMarkTranscriptionCancelled(status: TranscriptionJobStatus): Boolean =
    status == TranscriptionJobStatus.CANCEL_REQUESTED ||
        status == TranscriptionJobStatus.CANCELLED

internal fun maximumThermalStatus(
    previous: Int?,
    current: Int?,
): Int? = when {
    previous == null -> current
    current == null -> previous
    else -> maxOf(previous, current)
}

internal fun shouldUsePreTranscribedWindows(
    reuseRecordingStageResults: Boolean,
    transcriptionModel: TranscriptionModel,
): Boolean =
    transcriptionModel == TranscriptionModel.WHISPER_LARGE_V3_TURBO &&
        reuseRecordingStageResults

internal fun transcriptionWorkerCount(
    transcriptionModel: TranscriptionModel,
    requestedBackend: WhisperBackendRequest,
    usePreTranscribedWindows: Boolean,
    configuredCpuWorkerCount: Int,
): Int {
    require(configuredCpuWorkerCount > 0)
    return if (
        requestedBackend == WhisperBackendRequest.GPU ||
        transcriptionModel == TranscriptionModel.PARAKEET_TDT_0_6B_V3 ||
        usePreTranscribedWindows
    ) {
        1
    } else {
        configuredCpuWorkerCount
    }
}
