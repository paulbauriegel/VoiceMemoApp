package com.example.voicememo.transcription

import android.content.Context
import android.content.res.AssetManager
import android.os.Build
import android.os.Debug
import android.os.PowerManager
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.voicememo.VoiceMemoApplication
import com.google.ai.edge.litert.CompiledModel
import java.io.File
import java.security.MessageDigest
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.runBlocking
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Non-persisting whole-memo quality and latency benchmark.
 *
 * The benchmark reads one existing memo from app-private storage and never
 * creates or updates a transcript, job, chunk, memo, or audio file. The CPU
 * result produced in this same run is the word-error-rate oracle.
 */
@RunWith(AndroidJUnit4::class)
class GpuWholeMemoWerBenchmarkTest {
    @Test
    fun compareWholeMemoBackends() = runBlocking {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val arguments = InstrumentationRegistry.getArguments()
        val context = instrumentation.targetContext
        val application = context.applicationContext as VoiceMemoApplication
        require(application.container.memoRepository.getOldestRunnableJob() == null) {
            "A VoiceMemo transcription job is queued, running, or paused."
        }
        val runId = requireSafeRunId(arguments.getString(ARG_RUN_ID))
        val candidateOnly =
            arguments.getString(ARG_CANDIDATE_ONLY)
                ?.toBooleanStrictOrNull()
                ?: false
        val candidateBackend =
            arguments.getString(ARG_CANDIDATE_BACKEND) ?: CANDIDATE_BACKEND_GPU
        require(
            candidateBackend in setOf(
                CANDIDATE_BACKEND_GPU,
                CANDIDATE_BACKEND_CPU,
                CANDIDATE_BACKEND_WHISPER_CPP,
            ),
        ) {
            "Unsupported candidate backend: $candidateBackend"
        }
        val cpuWorkers = arguments.getString(ARG_CPU_WORKERS)?.toIntOrNull() ?: 2
        val cpuThreads = arguments.getString(ARG_CPU_THREADS)?.toIntOrNull() ?: 3
        val whisperCppThreads =
            arguments.getString(ARG_WHISPER_CPP_THREADS)?.toIntOrNull() ?: 4
        val whisperCppBackend = WhisperCppBenchmarkBackend.fromArgument(
            arguments.getString(ARG_WHISPER_CPP_BACKEND)
                ?: WhisperCppBenchmarkBackend.CPU.argument,
        )
        val timestampTokensEnabled =
            arguments.getString(ARG_TIMESTAMP_TOKENS_ENABLED)
                ?.toBooleanStrictOrNull()
                ?: false
        require(cpuWorkers in setOf(1, 2, 4)) {
            "CPU workers must be 1, 2, or 4."
        }
        require(cpuThreads in 1..8) { "CPU threads must be between 1 and 8." }
        require(whisperCppThreads in 1..8) {
            "whisper.cpp threads must be between 1 and 8."
        }
        val externalModelName = arguments.getString(ARG_EXTERNAL_MODEL_NAME)
        val externalModelSha256 = arguments.getString(ARG_EXTERNAL_MODEL_SHA256)
        val externalEncoderModelName = arguments.getString(ARG_EXTERNAL_ENCODER_MODEL_NAME)
        val externalEncoderModelSha256 =
            arguments.getString(ARG_EXTERNAL_ENCODER_MODEL_SHA256)
        val externalDecoderModelName = arguments.getString(ARG_EXTERNAL_DECODER_MODEL_NAME)
        val externalDecoderModelSha256 =
            arguments.getString(ARG_EXTERNAL_DECODER_MODEL_SHA256)
        val externalProjectionModelName =
            arguments.getString(ARG_EXTERNAL_PROJECTION_MODEL_NAME)
        val externalProjectionModelSha256 =
            arguments.getString(ARG_EXTERNAL_PROJECTION_MODEL_SHA256)
        val externalCrossCacheModelName =
            arguments.getString(ARG_EXTERNAL_CROSS_CACHE_MODEL_NAME)
        val externalCrossCacheModelSha256 =
            arguments.getString(ARG_EXTERNAL_CROSS_CACHE_MODEL_SHA256)
        val externalAudioName = arguments.getString(ARG_EXTERNAL_AUDIO_NAME)
        val memoId = arguments.getString(ARG_MEMO_ID)
        val forcedLanguage = arguments.getString(ARG_FORCED_LANGUAGE)
        val gpuPrecisionName = arguments.getString(ARG_GPU_PRECISION) ?: GPU_PRECISION_FP32
        fun parseGpuPrecision(value: String): CompiledModel.GpuOptions.Precision = when (value) {
            GPU_PRECISION_FP32 -> CompiledModel.GpuOptions.Precision.FP32
            GPU_PRECISION_FP16_NATIVE -> CompiledModel.GpuOptions.Precision.FP16
            GPU_PRECISION_FP16_FP32_ACCUM -> CompiledModel.GpuOptions.Precision.FP16
            else -> error("Unsupported GPU precision mode: $value")
        }
        val gpuPrecision = parseGpuPrecision(gpuPrecisionName)
        val encoderGpuPrecisionName =
            arguments.getString(ARG_ENCODER_GPU_PRECISION) ?: gpuPrecisionName
        val decoderGpuPrecisionName =
            arguments.getString(ARG_DECODER_GPU_PRECISION) ?: gpuPrecisionName
        val encoderGpuPrecision = parseGpuPrecision(encoderGpuPrecisionName)
        val decoderGpuPrecision = parseGpuPrecision(decoderGpuPrecisionName)
        val splitModelRequested = externalEncoderModelName != null || externalDecoderModelName != null
        val activeRowRequested = externalProjectionModelName != null
        val kvCacheRequested = externalCrossCacheModelName != null
        val useFullFloatDecoder =
            arguments.getString(ARG_USE_FULL_FLOAT_DECODER)
                ?.toBooleanStrictOrNull()
                ?: false
        val promotionGate =
            arguments.getString(ARG_PROMOTION_GATE)
                ?.toBooleanStrictOrNull()
                ?: false
        require(!promotionGate || useFullFloatDecoder) {
            "The GPU promotion gate requires the complete FLOAT32 decoder model."
        }
        require(!candidateOnly || externalModelName != null || splitModelRequested) {
            "Candidate-only mode requires an external combined or split model."
        }
        require((externalModelName == null) == (externalModelSha256 == null)) {
            "External model name and SHA-256 must be supplied together."
        }
        require(
            externalModelSha256 == null ||
                SAFE_SHA256.matches(externalModelSha256),
        ) {
            "External model SHA-256 is invalid."
        }
        require(
            listOf(
                externalEncoderModelName,
                externalEncoderModelSha256,
                externalDecoderModelName,
                externalDecoderModelSha256,
            ).all { it != null } ||
                listOf(
                    externalEncoderModelName,
                    externalEncoderModelSha256,
                    externalDecoderModelName,
                    externalDecoderModelSha256,
                ).all { it == null },
        ) {
            "Split model names and SHA-256 values must be supplied together."
        }
        require(!splitModelRequested || candidateBackend == CANDIDATE_BACKEND_GPU) {
            "Split models are supported only by the GPU benchmark candidate."
        }
        require((externalProjectionModelName == null) == (externalProjectionModelSha256 == null)) {
            "External projection model name and SHA-256 must be supplied together."
        }
        require(!activeRowRequested || splitModelRequested) {
            "Active-row projection requires external encoder and decoder-core models."
        }
        require((externalCrossCacheModelName == null) == (externalCrossCacheModelSha256 == null)) {
            "External cross-cache model name and SHA-256 must be supplied together."
        }
        require(!kvCacheRequested || activeRowRequested) {
            "KV-cache decoding requires the active-row projection model."
        }
        val whisperCppModel =
            if (candidateBackend == CANDIDATE_BACKEND_WHISPER_CPP) {
                val model = WhisperCppBenchmarkModel.fromFileName(
                    requireNotNull(externalModelName),
                )
                require(model != null && externalModelSha256 == model.sha256) {
                    "whisper.cpp requires a pinned large-v3-turbo benchmark model."
                }
                model
            } else {
                null
            }
        require(!candidateOnly || externalAudioName != null || memoId != null) {
            "Candidate-only mode requires an external audio file or existing memo ID."
        }
        require(!candidateOnly || !promotionGate) {
            "Candidate-only mode does not use the existing promotion gate."
        }
        val externalModelFile = externalModelName?.let { name ->
            requireStagedFile(
                context = context,
                directory = if (candidateBackend == CANDIDATE_BACKEND_WHISPER_CPP) {
                    WHISPER_CPP_MODEL_DIRECTORY
                } else {
                    EXTERNAL_MODEL_DIRECTORY
                },
                name = name,
                extension = if (candidateBackend == CANDIDATE_BACKEND_WHISPER_CPP) {
                    ".bin"
                } else {
                    ".tflite"
                },
            )
        }
        val externalEncoderModelFile = externalEncoderModelName?.let { name ->
            requireStagedFile(context, EXTERNAL_MODEL_DIRECTORY, name, ".tflite").also {
                require(sha256(it) == externalEncoderModelSha256) {
                    "Staged encoder model SHA-256 does not match."
                }
            }
        }
        val externalDecoderModelFile = externalDecoderModelName?.let { name ->
            requireStagedFile(context, EXTERNAL_MODEL_DIRECTORY, name, ".tflite").also {
                require(sha256(it) == externalDecoderModelSha256) {
                    "Staged decoder model SHA-256 does not match."
                }
            }
        }
        val externalProjectionModelFile = externalProjectionModelName?.let { name ->
            requireStagedFile(context, EXTERNAL_MODEL_DIRECTORY, name, ".tflite").also {
                require(sha256(it) == externalProjectionModelSha256) {
                    "Staged active-row projection SHA-256 does not match."
                }
            }
        }
        val externalCrossCacheModelFile = externalCrossCacheModelName?.let { name ->
            requireStagedFile(context, EXTERNAL_MODEL_DIRECTORY, name, ".tflite").also {
                require(sha256(it) == externalCrossCacheModelSha256) {
                    "Staged cross-cache model SHA-256 does not match."
                }
            }
        }
        if (candidateBackend == CANDIDATE_BACKEND_WHISPER_CPP) {
            val modelFile = requireNotNull(externalModelFile)
            val model = requireNotNull(whisperCppModel)
            require(modelFile.length() == model.sizeBytes) {
                "Staged whisper.cpp model size does not match the pinned artifact."
            }
            require(sha256(modelFile) == model.sha256) {
                "Staged whisper.cpp model SHA-256 does not match the pinned artifact."
            }
        }
        val gpuModelOverride = when {
            externalModelFile != null && candidateBackend != CANDIDATE_BACKEND_WHISPER_CPP ->
                GpuModelOverride(
                    filePath = externalModelFile.absolutePath,
                )
            useFullFloatDecoder -> GpuModelOverride(
                assets = instrumentation.context.assets,
                assetPath = FULL_FLOAT_DECODER_ASSET,
            )
            else -> null
        }
        val audioFile = if (externalAudioName != null) {
            requireStagedFile(
                context = context,
                directory = EXTERNAL_AUDIO_DIRECTORY,
                name = externalAudioName,
                extension = ".wav",
            )
        } else {
            val requiredMemoId = requireArgument(memoId, ARG_MEMO_ID)
            val memo = requireNotNull(
                application.container.memoRepository.getMemo(requiredMemoId),
            ) {
                "Unknown memo $requiredMemoId."
            }
            require(!memo.pendingDeletion) { "Memo $requiredMemoId is pending deletion." }
            File(memo.audioPath).also {
                require(it.isFile) { "Memo audio is missing." }
            }
        }
        val requestedLanguage = forcedLanguage?.let {
            require(SAFE_LANGUAGE.matches(it)) { "Invalid forced language." }
            WhisperLanguage.Forced(it)
        } ?: WhisperLanguage.Auto

        val reportDirectory = File(context.cacheDir, REPORT_DIRECTORY).apply { mkdirs() }
        val reportFile = File(reportDirectory, "whole-$runId.json")
        val temporaryReport = File(reportDirectory, "${reportFile.name}.tmp")
        reportFile.delete()
        temporaryReport.delete()

        val report = JSONObject()
            .put("run_id", runId)
            .put("memo_id", memoId ?: JSONObject.NULL)
            .put("audio_file_name", externalAudioName ?: JSONObject.NULL)
            .put("persisted_output", false)
            .put("candidate_only", candidateOnly)
            .put("candidate_backend", candidateBackend)
            .put("cpu_workers", cpuWorkers)
            .put("cpu_threads_per_worker", cpuThreads)
            .put("whisper_cpp_threads", whisperCppThreads)
            .put("whisper_cpp_backend", whisperCppBackend.argument)
            .put("whisper_cpp_model", whisperCppModel?.argument ?: JSONObject.NULL)
            .put("timestamp_tokens_enabled", timestampTokensEnabled)
            .put("use_full_float_decoder", useFullFloatDecoder)
            .put("promotion_gate", promotionGate)
            .put("external_model_name", externalModelName ?: JSONObject.NULL)
            .put("external_encoder_model_name", externalEncoderModelName ?: JSONObject.NULL)
            .put("external_decoder_model_name", externalDecoderModelName ?: JSONObject.NULL)
            .put(
                "external_projection_model_name",
                externalProjectionModelName ?: JSONObject.NULL,
            )
            .put(
                "external_cross_cache_model_name",
                externalCrossCacheModelName ?: JSONObject.NULL,
            )
            .put("requested_language", forcedLanguage ?: "auto")
            .put("gpu_precision", gpuPrecisionName)
            .put("encoder_gpu_precision", encoderGpuPrecisionName)
            .put("decoder_gpu_precision", decoderGpuPrecisionName)
            .put(
                "model_sha256",
                externalModelSha256 ?: externalDecoderModelSha256 ?: WhisperAssets.MODEL_SHA256,
            )
            .put("encoder_model_sha256", externalEncoderModelSha256 ?: JSONObject.NULL)
            .put("decoder_model_sha256", externalDecoderModelSha256 ?: JSONObject.NULL)
            .put("projection_model_sha256", externalProjectionModelSha256 ?: JSONObject.NULL)
            .put("cross_cache_model_sha256", externalCrossCacheModelSha256 ?: JSONObject.NULL)
            .put("status", STATUS_RUNNING)
        fun publishCheckpoint() {
            temporaryReport.writeText(report.toString(2))
            reportFile.delete()
            check(temporaryReport.renameTo(reportFile)) {
                "Could not publish whole-memo GPU WER report."
            }
        }

        try {
            val audioInfo = StreamingWavChunkSource.open(audioFile).use { source ->
                JSONObject()
                    .put("source_sample_rate", source.info.sourceSampleRate)
                    .put("source_channels", source.info.sourceChannels)
                    .put("whisper_sample_count", source.info.whisperSampleCount)
                    .put("chunk_count", source.chunkCount)
            }
            report.put("audio", audioInfo)

            if (candidateOnly) {
                val externalModelPath = externalModelFile?.absolutePath
                val candidate = benchmarkWithEngines(
                    context = context,
                    audioFile = audioFile,
                    mode = when {
                        whisperCppModel == WhisperCppBenchmarkModel.Q8_0 &&
                            whisperCppBackend == WhisperCppBenchmarkBackend.VULKAN ->
                            Mode.WHISPER_CPP_Q8_0_VULKAN
                        whisperCppModel == WhisperCppBenchmarkModel.Q8_0 ->
                            Mode.WHISPER_CPP_Q8_0_CPU
                        whisperCppModel == WhisperCppBenchmarkModel.FP16 &&
                            whisperCppBackend == WhisperCppBenchmarkBackend.VULKAN ->
                            Mode.WHISPER_CPP_FP16_VULKAN
                        whisperCppModel == WhisperCppBenchmarkModel.FP16 ->
                            Mode.WHISPER_CPP_FP16_CPU
                        candidateBackend == CANDIDATE_BACKEND_WHISPER_CPP &&
                            whisperCppBackend == WhisperCppBenchmarkBackend.VULKAN ->
                            Mode.WHISPER_CPP_Q5_0_VULKAN
                        candidateBackend == CANDIDATE_BACKEND_WHISPER_CPP ->
                            Mode.WHISPER_CPP_Q5_0_CPU
                        candidateBackend == CANDIDATE_BACKEND_CPU && cpuWorkers == 1 ->
                            Mode.CPU_1X3
                        candidateBackend == CANDIDATE_BACKEND_CPU && cpuWorkers == 4 ->
                            Mode.CPU_4X3
                        candidateBackend == CANDIDATE_BACKEND_CPU -> Mode.CPU_2X3
                        splitModelRequested -> splitGpuMode(
                            encoderGpuPrecisionName,
                            decoderGpuPrecisionName,
                        )
                        gpuPrecisionName == GPU_PRECISION_FP16_NATIVE ->
                            Mode.FULL_GPU_FP16_NATIVE_NO_SOURCE_QUANT
                        gpuPrecisionName == GPU_PRECISION_FP16_FP32_ACCUM ->
                            Mode.FULL_GPU_FP16_FP32_ACCUM_NO_SOURCE_QUANT
                        else -> Mode.FULL_GPU_NO_SOURCE_QUANT
                    },
                    language = requestedLanguage,
                    openEngines = {
                        when (candidateBackend) {
                            CANDIDATE_BACKEND_WHISPER_CPP -> listOf(
                                WhisperCppBenchmarkEngine.open(
                                    modelFile = requireNotNull(externalModelFile),
                                    threadCount = whisperCppThreads,
                                    backend = whisperCppBackend,
                                    timestampTokensEnabled = timestampTokensEnabled,
                                ),
                            )
                            CANDIDATE_BACKEND_CPU -> openCpuEngines(
                                context = context,
                                workerCount = cpuWorkers,
                                threadCount = cpuThreads,
                                modelFilePath = requireNotNull(externalModelPath),
                                timestampTokensEnabled = timestampTokensEnabled,
                            )
                            else -> if (kvCacheRequested) {
                                listOf(
                                    LiteRtKvCacheWhisperEngine.open(
                                        context = context,
                                        encoderModelPath =
                                            requireNotNull(externalEncoderModelFile).absolutePath,
                                        crossCacheModelPath =
                                            requireNotNull(externalCrossCacheModelFile).absolutePath,
                                        stepModelPath =
                                            requireNotNull(externalDecoderModelFile).absolutePath,
                                        projectionModelPath =
                                            requireNotNull(externalProjectionModelFile).absolutePath,
                                        encoderPrecision = encoderGpuPrecision,
                                        decoderPrecision = decoderGpuPrecision,
                                    ),
                                )
                            } else if (activeRowRequested) {
                                listOf(
                                    LiteRtActiveRowWhisperEngine.open(
                                        context = context,
                                        encoderModelPath =
                                            requireNotNull(externalEncoderModelFile).absolutePath,
                                        decoderCorePath =
                                            requireNotNull(externalDecoderModelFile).absolutePath,
                                        activeRowProjectionPath =
                                            requireNotNull(externalProjectionModelFile).absolutePath,
                                        encoderPrecision = encoderGpuPrecision,
                                        decoderPrecision = decoderGpuPrecision,
                                    ),
                                )
                            } else listOf(
                                LiteRtFullGpuWhisperEngine.open(
                                    context = context,
                                    allowSourceQuantizedFcConvOps = false,
                                    diagnosticPrecision = gpuPrecision,
                                    diagnosticModelFilePath = externalModelPath,
                                    diagnosticEncoderModelFilePath =
                                        externalEncoderModelFile?.absolutePath,
                                    diagnosticDecoderModelFilePath =
                                        externalDecoderModelFile?.absolutePath,
                                    diagnosticEncoderPrecision = encoderGpuPrecision,
                                    diagnosticDecoderPrecision = decoderGpuPrecision,
                                    timestampsEnabled = timestampTokensEnabled,
                                ),
                            )
                        }
                    },
                )
                report
                    .put("candidate", candidate.toJson(referenceText = null))
                    .put("status", STATUS_COMPLETED)
                publishCheckpoint()
                return@runBlocking
            }

            val cpu = benchmarkWithEngines(
                context = context,
                audioFile = audioFile,
                mode = Mode.CPU_2X3,
                language = WhisperLanguage.Auto,
                openEngines = {
                    val first = LiteRtWhisperEngine.open(context, threadCount = 3)
                    try {
                        listOf(
                            first,
                            LiteRtWhisperEngine.open(context, threadCount = 3),
                        )
                    } catch (error: Throwable) {
                        first.close()
                        throw error
                    }
                },
            )
            require(cpu.transcript.detectedLanguage != "und") {
                "CPU reference could not detect a language."
            }
            val referenceLanguage = WhisperLanguage.Forced(cpu.transcript.detectedLanguage)
            report
                .put("reference", cpu.toJson(referenceText = null))
                .put("candidates", JSONArray())
            publishCheckpoint()

            val candidates = report.getJSONArray("candidates")
            if (!promotionGate) {
                candidates.put(
                    benchmarkCandidate(
                        context = context,
                        audioFile = audioFile,
                        mode = Mode.FULL_GPU_DOMINANT,
                        reference = cpu.transcript,
                        language = referenceLanguage,
                        openEngines = {
                            listOf(
                                LiteRtFullGpuWhisperEngine.open(
                                    context = context,
                                    diagnosticModelAssets = gpuModelOverride?.assets,
                                    diagnosticModelPath = gpuModelOverride?.assetPath,
                                    diagnosticModelFilePath = gpuModelOverride?.filePath,
                                ),
                            )
                        },
                    ),
                )
                publishCheckpoint()
            }
            candidates.put(
                benchmarkCandidate(
                    context = context,
                    audioFile = audioFile,
                    mode = Mode.FULL_GPU_NO_SOURCE_QUANT,
                    reference = cpu.transcript,
                    language = referenceLanguage,
                    openEngines = {
                        listOf(
                            LiteRtFullGpuWhisperEngine.open(
                                context = context,
                                allowSourceQuantizedFcConvOps = false,
                                diagnosticModelAssets = gpuModelOverride?.assets,
                                diagnosticModelPath = gpuModelOverride?.assetPath,
                                diagnosticModelFilePath = gpuModelOverride?.filePath,
                            ),
                        )
                    },
                ),
            )
            publishCheckpoint()
            if (promotionGate) {
                report.put("status", STATUS_COMPLETED)
                publishCheckpoint()
                return@runBlocking
            }
            candidates.put(
                benchmarkCandidate(
                    context = context,
                    audioFile = audioFile,
                    mode = Mode.GPU_ENCODER_CPU_DECODER,
                    reference = cpu.transcript,
                    language = referenceLanguage,
                    openEngines = {
                        listOf(LiteRtGpuWhisperEngine.open(context, cpuThreadCount = 3))
                    },
                ),
            )
            publishCheckpoint()
            candidates.put(
                benchmarkCandidate(
                    context = context,
                    audioFile = audioFile,
                    mode = Mode.CPU_ENCODER_GPU_DECODER_NO_SOURCE_QUANT,
                    reference = cpu.transcript,
                    language = referenceLanguage,
                    openEngines = {
                        listOf(
                            openCpuEncoderGpuDecoderEngine(
                                context = context,
                                allowSourceQuantizedFcConvOps = false,
                                gpuModelOverride = gpuModelOverride,
                            ),
                        )
                    },
                ),
            )
            publishCheckpoint()
            candidates.put(
                benchmarkCandidate(
                    context = context,
                    audioFile = audioFile,
                    mode = Mode.CPU_ENCODER_GPU_DECODER,
                    reference = cpu.transcript,
                    language = referenceLanguage,
                    openEngines = {
                        listOf(
                            openCpuEncoderGpuDecoderEngine(
                                context = context,
                                gpuModelOverride = gpuModelOverride,
                            ),
                        )
                    },
                ),
            )
            publishCheckpoint()
            candidates.put(
                benchmarkCandidate(
                    context = context,
                    audioFile = audioFile,
                    mode = Mode.STRICT_GPU_MODEL,
                    reference = cpu.transcript,
                    language = referenceLanguage,
                    openEngines = {
                        listOf(
                            LiteRtFullGpuWhisperEngine.open(
                                context = context,
                                strictGpuOnly = true,
                                diagnosticModelAssets = gpuModelOverride?.assets,
                                diagnosticModelPath = gpuModelOverride?.assetPath,
                                diagnosticModelFilePath = gpuModelOverride?.filePath,
                            ),
                        )
                    },
                ),
            )
            publishCheckpoint()
            report
                .put("status", STATUS_COMPLETED)
        } catch (error: Throwable) {
            report
                .put("status", STATUS_FAILED)
                .put("error_type", error.javaClass.name)
                .put("error_message", sanitizedMessage(error))
        }

        publishCheckpoint()
    }

    private fun benchmarkCandidate(
        context: Context,
        audioFile: File,
        mode: Mode,
        reference: LongFormTranscript,
        language: WhisperLanguage.Forced,
        openEngines: () -> List<WhisperEngine>,
    ): JSONObject = try {
        benchmarkWithEngines(
            context = context,
            audioFile = audioFile,
            mode = mode,
            language = language,
            openEngines = openEngines,
        ).toJson(referenceText = reference.text)
    } catch (error: Throwable) {
        JSONObject()
            .put("mode", mode.id)
            .put("status", STATUS_FAILED)
            .put("coverage", mode.coverageJson())
            .put("error_type", error.javaClass.name)
            .put("error_message", sanitizedMessage(error))
    }

    private fun benchmarkWithEngines(
        context: Context,
        audioFile: File,
        mode: Mode,
        language: WhisperLanguage,
        openEngines: () -> List<WhisperEngine>,
    ): BenchmarkResult {
        val totalStarted = System.nanoTime()
        val loadStarted = System.nanoTime()
        val engines = openEngines()
        require(engines.isNotEmpty()) { "Benchmark mode ${mode.id} opened no engines." }
        val modelLoadMillis = elapsedMillis(loadStarted)
        try {
            val inferenceStarted = System.nanoTime()
            val transcript = StreamingWavChunkSource.open(audioFile).use { source ->
                val windows = (0 until source.chunkCount).map(source::window)
                LongFormTranscriber(engines).transcribeBatches(
                    windows = windows,
                    loadAudio = { window -> source.readChunk(window.index).samples },
                    language = language,
                )
            }
            val inferenceWallMillis = elapsedMillis(inferenceStarted)
            return BenchmarkResult(
                mode = mode,
                transcript = transcript,
                modelLoadMillis = modelLoadMillis,
                inferenceWallMillis = inferenceWallMillis,
                totalWallMillis = elapsedMillis(totalStarted),
                memory = memorySnapshot(context),
            )
        } finally {
            engines.asReversed().forEach { engine -> runCatching(engine::close) }
        }
    }

    private fun openCpuEncoderGpuDecoderEngine(
        context: Context,
        allowSourceQuantizedFcConvOps: Boolean = true,
        gpuModelOverride: GpuModelOverride? = null,
    ): WhisperEngine {
        val cpu = LiteRtWhisperEngine.open(context, threadCount = 3)
        try {
            val gpu = LiteRtFullGpuWhisperEngine.open(
                context = context,
                allowSourceQuantizedFcConvOps = allowSourceQuantizedFcConvOps,
                diagnosticModelAssets = gpuModelOverride?.assets,
                diagnosticModelPath = gpuModelOverride?.assetPath,
                diagnosticModelFilePath = gpuModelOverride?.filePath,
            )
            return CpuEncoderGpuDecoderEngine(cpu = cpu, gpu = gpu)
        } catch (error: Throwable) {
            cpu.close()
            throw error
        }
    }

    private fun openCpuEngines(
        context: Context,
        workerCount: Int,
        threadCount: Int,
        modelFilePath: String,
        timestampTokensEnabled: Boolean,
    ): List<WhisperEngine> {
        val engines = mutableListOf<WhisperEngine>()
        try {
            repeat(workerCount) {
                engines += LiteRtWhisperEngine.open(
                    context = context,
                    threadCount = threadCount,
                    timestampsEnabled = timestampTokensEnabled,
                    diagnosticModelFilePath = modelFilePath,
                )
            }
            return engines
        } catch (error: Throwable) {
            engines.asReversed().forEach { engine -> runCatching(engine::close) }
            throw error
        }
    }

    private class CpuEncoderGpuDecoderEngine(
        private val cpu: LiteRtWhisperEngine,
        private val gpu: LiteRtFullGpuWhisperEngine,
    ) : WhisperEngine {
        private val closed = AtomicBoolean(false)

        override fun transcribeChunk(
            audio: FloatArray,
            language: WhisperLanguage,
            shouldCancel: () -> Boolean,
            onTokenProgress: (generated: Int, limit: Int) -> Unit,
        ): WhisperChunkResult {
            check(!closed.get()) { "CPU-encoder/GPU-decoder engine is closed." }
            val started = System.nanoTime()
            val encoderState = cpu.encodeChunk(audio, shouldCancel)
            return gpu.transcribeEncodedChunk(
                encoderState = encoderState,
                language = language,
                shouldCancel = shouldCancel,
                onTokenProgress = onTokenProgress,
                startedNanos = started,
            )
        }

        override fun decodeTokens(tokens: List<Int>): String = cpu.decodeTokens(tokens)

        override fun close() {
            if (!closed.compareAndSet(false, true)) return
            runCatching(gpu::close)
            runCatching(cpu::close)
        }
    }

    private data class BenchmarkResult(
        val mode: Mode,
        val transcript: LongFormTranscript,
        val modelLoadMillis: Long,
        val inferenceWallMillis: Long,
        val totalWallMillis: Long,
        val memory: JSONObject,
    ) {
        fun toJson(referenceText: String?): JSONObject {
            val result = JSONObject()
                .put("mode", mode.id)
                .put("status", STATUS_COMPLETED)
                .put("coverage", mode.coverageJson())
                .put("model_load_millis", modelLoadMillis)
                .put("inference_wall_millis", inferenceWallMillis)
                .put("total_wall_millis", totalWallMillis)
                .put("detected_language", transcript.detectedLanguage)
                .put("text", transcript.text)
                .put("tokens", JSONArray(transcript.tokens))
                .put("token_count", transcript.tokens.size)
                .put(
                    "timestamp_token_count",
                    transcript.tokens.count {
                        it >= WhisperTimestampTokens.DEFAULT_TIMESTAMP_BEGIN_TOKEN_ID
                    },
                )
                .put("word_count", normalizedWords(transcript.text).size)
                .put("warnings", JSONArray(transcript.warnings))
                .put("memory_after_inference", memory)
                .put(
                    "chunks",
                    JSONArray().also { chunks ->
                        transcript.chunks.forEach { chunk ->
                            chunks.put(
                                JSONObject()
                                    .put("index", chunk.window.index)
                                    .put("start_sample", chunk.window.startSample)
                                    .put("end_sample", chunk.window.endSample)
                                    .put("token_count", chunk.result.tokens.size)
                                    .put("tokens", JSONArray(chunk.result.tokens))
                                    .put(
                                        "timestamp_token_count",
                                        chunk.result.tokens.count {
                                            it >= WhisperTimestampTokens
                                                .DEFAULT_TIMESTAMP_BEGIN_TOKEN_ID
                                        },
                                    )
                                    .put(
                                        "inference_millis",
                                        chunk.result.inferenceMillis,
                                    )
                                    .put("truncated", chunk.result.truncated)
                                    .put("skipped_by_vad", chunk.skippedByVad)
                                    .put(
                                        "retried_as_half_windows",
                                        chunk.retriedAsHalfWindows,
                                    ),
                            )
                        }
                    },
                )
            if (referenceText != null) {
                result.put("word_error", wordError(referenceText, transcript.text).toJson())
            } else {
                result.put(
                    "word_error",
                    WordError(
                        referenceWords = normalizedWords(transcript.text).size,
                        candidateWords = normalizedWords(transcript.text).size,
                        substitutions = 0,
                        deletions = 0,
                        insertions = 0,
                    ).toJson(),
                )
            }
            return result
        }
    }

    private fun splitGpuMode(encoderPrecision: String, decoderPrecision: String): Mode {
        val normalizedEncoder = if (encoderPrecision == GPU_PRECISION_FP16_FP32_ACCUM) {
            GPU_PRECISION_FP16_NATIVE
        } else {
            encoderPrecision
        }
        val normalizedDecoder = if (decoderPrecision == GPU_PRECISION_FP16_FP32_ACCUM) {
            GPU_PRECISION_FP16_NATIVE
        } else {
            decoderPrecision
        }
        return when (normalizedEncoder to normalizedDecoder) {
            GPU_PRECISION_FP32 to GPU_PRECISION_FP32 -> Mode.SPLIT_GPU_FP32_FP32
            GPU_PRECISION_FP32 to GPU_PRECISION_FP16_NATIVE -> Mode.SPLIT_GPU_FP32_FP16
            GPU_PRECISION_FP16_NATIVE to GPU_PRECISION_FP32 -> Mode.SPLIT_GPU_FP16_FP32
            GPU_PRECISION_FP16_NATIVE to GPU_PRECISION_FP16_NATIVE -> Mode.SPLIT_GPU_FP16_FP16
            else -> error(
                "Split benchmark supports only native FP32/FP16 precision combinations.",
            )
        }
    }

    private enum class Mode(
        val id: String,
        val frontend: String,
        val encoder: String,
        val decoder: String,
        val tokenSelection: String,
        val neuralModelStrictlyGpuOnly: Boolean,
    ) {
        CPU_2X3(
            id = "cpu-2x3",
            frontend = "CPU",
            encoder = "CPU_XNNPACK",
            decoder = "CPU_XNNPACK",
            tokenSelection = "CPU",
            neuralModelStrictlyGpuOnly = false,
        ),
        CPU_4X3(
            id = "cpu-4x3",
            frontend = "CPU",
            encoder = "CPU_XNNPACK",
            decoder = "CPU_XNNPACK",
            tokenSelection = "CPU",
            neuralModelStrictlyGpuOnly = false,
        ),
        CPU_1X3(
            id = "cpu-1x3",
            frontend = "CPU",
            encoder = "CPU_XNNPACK",
            decoder = "CPU_XNNPACK",
            tokenSelection = "CPU",
            neuralModelStrictlyGpuOnly = false,
        ),
        FULL_GPU_DOMINANT(
            id = "gpu-encoder-gpu-decoder",
            frontend = "CPU",
            encoder = "GPU_OPENCL_FP32",
            decoder = "GPU_OPENCL_FP32_WITH_CPU_GATHER",
            tokenSelection = "CPU",
            neuralModelStrictlyGpuOnly = false,
        ),
        FULL_GPU_NO_SOURCE_QUANT(
            id = "gpu-encoder-gpu-decoder-no-source-quant",
            frontend = "CPU",
            encoder = "GPU_OPENCL_FP32_NO_SOURCE_QUANT",
            decoder = "GPU_OPENCL_FP32_WITH_CPU_GATHER_NO_SOURCE_QUANT",
            tokenSelection = "CPU",
            neuralModelStrictlyGpuOnly = false,
        ),
        FULL_GPU_FP16_FP32_ACCUM_NO_SOURCE_QUANT(
            id = "gpu-fp16-storage-fp32-accum-no-source-quant",
            frontend = "CPU",
            encoder = "GPU_OPENCL_F32_F16_NO_SOURCE_QUANT",
            decoder = "GPU_OPENCL_F32_F16_WITH_CPU_GATHER_NO_SOURCE_QUANT",
            tokenSelection = "CPU",
            neuralModelStrictlyGpuOnly = false,
        ),
        FULL_GPU_FP16_NATIVE_NO_SOURCE_QUANT(
            id = "gpu-native-fp16-no-source-quant",
            frontend = "CPU",
            encoder = "GPU_OPENCL_FP16_NO_SOURCE_QUANT",
            decoder = "GPU_OPENCL_FP16_WITH_CPU_GATHER_NO_SOURCE_QUANT",
            tokenSelection = "CPU",
            neuralModelStrictlyGpuOnly = false,
        ),
        SPLIT_GPU_FP32_FP32(
            id = "split-gpu-encoder-fp32-decoder-fp32-no-source-quant",
            frontend = "CPU",
            encoder = "GPU_OPENCL_FP32_NO_SOURCE_QUANT_SPLIT_MODEL",
            decoder = "GPU_OPENCL_FP32_WITH_CPU_GATHER_SPLIT_MODEL",
            tokenSelection = "CPU",
            neuralModelStrictlyGpuOnly = false,
        ),
        SPLIT_GPU_FP32_FP16(
            id = "split-gpu-encoder-fp32-decoder-fp16-no-source-quant",
            frontend = "CPU",
            encoder = "GPU_OPENCL_FP32_NO_SOURCE_QUANT_SPLIT_MODEL",
            decoder = "GPU_OPENCL_FP16_WITH_CPU_GATHER_SPLIT_MODEL",
            tokenSelection = "CPU",
            neuralModelStrictlyGpuOnly = false,
        ),
        SPLIT_GPU_FP16_FP32(
            id = "split-gpu-encoder-fp16-decoder-fp32-no-source-quant",
            frontend = "CPU",
            encoder = "GPU_OPENCL_FP16_NO_SOURCE_QUANT_SPLIT_MODEL",
            decoder = "GPU_OPENCL_FP32_WITH_CPU_GATHER_SPLIT_MODEL",
            tokenSelection = "CPU",
            neuralModelStrictlyGpuOnly = false,
        ),
        SPLIT_GPU_FP16_FP16(
            id = "split-gpu-encoder-fp16-decoder-fp16-no-source-quant",
            frontend = "CPU",
            encoder = "GPU_OPENCL_FP16_NO_SOURCE_QUANT_SPLIT_MODEL",
            decoder = "GPU_OPENCL_FP16_WITH_CPU_GATHER_SPLIT_MODEL",
            tokenSelection = "CPU",
            neuralModelStrictlyGpuOnly = false,
        ),
        GPU_ENCODER_CPU_DECODER(
            id = "gpu-encoder-cpu-decoder",
            frontend = "CPU",
            encoder = "GPU_OPENCL_FP32",
            decoder = "CPU_XNNPACK",
            tokenSelection = "CPU",
            neuralModelStrictlyGpuOnly = false,
        ),
        CPU_ENCODER_GPU_DECODER(
            id = "cpu-encoder-gpu-decoder",
            frontend = "CPU",
            encoder = "CPU_XNNPACK",
            decoder = "GPU_OPENCL_FP32_WITH_CPU_GATHER",
            tokenSelection = "CPU",
            neuralModelStrictlyGpuOnly = false,
        ),
        CPU_ENCODER_GPU_DECODER_NO_SOURCE_QUANT(
            id = "cpu-encoder-gpu-decoder-no-source-quant",
            frontend = "CPU",
            encoder = "CPU_XNNPACK",
            decoder = "GPU_OPENCL_FP32_WITH_CPU_GATHER_NO_SOURCE_QUANT",
            tokenSelection = "CPU",
            neuralModelStrictlyGpuOnly = false,
        ),
        STRICT_GPU_MODEL(
            id = "strict-gpu-model",
            frontend = "CPU",
            encoder = "GPU_OPENCL_FP32",
            decoder = "GPU_OPENCL_FP32",
            tokenSelection = "CPU",
            neuralModelStrictlyGpuOnly = true,
        ),
        WHISPER_CPP_Q5_0_CPU(
            id = "whisper.cpp-large-v3-turbo-q5_0-cpu",
            frontend = "WHISPER_CPP_CPU",
            encoder = "WHISPER_CPP_GGML_Q5_0_CPU",
            decoder = "WHISPER_CPP_GGML_Q5_0_CPU",
            tokenSelection = "WHISPER_CPP_CPU",
            neuralModelStrictlyGpuOnly = false,
        ),
        WHISPER_CPP_Q5_0_VULKAN(
            id = "whisper.cpp-large-v3-turbo-q5_0-vulkan",
            frontend = "WHISPER_CPP_CPU",
            encoder = "WHISPER_CPP_GGML_Q5_0_VULKAN_WITH_CPU_FALLBACK_OPS",
            decoder = "WHISPER_CPP_GGML_Q5_0_VULKAN_WITH_CPU_FALLBACK_OPS",
            tokenSelection = "WHISPER_CPP_CPU",
            neuralModelStrictlyGpuOnly = false,
        ),
        WHISPER_CPP_Q8_0_CPU(
            id = "whisper.cpp-large-v3-turbo-q8_0-cpu",
            frontend = "WHISPER_CPP_CPU",
            encoder = "WHISPER_CPP_GGML_Q8_0_CPU",
            decoder = "WHISPER_CPP_GGML_Q8_0_CPU",
            tokenSelection = "WHISPER_CPP_CPU",
            neuralModelStrictlyGpuOnly = false,
        ),
        WHISPER_CPP_Q8_0_VULKAN(
            id = "whisper.cpp-large-v3-turbo-q8_0-vulkan",
            frontend = "WHISPER_CPP_CPU",
            encoder = "WHISPER_CPP_GGML_Q8_0_VULKAN_WITH_CPU_FALLBACK_OPS",
            decoder = "WHISPER_CPP_GGML_Q8_0_VULKAN_WITH_CPU_FALLBACK_OPS",
            tokenSelection = "WHISPER_CPP_CPU",
            neuralModelStrictlyGpuOnly = false,
        ),
        WHISPER_CPP_FP16_CPU(
            id = "whisper.cpp-large-v3-turbo-fp16-cpu",
            frontend = "WHISPER_CPP_CPU",
            encoder = "WHISPER_CPP_GGML_FP16_CPU",
            decoder = "WHISPER_CPP_GGML_FP16_CPU",
            tokenSelection = "WHISPER_CPP_CPU",
            neuralModelStrictlyGpuOnly = false,
        ),
        WHISPER_CPP_FP16_VULKAN(
            id = "whisper.cpp-large-v3-turbo-fp16-vulkan",
            frontend = "WHISPER_CPP_CPU",
            encoder = "WHISPER_CPP_GGML_FP16_VULKAN_WITH_CPU_FALLBACK_OPS",
            decoder = "WHISPER_CPP_GGML_FP16_VULKAN_WITH_CPU_FALLBACK_OPS",
            tokenSelection = "WHISPER_CPP_CPU",
            neuralModelStrictlyGpuOnly = false,
        ),
        ;

        fun coverageJson(): JSONObject = JSONObject()
            .put("frontend", frontend)
            .put("encoder", encoder)
            .put("decoder", decoder)
            .put("token_selection", tokenSelection)
            .put("neural_model_strictly_gpu_only", neuralModelStrictlyGpuOnly)
    }

    private data class WordError(
        val referenceWords: Int,
        val candidateWords: Int,
        val substitutions: Int,
        val deletions: Int,
        val insertions: Int,
    ) {
        val edits: Int get() = substitutions + deletions + insertions
        val rate: Double
            get() = if (referenceWords == 0) {
                if (candidateWords == 0) 0.0 else 1.0
            } else {
                edits.toDouble() / referenceWords
            }

        fun toJson(): JSONObject = JSONObject()
            .put("reference_words", referenceWords)
            .put("candidate_words", candidateWords)
            .put("substitutions", substitutions)
            .put("deletions", deletions)
            .put("insertions", insertions)
            .put("edits", edits)
            .put("rate", rate)
            .put("percent", rate * 100.0)
    }

    private data class EditCell(
        val cost: Int,
        val substitutions: Int,
        val deletions: Int,
        val insertions: Int,
    )

    private data class GpuModelOverride(
        val assets: AssetManager? = null,
        val assetPath: String? = null,
        val filePath: String? = null,
    )

    companion object {
        private const val ARG_MEMO_ID = "memoId"
        private const val ARG_RUN_ID = "runId"
        private const val ARG_CANDIDATE_ONLY = "candidateOnly"
        private const val ARG_CANDIDATE_BACKEND = "candidateBackend"
        private const val ARG_CPU_WORKERS = "cpuWorkers"
        private const val ARG_CPU_THREADS = "cpuThreads"
        private const val ARG_WHISPER_CPP_THREADS = "whisperCppThreads"
        private const val ARG_WHISPER_CPP_BACKEND = "whisperCppBackend"
        private const val ARG_TIMESTAMP_TOKENS_ENABLED = "timestampTokensEnabled"
        private const val ARG_EXTERNAL_MODEL_NAME = "externalModelName"
        private const val ARG_EXTERNAL_MODEL_SHA256 = "externalModelSha256"
        private const val ARG_EXTERNAL_ENCODER_MODEL_NAME = "externalEncoderModelName"
        private const val ARG_EXTERNAL_ENCODER_MODEL_SHA256 = "externalEncoderModelSha256"
        private const val ARG_EXTERNAL_DECODER_MODEL_NAME = "externalDecoderModelName"
        private const val ARG_EXTERNAL_DECODER_MODEL_SHA256 = "externalDecoderModelSha256"
        private const val ARG_EXTERNAL_PROJECTION_MODEL_NAME = "externalProjectionModelName"
        private const val ARG_EXTERNAL_PROJECTION_MODEL_SHA256 =
            "externalProjectionModelSha256"
        private const val ARG_EXTERNAL_CROSS_CACHE_MODEL_NAME = "externalCrossCacheModelName"
        private const val ARG_EXTERNAL_CROSS_CACHE_MODEL_SHA256 =
            "externalCrossCacheModelSha256"
        private const val ARG_EXTERNAL_AUDIO_NAME = "externalAudioName"
        private const val ARG_FORCED_LANGUAGE = "forcedLanguage"
        private const val ARG_GPU_PRECISION = "gpuPrecision"
        private const val ARG_ENCODER_GPU_PRECISION = "encoderGpuPrecision"
        private const val ARG_DECODER_GPU_PRECISION = "decoderGpuPrecision"
        private const val ARG_USE_FULL_FLOAT_DECODER = "useFullFloatDecoder"
        private const val ARG_PROMOTION_GATE = "promotionGate"
        private const val FULL_FLOAT_DECODER_ASSET =
            "whisper-full-float-decoder/decoder.tflite"
        private const val REPORT_DIRECTORY = "gpu-wer"
        private const val EXTERNAL_MODEL_DIRECTORY = "gpu-models"
        private const val WHISPER_CPP_MODEL_DIRECTORY = "whispercpp-models"
        private const val EXTERNAL_AUDIO_DIRECTORY = "gpu-audio"
        private const val STATUS_RUNNING = "running"
        private const val STATUS_COMPLETED = "completed"
        private const val STATUS_FAILED = "failed"
        private const val GPU_PRECISION_FP32 = "fp32"
        private const val GPU_PRECISION_FP16_NATIVE = "fp16-native"
        private const val GPU_PRECISION_FP16_FP32_ACCUM = "fp16-fp32-accum"
        private const val CANDIDATE_BACKEND_GPU = "gpu"
        private const val CANDIDATE_BACKEND_CPU = "cpu"
        private const val CANDIDATE_BACKEND_WHISPER_CPP = "whispercpp"
        private val WORD_PATTERN =
            Regex("""[\p{L}\p{N}]+(?:['’\-][\p{L}\p{N}]+)*""")
        private val SAFE_FILE_NAME = Regex("""[A-Za-z0-9][A-Za-z0-9._-]{0,127}""")
        private val SAFE_LANGUAGE = Regex("""[a-z]{2,3}""")
        private val SAFE_SHA256 = Regex("""[0-9a-f]{64}""")

        private fun requireStagedFile(
            context: Context,
            directory: String,
            name: String,
            extension: String,
        ): File {
            require(SAFE_FILE_NAME.matches(name) && name.endsWith(extension)) {
                "Invalid staged file name."
            }
            val parent = File(context.filesDir, directory).canonicalFile
            val file = File(parent, name).canonicalFile
            require(file.parentFile == parent && file.isFile) {
                "Staged file is missing."
            }
            return file
        }

        private fun wordError(reference: String, candidate: String): WordError {
            val expected = normalizedWords(reference)
            val actual = normalizedWords(candidate)
            val rows = Array(expected.size + 1) {
                Array(actual.size + 1) { EditCell(0, 0, 0, 0) }
            }
            for (index in 1..expected.size) {
                rows[index][0] = EditCell(index, 0, index, 0)
            }
            for (index in 1..actual.size) {
                rows[0][index] = EditCell(index, 0, 0, index)
            }
            for (expectedIndex in 1..expected.size) {
                for (actualIndex in 1..actual.size) {
                    val diagonal = rows[expectedIndex - 1][actualIndex - 1]
                    val substitution = if (
                        expected[expectedIndex - 1] == actual[actualIndex - 1]
                    ) {
                        diagonal
                    } else {
                        diagonal.copy(
                            cost = diagonal.cost + 1,
                            substitutions = diagonal.substitutions + 1,
                        )
                    }
                    val deletionBase = rows[expectedIndex - 1][actualIndex]
                    val deletion = deletionBase.copy(
                        cost = deletionBase.cost + 1,
                        deletions = deletionBase.deletions + 1,
                    )
                    val insertionBase = rows[expectedIndex][actualIndex - 1]
                    val insertion = insertionBase.copy(
                        cost = insertionBase.cost + 1,
                        insertions = insertionBase.insertions + 1,
                    )
                    rows[expectedIndex][actualIndex] =
                        listOf(substitution, deletion, insertion).minWith(
                            compareBy<EditCell>(
                                { it.cost },
                                { it.substitutions },
                                { it.deletions },
                                { it.insertions },
                            ),
                        )
                }
            }
            val result = rows[expected.size][actual.size]
            return WordError(
                referenceWords = expected.size,
                candidateWords = actual.size,
                substitutions = result.substitutions,
                deletions = result.deletions,
                insertions = result.insertions,
            )
        }

        private fun normalizedWords(text: String): List<String> =
            WORD_PATTERN.findAll(text.lowercase(Locale.ROOT))
                .map(MatchResult::value)
                .toList()

        private fun elapsedMillis(startedNanos: Long): Long =
            (System.nanoTime() - startedNanos) / 1_000_000L

        private fun sha256(file: File): String {
            val digest = MessageDigest.getInstance("SHA-256")
            file.inputStream().buffered().use { stream ->
                val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                while (true) {
                    val count = stream.read(buffer)
                    if (count < 0) break
                    digest.update(buffer, 0, count)
                }
            }
            return digest.digest().joinToString("") { "%02x".format(it) }
        }

        private fun requireArgument(value: String?, label: String): String {
            val result = value?.trim().orEmpty()
            require(result.matches(Regex("[A-Za-z0-9-]+"))) {
                "$label is missing or invalid."
            }
            return result
        }

        private fun requireSafeRunId(value: String?): String {
            val result = value?.trim().orEmpty()
            require(result.matches(Regex("[A-Za-z0-9-]+"))) {
                "$ARG_RUN_ID is missing or invalid."
            }
            return result
        }

        private fun sanitizedMessage(error: Throwable): String =
            error.message
                ?.lineSequence()
                ?.firstOrNull()
                ?.trim()
                ?.take(500)
                ?.takeIf(String::isNotEmpty)
                ?: error.javaClass.simpleName

        private fun memorySnapshot(context: Context): JSONObject {
            val memoryInfo = Debug.MemoryInfo()
            Debug.getMemoryInfo(memoryInfo)
            val runtime = Runtime.getRuntime()
            val thermalStatus = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                context.getSystemService(PowerManager::class.java)?.currentThermalStatus
            } else {
                null
            }
            return JSONObject()
                .put("total_pss_kib", memoryInfo.totalPss)
                .put("native_heap_allocated_bytes", Debug.getNativeHeapAllocatedSize())
                .put("java_heap_used_bytes", runtime.totalMemory() - runtime.freeMemory())
                .put("java_heap_committed_bytes", runtime.totalMemory())
                .put("java_heap_max_bytes", runtime.maxMemory())
                .put("thermal_status", thermalStatus ?: JSONObject.NULL)
        }
    }
}
