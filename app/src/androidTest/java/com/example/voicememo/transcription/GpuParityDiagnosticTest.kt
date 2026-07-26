package com.example.voicememo.transcription

import android.content.Context
import android.os.Build
import android.os.Bundle
import android.os.Debug
import android.os.PowerManager
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.voicememo.VoiceMemoApplication
import com.example.voicememo.data.TranscriptionJobStatus
import com.google.ai.edge.litert.CompiledModel
import java.io.File
import java.security.MessageDigest
import kotlin.math.abs
import kotlin.math.sqrt
import kotlinx.coroutines.runBlocking
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Non-persisting, device-only accelerator diagnostic.
 *
 * This reads a previously completed CPU chunk as the token oracle, but never
 * creates or updates transcription rows. CPU and GPU engines are opened
 * sequentially to keep peak memory bounded.
 */
@RunWith(AndroidJUnit4::class)
class GpuParityDiagnosticTest {
    @Test
    fun compareEncoderAndDecoderBoundaries() = runBlocking {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val arguments = InstrumentationRegistry.getArguments()
        val context = instrumentation.targetContext
        val application = context.applicationContext as VoiceMemoApplication
        val repository = application.container.memoRepository
        val memoId = requireArgument(arguments.getString(ARG_MEMO_ID), ARG_MEMO_ID)
        val referenceJobId = requireArgument(
            arguments.getString(ARG_REFERENCE_JOB),
            ARG_REFERENCE_JOB,
        )
        val diagnosticRunId = requireDiagnosticRunId(
            arguments.getString(ARG_DIAGNOSTIC_RUN_ID),
        )
        val chunkIndex = arguments.getString(ARG_CHUNK_INDEX)?.toIntOrNull() ?: 0
        val runAutoregressive =
            arguments.getString(ARG_RUN_AUTOREGRESSIVE)?.toBooleanStrictOrNull() ?: true
        val requestedTeacherForcedSteps =
            arguments.getString(ARG_TEACHER_FORCED_STEPS)
        val useLiveCpuReference =
            arguments.getString(ARG_LIVE_CPU_REFERENCE)?.toBooleanStrictOrNull() ?: false
        val speculativeBlockSize =
            arguments.getString(ARG_SPECULATIVE_BLOCK_SIZE)?.toIntOrNull() ?: 0
        val runDecoderSplit =
            arguments.getString(ARG_RUN_DECODER_SPLIT)?.toBooleanStrictOrNull() ?: false
        val reverseDecoderSplit =
            arguments.getString(ARG_REVERSE_DECODER_SPLIT)
                ?.toBooleanStrictOrNull() ?: false
        val runGpuEncoderCpuDecoder =
            arguments.getString(ARG_RUN_GPU_ENCODER_CPU_DECODER)
                ?.toBooleanStrictOrNull() ?: false
        val splitAllowSourceQuantization =
            arguments.getString(ARG_SOURCE_QUANTIZATION)?.toBooleanStrictOrNull() ?: true
        val splitConstantTensorSharing =
            arguments.getString(ARG_CONSTANT_TENSOR_SHARING)
                ?.toBooleanStrictOrNull() ?: true
        val splitStrictGpuOnly =
            arguments.getString(ARG_STRICT_GPU_ONLY)?.toBooleanStrictOrNull() ?: false
        val splitGpuBackend = parseSplitGpuBackend(arguments.getString(ARG_GPU_BACKEND))
        val decoderSplitStage =
            arguments.getString(ARG_DECODER_SPLIT_STAGE) ?: SPLIT_STAGE_POST_LAYER_NORM
        require(decoderSplitStage in SUPPORTED_SPLIT_STAGES) {
            "$ARG_DECODER_SPLIT_STAGE must be one of " +
                SUPPORTED_SPLIT_STAGES.joinToString()
        }
        val maximumSpeculativeBlockSize = if (runDecoderSplit) {
            MAX_BOUNDARY_SPECULATIVE_BLOCK_SIZE
        } else {
            MAX_SPECULATIVE_BLOCK_SIZE
        }
        require(speculativeBlockSize in 0..maximumSpeculativeBlockSize) {
            "$ARG_SPECULATIVE_BLOCK_SIZE must be between 0 and " +
                "$maximumSpeculativeBlockSize."
        }
        val reportDirectory = File(context.cacheDir, REPORT_DIRECTORY).apply { mkdirs() }
        val reportFile = File(
            reportDirectory,
            "matrix-chunk-$chunkIndex-$diagnosticRunId.json",
        )
        val temporaryReport = File(reportDirectory, "${reportFile.name}.tmp")
        reportFile.delete()
        temporaryReport.delete()

        val report = JSONObject()
            .put("diagnostic_run_id", diagnosticRunId)
            .put("memo_id", memoId)
            .put("reference_job_id", referenceJobId)
            .put("chunk_index", chunkIndex)
            .put("persisted_output", false)
            .put("run_autoregressive", runAutoregressive)
            .put(
                "requested_teacher_forced_steps",
                requestedTeacherForcedSteps ?: JSONObject.NULL,
            )
            .put("use_live_cpu_reference", useLiveCpuReference)
            .put("speculative_block_size", speculativeBlockSize)
            .put("run_decoder_split", runDecoderSplit)
            .put("reverse_decoder_split", reverseDecoderSplit)
            .put("run_gpu_encoder_cpu_decoder", runGpuEncoderCpuDecoder)
            .put("decoder_split_stage", decoderSplitStage)

        try {
            require(!reverseDecoderSplit || runDecoderSplit) {
                "Reverse decoder split requires decoder-split diagnostics."
            }
            require(
                !reverseDecoderSplit ||
                    WhisperDecoderBoundaryStage.fromId(decoderSplitStage) != null,
            ) {
                "Reverse decoder split requires a generated decoder-boundary stage."
            }
            require(
                !runDecoderSplit ||
                    speculativeBlockSize == 0 ||
                    decoderSplitStage == SPLIT_STAGE_AFTER_1_BLOCK,
            ) {
                "Decoder-boundary speculative diagnostics require " +
                    "$SPLIT_STAGE_AFTER_1_BLOCK."
            }
            require(!runDecoderSplit || !runGpuEncoderCpuDecoder) {
                "Decoder-split and GPU-encoder/CPU-decoder diagnostics cannot run together."
            }
            val gpuProfiles = if (runDecoderSplit) {
                emptyList()
            } else {
                parseGpuProfiles(arguments)
            }
            require(
                runDecoderSplit ||
                    speculativeBlockSize == 0 ||
                    gpuProfiles.size == 1,
            ) {
                "Speculative decoding requires exactly one GPU profile per run."
            }
            report.put(
                "requested_gpu_profiles",
                JSONArray().also { profiles ->
                    gpuProfiles.forEach { profile -> profiles.put(profile.name) }
                },
            )
            val memo = requireNotNull(repository.getMemo(memoId)) {
                "Unknown memo $memoId."
            }
            val referenceJob = requireNotNull(repository.getTranscriptionJob(referenceJobId)) {
                "Unknown reference job $referenceJobId."
            }
            require(referenceJob.memoId == memoId) {
                "Reference job belongs to a different memo."
            }
            require(referenceJob.status == TranscriptionJobStatus.SUCCEEDED) {
                "Reference job must be SUCCEEDED."
            }
            if (!useLiveCpuReference) {
                require(
                    WhisperInferenceFingerprint.isCurrentCpuReference(
                        referenceJob.inferenceFingerprint,
                    ),
                ) {
                    "Reference job must use the current model, tokenizer, pipeline, " +
                        "and CPU/XNNPACK, or the diagnostic must request a live CPU reference."
                }
            }
            val referenceChunk = repository.completedTranscriptChunks(referenceJobId)
                .singleOrNull { it.chunkIndex == chunkIndex }
                ?: error("Reference chunk $chunkIndex is not complete.")
            val referenceLanguage = referenceChunk.detectedLanguage
                ?: referenceJob.detectedLanguage
                ?: error("Reference job has no detected language.")
            val language = WhisperLanguage.Forced(referenceLanguage)
            val audio = StreamingWavChunkSource.open(File(memo.audioPath)).use { source ->
                source.readChunk(chunkIndex).also { chunk ->
                    require(
                        chunk.window.startSample * 1_000L /
                            WhisperFeatureExtractor.SAMPLE_RATE ==
                            referenceChunk.windowStartMillis,
                    )
                    require(
                        chunk.window.endSample * 1_000L /
                            WhisperFeatureExtractor.SAMPLE_RATE ==
                            referenceChunk.windowEndMillis,
                    )
                }.samples
            }
            report.put("audio_samples", audio.size)

            val cpuDiagnostic = LiteRtWhisperEngine.open(context, threadCount = 3).use { engine ->
                val cpuEncodeStarted = System.nanoTime()
                val encoderState = engine.encodeChunk(audio)
                report.put("cpu_encode_millis", elapsedMillis(cpuEncodeStarted))
                val liveCpuReference = if (useLiveCpuReference) {
                    engine.transcribeEncodedChunk(
                        encoderState = encoderState,
                        language = language,
                    ).also { result ->
                        report.put(
                            "live_cpu_reference_vs_persisted",
                            compareTokens(referenceChunk.tokenIds, result),
                        )
                    }
                } else {
                    null
                }
                val referenceTokens =
                    liveCpuReference?.tokens ?: referenceChunk.tokenIds
                val referenceTruncated =
                    liveCpuReference?.truncated ?: referenceChunk.truncated
                val defaultTeacherForcedStepCount =
                    if (runDecoderSplit && reverseDecoderSplit) {
                        referenceTokens.size
                    } else {
                        TEACHER_FORCED_STEP_COUNT
                    }
                val teacherForcedStepCount = resolveTeacherForcedStepCount(
                    requested = requestedTeacherForcedSteps,
                    defaultCount = defaultTeacherForcedStepCount,
                    referenceTokenCount = referenceTokens.size,
                )
                report
                    .put("reference_token_count", referenceTokens.size)
                    .put("reference_token_sha256", tokenDigest(referenceTokens))
                    .put("teacher_forced_step_count", teacherForcedStepCount)
                val rows = (0 until teacherForcedStepCount).map { step ->
                    val started = System.nanoTime()
                    TeacherForcedRow(
                        row = engine.teacherForcedLogitsRow(
                            encoderState = encoderState,
                            language = language,
                            referenceTokenPrefix = referenceTokens.take(step),
                        ),
                        inferenceMillis = elapsedMillis(started),
                    )
                }
                val referenceDecode = when {
                    liveCpuReference != null -> {
                        compareTokens(referenceTokens, liveCpuReference)
                    }
                    speculativeBlockSize > 0 ||
                        (runDecoderSplit && runAutoregressive) ||
                        runGpuEncoderCpuDecoder -> {
                        val result = engine.transcribeEncodedChunk(
                            encoderState = encoderState,
                            language = language,
                        )
                        compareTokens(referenceTokens, result).also { comparison ->
                            require(comparison.getBoolean("exact_token_match")) {
                                "The live CPU decoder did not reproduce the trusted " +
                                    "reference chunk."
                            }
                        }
                    }
                    else -> null
                }
                CpuDiagnostic(
                    encoderState = encoderState,
                    teacherForcedRows = rows,
                    referenceTokens = referenceTokens,
                    referenceTruncated = referenceTruncated,
                    referenceDecode = referenceDecode,
                )
            }
            cpuDiagnostic.referenceDecode?.let { referenceDecode ->
                report
                    .put("cpu_reference_decode", referenceDecode)
                    .put("cpu_only_memory_after_decode", processMemorySnapshot(context))
            }

            val candidates = JSONArray()
            var successfulCandidateCount = 0
            if (runDecoderSplit) {
                val candidate = runDecoderSplitCandidate(
                    context = context,
                    diagnosticAssets = instrumentation.context.assets,
                    language = language,
                    referenceTokens = cpuDiagnostic.referenceTokens,
                    referenceTruncated = cpuDiagnostic.referenceTruncated,
                    cpuDiagnostic = cpuDiagnostic,
                    runAutoregressive = runAutoregressive,
                    allowSourceQuantizedFcConvOps = splitAllowSourceQuantization,
                    constantTensorSharing = splitConstantTensorSharing,
                    gpuBackend = splitGpuBackend,
                    strictGpuOnly = splitStrictGpuOnly,
                    splitStage = decoderSplitStage,
                    speculativeBlockSize = speculativeBlockSize,
                    reverseDecoderSplit = reverseDecoderSplit,
                )
                candidates.put(candidate)
                if (candidate.optString("status") == STATUS_COMPLETED) {
                    successfulCandidateCount++
                }
            } else {
                gpuProfiles.forEach { profile ->
                    val candidate = runGpuCandidate(
                        context = context,
                        profile = profile,
                        audio = audio,
                        language = language,
                        referenceTokens = cpuDiagnostic.referenceTokens,
                        referenceTruncated = cpuDiagnostic.referenceTruncated,
                        cpuDiagnostic = cpuDiagnostic,
                        teacherForcedStepCount = cpuDiagnostic.teacherForcedRows.size,
                        runAutoregressive = runAutoregressive,
                        speculativeBlockSize = speculativeBlockSize,
                        runGpuEncoderCpuDecoder = runGpuEncoderCpuDecoder,
                    )
                    candidates.put(candidate)
                    if (candidate.optString("status") == STATUS_COMPLETED) {
                        successfulCandidateCount++
                    }
                }
            }
            report
                .put("candidates", candidates)
                .put("candidate_count", candidates.length())
                .put("successful_candidate_count", successfulCandidateCount)
                .put("status", STATUS_COMPLETED)
        } catch (error: Throwable) {
            report
                .put("status", STATUS_FAILED)
                .put("error_type", error.javaClass.name)
                .put("error_message", sanitizedErrorMessage(error))
        }

        temporaryReport.writeText(report.toString(2))
        check(temporaryReport.renameTo(reportFile)) {
            "Could not publish GPU diagnostic report."
        }
    }

    private fun runGpuCandidate(
        context: android.content.Context,
        profile: GpuProfile,
        audio: FloatArray,
        language: WhisperLanguage.Forced,
        referenceTokens: List<Int>,
        referenceTruncated: Boolean,
        cpuDiagnostic: CpuDiagnostic,
        teacherForcedStepCount: Int,
        runAutoregressive: Boolean,
        speculativeBlockSize: Int,
        runGpuEncoderCpuDecoder: Boolean,
    ): JSONObject {
        val candidate = JSONObject()
            .put("name", profile.name)
            .put("options", profile.options.toJson())
        try {
            val gpuModelLoadStarted = System.nanoTime()
            var gpuEncoderStateForCpuDecode: FloatArray? = null
            var gpuEncodeMillisForCpuDecode = 0L
            var gpuCloseStartedNanos = 0L
            LiteRtFullGpuWhisperEngine.open(
                context = context,
                diagnosticHintWaitingForCompletion =
                    profile.options.hintWaitingForCompletion,
                diagnosticKernelBatchSize = profile.options.kernelBatchSize,
                allowSourceQuantizedFcConvOps =
                    profile.options.allowSourceQuantizedFcConvOps,
                diagnosticBackend = profile.options.backend,
                diagnosticBufferStorageType = profile.options.bufferStorageType,
                diagnosticPreferTextureWeights = profile.options.preferTextureWeights,
                diagnosticBenchmarkMode = profile.options.benchmarkMode,
            ).use { gpuEngine ->
                candidate.put(
                    "gpu_model_load_millis",
                    elapsedMillis(gpuModelLoadStarted),
                )
                candidate.put("gpu_only_memory_after_open", processMemorySnapshot(context))

                val gpuEncodeStarted = System.nanoTime()
                val gpuEncoderState = gpuEngine.encodeChunk(audio)
                val gpuEncodeMillis = elapsedMillis(gpuEncodeStarted)
                candidate
                    .put("gpu_encode_millis", gpuEncodeMillis)
                    .put(
                        "encoder_comparison",
                        compareFloatArrays(cpuDiagnostic.encoderState, gpuEncoderState),
                    )
                if (runGpuEncoderCpuDecoder) {
                    gpuEncoderStateForCpuDecode = gpuEncoderState
                    gpuEncodeMillisForCpuDecode = gpuEncodeMillis
                }

                val teacherForcedSteps = JSONArray()
                (0 until teacherForcedStepCount).forEach { step ->
                    val gpuStarted = System.nanoTime()
                    val gpuRow = gpuEngine.teacherForcedLogitsRow(
                        encoderState = cpuDiagnostic.encoderState,
                        language = language,
                        referenceTokenPrefix = referenceTokens.take(step),
                    )
                    teacherForcedSteps.put(
                        compareLogitRows(
                            cpu = cpuDiagnostic.teacherForcedRows[step],
                            gpu = gpuRow,
                            step = step,
                            referenceToken = referenceTokens[step],
                            gpuInferenceMillis = elapsedMillis(gpuStarted),
                        ),
                    )
                }
                candidate.put(
                    "teacher_forced_cpu_encoder_gpu_decoder",
                    JSONObject()
                        .put("step_count", teacherForcedStepCount)
                        .put("steps", teacherForcedSteps)
                        .put(
                            "summary",
                            summarizeTeacherForcedSteps(teacherForcedSteps),
                        ),
                )

                if (runAutoregressive) {
                    val cpuStateDecode = gpuEngine.transcribeEncodedChunk(
                        encoderState = cpuDiagnostic.encoderState,
                        language = language,
                    )
                    candidate.put(
                        "cpu_encoder_gpu_decoder",
                        compareTokens(referenceTokens, cpuStateDecode),
                    )

                    val gpuStateDecode = gpuEngine.transcribeEncodedChunk(
                        encoderState = gpuEncoderState,
                        language = language,
                    )
                    candidate.put(
                        "gpu_encoder_gpu_decoder",
                        compareTokens(referenceTokens, gpuStateDecode),
                    )
                }

                if (speculativeBlockSize > 0) {
                    candidate.put(
                        "cpu_verified_speculative_decode",
                        runCpuVerifiedSpeculativeDecode(
                            context = context,
                            gpuEngine = gpuEngine,
                            encoderState = cpuDiagnostic.encoderState,
                            language = language,
                            referenceTokens = referenceTokens,
                            referenceTruncated = referenceTruncated,
                            blockSize = speculativeBlockSize,
                        ),
                    )
                }
                if (runGpuEncoderCpuDecoder) {
                    gpuCloseStartedNanos = System.nanoTime()
                }
            }
            if (runGpuEncoderCpuDecoder) {
                val gpuModelCloseMillis = elapsedMillis(gpuCloseStartedNanos)
                candidate.put(
                    "gpu_encoder_cpu_decoder",
                    runGpuEncoderCpuDecoderLane(
                        context = context,
                        gpuEncoderState = requireNotNull(gpuEncoderStateForCpuDecode),
                        language = language,
                        referenceTokens = referenceTokens,
                        referenceTruncated = referenceTruncated,
                        gpuEncodeMillis = gpuEncodeMillisForCpuDecode,
                        gpuModelCloseMillis = gpuModelCloseMillis,
                    ),
                )
            }
            candidate.put("status", STATUS_COMPLETED)
        } catch (error: Throwable) {
            candidate
                .put("status", STATUS_FAILED)
                .put("error_type", error.javaClass.name)
                .put("error_message", sanitizedErrorMessage(error))
        }
        return candidate
    }

    /**
     * Closes the GPU model before opening the CPU decoder. Keeping only the
     * encoder-state FloatArray across that boundary makes this lane suitable
     * for memory-constrained devices and prevents a diagnostic from silently
     * becoming the production backend.
     *
     * The result deliberately omits [WhisperChunkResult.text]. Token IDs are
     * safe to compare with the persisted CPU oracle without copying transcript
     * text into the diagnostic report.
     */
    private fun runGpuEncoderCpuDecoderLane(
        context: Context,
        gpuEncoderState: FloatArray,
        language: WhisperLanguage.Forced,
        referenceTokens: List<Int>,
        referenceTruncated: Boolean,
        gpuEncodeMillis: Long,
        gpuModelCloseMillis: Long,
    ): JSONObject {
        val cpuModelLoadStarted = System.nanoTime()
        return LiteRtWhisperEngine.open(context, threadCount = 3).use { cpuDecoder ->
            val cpuModelLoadMillis = elapsedMillis(cpuModelLoadStarted)
            val cpuDecodeStarted = System.nanoTime()
            val result = cpuDecoder.transcribeEncodedChunk(
                encoderState = gpuEncoderState,
                language = language,
            )
            val cpuDecodeMillis = elapsedMillis(cpuDecodeStarted)
            val comparison = WhisperSequentialHybridDiagnostic.compare(
                referenceTokens = referenceTokens,
                referenceTruncated = referenceTruncated,
                candidateTokens = result.tokens,
                candidateTruncated = result.truncated,
                timings = WhisperSequentialHybridDiagnostic.Timings(
                    gpuEncodeMillis = gpuEncodeMillis,
                    gpuModelCloseMillis = gpuModelCloseMillis,
                    cpuDecoderModelLoadMillis = cpuModelLoadMillis,
                    cpuDecodeMillis = cpuDecodeMillis,
                ),
            )
            JSONObject()
                .put("token_ids", JSONArray(comparison.tokenIds))
                .put("token_count", comparison.tokenIds.size)
                .put("token_sha256", comparison.tokenSha256)
                .put(
                    "first_token_mismatch",
                    comparison.firstTokenMismatch ?: JSONObject.NULL,
                )
                .put("exact_token_match", comparison.exactTokenMatch)
                .put("truncated", comparison.truncated)
                .put("exact_termination_match", comparison.exactTerminationMatch)
                .put("exact_cpu_result_match", comparison.exactCpuResultMatch)
                .put("gpu_encode_millis", comparison.timings.gpuEncodeMillis)
                .put(
                    "gpu_model_close_millis",
                    comparison.timings.gpuModelCloseMillis,
                )
                .put(
                    "cpu_decoder_model_load_millis",
                    comparison.timings.cpuDecoderModelLoadMillis,
                )
                .put("cpu_decode_millis", comparison.timings.cpuDecodeMillis)
                .put(
                    "inference_component_millis",
                    comparison.timings.inferenceComponentMillis,
                )
                .put(
                    "sequential_component_millis",
                    comparison.timings.sequentialComponentMillis,
                )
        }
    }

    private fun runDecoderSplitCandidate(
        context: Context,
        diagnosticAssets: android.content.res.AssetManager,
        language: WhisperLanguage.Forced,
        referenceTokens: List<Int>,
        referenceTruncated: Boolean,
        cpuDiagnostic: CpuDiagnostic,
        runAutoregressive: Boolean,
        allowSourceQuantizedFcConvOps: Boolean,
        constantTensorSharing: Boolean,
        gpuBackend: WhisperSplitDecoderGpuBackend,
        strictGpuOnly: Boolean,
        splitStage: String,
        speculativeBlockSize: Int,
        reverseDecoderSplit: Boolean,
    ): JSONObject {
        val boundaryStage = WhisperDecoderBoundaryStage.fromId(splitStage)
        if (boundaryStage != null) {
            return runDecoderBoundaryCandidate(
                context = context,
                diagnosticAssets = diagnosticAssets,
                stage = boundaryStage,
                language = language,
                referenceTokens = referenceTokens,
                referenceTruncated = referenceTruncated,
                cpuDiagnostic = cpuDiagnostic,
                runAutoregressive = runAutoregressive,
                allowSourceQuantizedFcConvOps = allowSourceQuantizedFcConvOps,
                constantTensorSharing = constantTensorSharing,
                gpuBackend = gpuBackend,
                strictGpuOnly = strictGpuOnly,
                speculativeBlockSize = speculativeBlockSize,
                reverseDecoderSplit = reverseDecoderSplit,
            )
        }
        val splitAssets = when (splitStage) {
            SPLIT_STAGE_POST_LAYER_NORM -> WhisperSplitDecoderAssets.Default
            SPLIT_STAGE_PRE_LAYER_NORM -> WhisperSplitDecoderAssets(
                decoderCore =
                    "${WhisperSplitDecoderAssets.DEFAULT_DIRECTORY}/" +
                        "whisper_large_v3_turbo_30s_i4_decode_pre_layer_norm.tflite",
                activeRowProjection =
                    "${WhisperSplitDecoderAssets.DEFAULT_DIRECTORY}/" +
                        "whisper_large_v3_turbo_active_row_layer_norm_projection_i8.tflite",
            )
            else -> error("Unsupported decoder split stage $splitStage.")
        }
        val candidate = JSONObject()
            .put("name", DECODER_SPLIT_PROFILE)
            .put(
                "options",
                JSONObject()
                    .put("gpu_core", "opencl-fp32")
                    .put("split_stage", splitStage)
                    .put(
                        "allow_src_quantized_fc_conv_ops",
                        allowSourceQuantizedFcConvOps,
                    )
                    .put("constant_tensor_sharing", constantTensorSharing)
                    .put("gpu_backend", gpuBackend.name.lowercase())
                    .put("strict_gpu_only", strictGpuOnly)
                    .put("active_row_projection", "cpu-xnnpack-3"),
            )
        try {
            val cpuHiddenRows = mutableListOf<FloatArray>()
            val cpuControl = JSONObject()
            val cpuLoadStarted = System.nanoTime()
            LiteRtSplitDecoderDiagnostic.open(
                configurationAssets = context.assets,
                diagnosticAssets = diagnosticAssets,
                assets = splitAssets,
                coreBackend = WhisperSplitDecoderCoreBackend.CPU_XNNPACK,
                coreCpuThreadCount = 3,
            ).use { splitCpu ->
                cpuControl
                    .put("model_load_millis", elapsedMillis(cpuLoadStarted))
                    .put("memory_after_open", processMemorySnapshot(context))
                val steps = JSONArray()
                (0 until TEACHER_FORCED_STEP_COUNT).forEach { stepIndex ->
                    val step = splitCpu.teacherForcedStep(
                        encoderState = cpuDiagnostic.encoderState,
                        language = language,
                        referenceTokenPrefix = referenceTokens.take(stepIndex),
                    )
                    cpuHiddenRows += step.activeHiddenRow
                    val comparison = compareSplitProjectedStep(
                        cpu = cpuDiagnostic.teacherForcedRows[stepIndex],
                        split = step,
                        step = stepIndex,
                        referenceToken = referenceTokens[stepIndex],
                    )
                    steps.put(comparison)
                    require(comparison.getBoolean("exact_logits_match")) {
                        "CPU split control step $stepIndex did not reproduce the trusted " +
                            "CPU logits bit-exactly."
                    }
                }
                cpuControl.put("teacher_forced_steps", steps)
                if (runAutoregressive) {
                    val fullDecode = runSplitAutoregressive(
                        engine = splitCpu,
                        encoderState = cpuDiagnostic.encoderState,
                        language = language,
                        referenceTokens = referenceTokens,
                        referenceTruncated = referenceTruncated,
                    )
                    cpuControl.put("autoregressive", fullDecode)
                    require(fullDecode.getBoolean("exact_cpu_result_match")) {
                        "CPU decoder core plus active-row projection did not reproduce " +
                            "the trusted CPU reference."
                    }
                }
                cpuControl.put("memory_after_decode", processMemorySnapshot(context))
            }
            candidate.put("cpu_control", cpuControl)

            val gpuResult = JSONObject()
            val gpuLoadStarted = System.nanoTime()
            LiteRtSplitDecoderDiagnostic.open(
                configurationAssets = context.assets,
                diagnosticAssets = diagnosticAssets,
                assets = splitAssets,
                coreBackend = WhisperSplitDecoderCoreBackend.GPU_COMPILED_MODEL,
                allowSourceQuantizedFcConvOps = allowSourceQuantizedFcConvOps,
                constantTensorSharing = constantTensorSharing,
                gpuBackend = gpuBackend,
                strictGpuOnly = strictGpuOnly,
            ).use { splitGpu ->
                gpuResult
                    .put("model_load_millis", elapsedMillis(gpuLoadStarted))
                    .put("memory_after_open", processMemorySnapshot(context))
                val steps = JSONArray()
                (0 until TEACHER_FORCED_STEP_COUNT).forEach { stepIndex ->
                    val step = splitGpu.teacherForcedStep(
                        encoderState = cpuDiagnostic.encoderState,
                        language = language,
                        referenceTokenPrefix = referenceTokens.take(stepIndex),
                    )
                    steps.put(
                        compareSplitProjectedStep(
                            cpu = cpuDiagnostic.teacherForcedRows[stepIndex],
                            split = step,
                            step = stepIndex,
                            referenceToken = referenceTokens[stepIndex],
                        ).put(
                            "hidden_cpu_gpu_comparison",
                            compareFloatArrays(
                                reference = cpuHiddenRows[stepIndex],
                                candidate = step.activeHiddenRow,
                            ),
                        ),
                    )
                }
                gpuResult.put("teacher_forced_steps", steps)
                if (runAutoregressive) {
                    gpuResult.put(
                        "autoregressive",
                        runSplitAutoregressive(
                            engine = splitGpu,
                            encoderState = cpuDiagnostic.encoderState,
                            language = language,
                            referenceTokens = referenceTokens,
                            referenceTruncated = referenceTruncated,
                        ),
                    )
                }
                gpuResult.put("memory_after_decode", processMemorySnapshot(context))
            }
            candidate
                .put("gpu_split", gpuResult)
                .put("status", STATUS_COMPLETED)
        } catch (error: Throwable) {
            candidate
                .put("status", STATUS_FAILED)
                .put("error_type", error.javaClass.name)
                .put("error_message", sanitizedErrorMessage(error))
        }
        return candidate
    }

    private fun runDecoderBoundaryCandidate(
        context: Context,
        diagnosticAssets: android.content.res.AssetManager,
        stage: WhisperDecoderBoundaryStage,
        language: WhisperLanguage.Forced,
        referenceTokens: List<Int>,
        referenceTruncated: Boolean,
        cpuDiagnostic: CpuDiagnostic,
        runAutoregressive: Boolean,
        allowSourceQuantizedFcConvOps: Boolean,
        constantTensorSharing: Boolean,
        gpuBackend: WhisperSplitDecoderGpuBackend,
        strictGpuOnly: Boolean,
        speculativeBlockSize: Int,
        reverseDecoderSplit: Boolean,
    ): JSONObject {
        val assets = WhisperDecoderBoundaryAssets.forStage(stage)
        val executionMode = if (reverseDecoderSplit) {
            WhisperDecoderBoundaryExecutionMode.CPU_PREFIX_GPU_TAIL
        } else {
            WhisperDecoderBoundaryExecutionMode.STAGE_ROUTED
        }
        val gpuBackendLabel = "${gpuBackend.name.lowercase()}-fp32"
        val prefixBackendLabel = if (reverseDecoderSplit) {
            "cpu-xnnpack-3"
        } else {
            gpuBackendLabel
        }
        val tailBackendLabel = if (reverseDecoderSplit) {
            gpuBackendLabel
        } else {
            "cpu-xnnpack-3"
        }
        val options = JSONObject()
            .put("split_stage", stage.id)
            .put("reverse_decoder_split", reverseDecoderSplit)
            .put(
                "execution_mode",
                if (reverseDecoderSplit) {
                    "cpu-prefix-gpu-tail"
                } else {
                    "gpu-prefix-cpu-tail"
                },
            )
            .put("prefix_backend", prefixBackendLabel)
            .put("tail_backend", tailBackendLabel)
            .put(
                "cpu_verifier_owner",
                if (reverseDecoderSplit) "prefix" else "tail",
            )
            .put(
                "allow_src_quantized_fc_conv_ops",
                allowSourceQuantizedFcConvOps,
            )
            .put("constant_tensor_sharing", constantTensorSharing)
            .put("gpu_backend", gpuBackend.name.lowercase())
            .put("strict_gpu_only", strictGpuOnly)
            .put("active_row_projection", "cpu-xnnpack-3")
        if (reverseDecoderSplit) {
            options
                .put("cpu_prefix", "xnnpack-3")
                .put("gpu_tail", gpuBackendLabel)
        } else {
            options
                .put("gpu_prefix", gpuBackendLabel)
                .put("cpu_tail", "xnnpack-3")
        }
        val candidate = JSONObject()
            .put("name", DECODER_SPLIT_PROFILE)
            .put("options", options)
        try {
            val teacherForcedStepCount =
                if (reverseDecoderSplit) referenceTokens.size else TEACHER_FORCED_STEP_COUNT
            val cpuBoundaryRows = mutableListOf<FloatArray>()
            val cpuPreLayerNormRows = mutableListOf<FloatArray>()
            val cpuControl = JSONObject()
            val cpuLoadStarted = System.nanoTime()
            LiteRtDecoderBoundaryDiagnostic.openCpuControl(
                configurationAssets = context.assets,
                diagnosticAssets = diagnosticAssets,
                stage = stage,
                assets = assets,
                cpuThreadCount = 3,
            ).use { boundaryCpu ->
                cpuControl
                    .put("model_load_millis", elapsedMillis(cpuLoadStarted))
                    .put("memory_after_open", processMemorySnapshot(context))
                val steps = JSONArray()
                (0 until teacherForcedStepCount).forEach { stepIndex ->
                    val step = boundaryCpu.teacherForcedStep(
                        encoderState = cpuDiagnostic.encoderState,
                        language = language,
                        referenceTokenPrefix = referenceTokens.take(stepIndex),
                    )
                    cpuBoundaryRows += step.activeBoundaryRow
                    cpuPreLayerNormRows += step.activePreLayerNormRow
                    val comparison = compareBoundaryProjectedStep(
                        cpu = cpuDiagnostic.teacherForcedRows[stepIndex],
                        split = step,
                        step = stepIndex,
                        referenceToken = referenceTokens[stepIndex],
                    )
                    steps.put(comparison)
                    require(comparison.getBoolean("exact_logits_match")) {
                        "CPU decoder-boundary control step $stepIndex did not reproduce " +
                            "the trusted CPU logits bit-exactly."
                    }
                }
                cpuControl.put("teacher_forced_steps", steps)
                if (runAutoregressive && speculativeBlockSize == 0) {
                    val fullDecode = runBoundaryAutoregressive(
                        engine = boundaryCpu,
                        encoderState = cpuDiagnostic.encoderState,
                        language = language,
                        referenceTokens = referenceTokens,
                        referenceTruncated = referenceTruncated,
                    )
                    cpuControl.put("autoregressive", fullDecode)
                    require(fullDecode.getBoolean("exact_cpu_result_match")) {
                        "The all-CPU decoder-boundary control did not reproduce the " +
                            "trusted CPU reference."
                    }
                }
                cpuControl.put("memory_after_decode", processMemorySnapshot(context))
            }
            candidate.put("cpu_control", cpuControl)

            val gpuResult = JSONObject()
            val gpuLoadStarted = System.nanoTime()
            LiteRtDecoderBoundaryDiagnostic.open(
                configurationAssets = context.assets,
                diagnosticAssets = diagnosticAssets,
                stage = stage,
                assets = assets,
                executionMode = executionMode,
                cpuThreadCount = 3,
                allowSourceQuantizedFcConvOps = allowSourceQuantizedFcConvOps,
                constantTensorSharing = constantTensorSharing,
                gpuBackend = gpuBackend,
                strictGpuOnly = strictGpuOnly,
            ).use { boundaryGpu ->
                gpuResult
                    .put("model_load_millis", elapsedMillis(gpuLoadStarted))
                    .put("memory_after_open", processMemorySnapshot(context))
                val steps = JSONArray()
                var firstTeacherForcedTop1Mismatch: Int? = null
                for (stepIndex in 0 until teacherForcedStepCount) {
                    val step = boundaryGpu.teacherForcedStep(
                        encoderState = cpuDiagnostic.encoderState,
                        language = language,
                        referenceTokenPrefix = referenceTokens.take(stepIndex),
                    )
                    val boundaryComparison = compareFloatArrays(
                        reference = cpuBoundaryRows[stepIndex],
                        candidate = step.activeBoundaryRow,
                    )
                    val preLayerNormComparison = compareFloatArrays(
                        reference = cpuPreLayerNormRows[stepIndex],
                        candidate = step.activePreLayerNormRow,
                    )
                    val comparison =
                        compareBoundaryProjectedStep(
                            cpu = cpuDiagnostic.teacherForcedRows[stepIndex],
                            split = step,
                            step = stepIndex,
                            referenceToken = referenceTokens[stepIndex],
                        )
                            .put("hidden_cpu_gpu_comparison", boundaryComparison)
                            .put("boundary_cpu_gpu_comparison", boundaryComparison)
                            .put(
                                "boundary_cpu_gpu_exact",
                                rawFloatArraysEqual(
                                    cpuBoundaryRows[stepIndex],
                                    step.activeBoundaryRow,
                                ),
                            )
                            .put(
                                "boundary_cpu_sha256",
                                floatDigest(cpuBoundaryRows[stepIndex]),
                            )
                            .put(
                                "boundary_gpu_sha256",
                                floatDigest(step.activeBoundaryRow),
                            )
                            .put(
                                "pre_layer_norm_cpu_gpu_comparison",
                                preLayerNormComparison,
                            )
                            .put(
                                "pre_layer_norm_cpu_gpu_exact",
                                rawFloatArraysEqual(
                                    cpuPreLayerNormRows[stepIndex],
                                    step.activePreLayerNormRow,
                                ),
                            )
                            .put(
                                "pre_layer_norm_cpu_sha256",
                                floatDigest(cpuPreLayerNormRows[stepIndex]),
                            )
                            .put(
                                "pre_layer_norm_gpu_sha256",
                                floatDigest(step.activePreLayerNormRow),
                            )
                    steps.put(comparison)
                    if (
                        reverseDecoderSplit &&
                        !comparison.getBoolean("selectable_top1_match")
                    ) {
                        firstTeacherForcedTop1Mismatch = stepIndex
                        break
                    }
                }
                gpuResult
                    .put("teacher_forced_requested_step_count", teacherForcedStepCount)
                    .put("teacher_forced_completed_step_count", steps.length())
                    .put(
                        "first_teacher_forced_top1_mismatch",
                        firstTeacherForcedTop1Mismatch ?: JSONObject.NULL,
                    )
                    .put("teacher_forced_steps", steps)
                if (
                    reverseDecoderSplit &&
                    (runAutoregressive || speculativeBlockSize > 0)
                ) {
                    require(firstTeacherForcedTop1Mismatch == null) {
                        "Reverse decoder split failed teacher-forced top-1 parity at " +
                            "step $firstTeacherForcedTop1Mismatch; decode was not run."
                    }
                }
                if (runAutoregressive && speculativeBlockSize == 0) {
                    gpuResult.put(
                        "autoregressive",
                        runBoundaryAutoregressive(
                            engine = boundaryGpu,
                            encoderState = cpuDiagnostic.encoderState,
                            language = language,
                            referenceTokens = referenceTokens,
                            referenceTruncated = referenceTruncated,
                        ),
                    )
                }
                if (speculativeBlockSize > 0) {
                    val speculative = runCpuVerifiedBoundarySpeculativeDecode(
                        context = context,
                        boundaryGpu = boundaryGpu,
                        encoderState = cpuDiagnostic.encoderState,
                        language = language,
                        referenceTokens = referenceTokens,
                        referenceTruncated = referenceTruncated,
                        blockSize = speculativeBlockSize,
                    )
                    candidate.put("cpu_verified_speculative_decode", speculative)
                    require(speculative.getBoolean("exact_cpu_result_match")) {
                        "CPU-verified decoder-boundary speculation did not reproduce " +
                            "the trusted CPU token and termination result."
                    }
                }
                gpuResult.put("memory_after_decode", processMemorySnapshot(context))
            }
            candidate
                .put("gpu_split", gpuResult)
                .put("status", STATUS_COMPLETED)
        } catch (error: Throwable) {
            candidate
                .put("status", STATUS_FAILED)
                .put("error_type", error.javaClass.name)
                .put("error_message", sanitizedErrorMessage(error))
        }
        return candidate
    }

    private fun compareSplitProjectedStep(
        cpu: TeacherForcedRow,
        split: WhisperSplitDecoderStep,
        step: Int,
        referenceToken: Int,
    ): JSONObject {
        val splitTop1 = split.rankedLogits.selectableTop1
        return JSONObject()
            .put("step", step)
            .put("reference_token_id", referenceToken)
            .put(
                "cpu_selectable_top1_token_id",
                cpu.row.selectableTop1?.tokenId ?: JSONObject.NULL,
            )
            .put(
                "split_selectable_top1_token_id",
                splitTop1?.tokenId ?: JSONObject.NULL,
            )
            .put(
                "selectable_top1_match",
                cpu.row.selectableTop1?.tokenId == splitTop1?.tokenId,
            )
            .put("split_matches_reference", splitTop1?.tokenId == referenceToken)
            .put("cpu_logits_sha256", floatDigest(cpu.row.logits))
            .put("split_logits_sha256", floatDigest(split.rankedLogits.logits))
            .put(
                "exact_logits_match",
                rawFloatArraysEqual(cpu.row.logits, split.rankedLogits.logits),
            )
            .put("decoder_core_millis", split.decoderCoreMillis)
            .put("active_row_projection_millis", split.projectionMillis)
            .put(
                "numeric_comparison",
                compareFloatArrays(cpu.row.logits, split.rankedLogits.logits),
            )
    }

    private fun compareBoundaryProjectedStep(
        cpu: TeacherForcedRow,
        split: WhisperDecoderBoundaryStep,
        step: Int,
        referenceToken: Int,
    ): JSONObject {
        val splitTop1 = split.rankedLogits.selectableTop1
        return JSONObject()
            .put("step", step)
            .put("reference_token_id", referenceToken)
            .put(
                "cpu_selectable_top1_token_id",
                cpu.row.selectableTop1?.tokenId ?: JSONObject.NULL,
            )
            .put(
                "split_selectable_top1_token_id",
                splitTop1?.tokenId ?: JSONObject.NULL,
            )
            .put(
                "selectable_top1_match",
                cpu.row.selectableTop1?.tokenId == splitTop1?.tokenId,
            )
            .put("split_matches_reference", splitTop1?.tokenId == referenceToken)
            .put("cpu_logits_sha256", floatDigest(cpu.row.logits))
            .put("split_logits_sha256", floatDigest(split.rankedLogits.logits))
            .put(
                "exact_logits_match",
                rawFloatArraysEqual(cpu.row.logits, split.rankedLogits.logits),
            )
            .put("decoder_prefix_millis", split.prefixMillis)
            .put("decoder_tail_millis", split.tailMillis)
            .put("decoder_core_millis", split.prefixMillis + split.tailMillis)
            .put("active_row_projection_millis", split.projectionMillis)
            .put(
                "numeric_comparison",
                compareFloatArrays(cpu.row.logits, split.rankedLogits.logits),
            )
    }

    private fun runSplitAutoregressive(
        engine: LiteRtSplitDecoderDiagnostic,
        encoderState: FloatArray,
        language: WhisperLanguage.Forced,
        referenceTokens: List<Int>,
        referenceTruncated: Boolean,
    ): JSONObject {
        val generated = mutableListOf<Int>()
        var reachedEos = false
        var coreMillis = 0L
        var projectionMillis = 0L
        var decoderInvocations = 0
        val startedNanos = System.nanoTime()
        while (generated.size < MAX_GENERATED_TOKENS) {
            val step = engine.teacherForcedStep(
                encoderState = encoderState,
                language = language,
                referenceTokenPrefix = generated,
            )
            decoderInvocations++
            coreMillis += step.decoderCoreMillis
            projectionMillis += step.projectionMillis
            val nextToken = step.rankedLogits.selectableTop1?.tokenId
                ?: throw WhisperModelContractException(
                    "Split decoder produced no selectable token.",
                )
            if (nextToken == EOS_TOKEN_ID) {
                reachedEos = true
                break
            }
            generated += nextToken
        }
        val truncated = !reachedEos && generated.size >= MAX_GENERATED_TOKENS
        val mismatch = firstMismatch(referenceTokens, generated)
        val exactTokens = mismatch == null
        val exactTermination = truncated == referenceTruncated
        return JSONObject()
            .put("decoder_invocations", decoderInvocations)
            .put("decoder_core_millis", coreMillis)
            .put("active_row_projection_millis", projectionMillis)
            .put("wall_millis", elapsedMillis(startedNanos))
            .put("token_count", generated.size)
            .put("token_sha256", tokenDigest(generated))
            .put("first_token_mismatch", mismatch ?: JSONObject.NULL)
            .put("exact_token_match", exactTokens)
            .put("reached_eos", reachedEos)
            .put("truncated", truncated)
            .put("exact_termination_match", exactTermination)
            .put("exact_cpu_result_match", exactTokens && exactTermination)
    }

    private fun runBoundaryAutoregressive(
        engine: LiteRtDecoderBoundaryDiagnostic,
        encoderState: FloatArray,
        language: WhisperLanguage.Forced,
        referenceTokens: List<Int>,
        referenceTruncated: Boolean,
    ): JSONObject {
        val generated = mutableListOf<Int>()
        var reachedEos = false
        var prefixMillis = 0L
        var tailMillis = 0L
        var projectionMillis = 0L
        var decoderInvocations = 0
        val startedNanos = System.nanoTime()
        while (generated.size < MAX_GENERATED_TOKENS) {
            val step = engine.teacherForcedStep(
                encoderState = encoderState,
                language = language,
                referenceTokenPrefix = generated,
            )
            decoderInvocations++
            prefixMillis += step.prefixMillis
            tailMillis += step.tailMillis
            projectionMillis += step.projectionMillis
            val nextToken = step.rankedLogits.selectableTop1?.tokenId
                ?: throw WhisperModelContractException(
                    "Decoder-boundary diagnostic produced no selectable token.",
                )
            if (nextToken == EOS_TOKEN_ID) {
                reachedEos = true
                break
            }
            generated += nextToken
        }
        val truncated = !reachedEos && generated.size >= MAX_GENERATED_TOKENS
        val mismatch = firstMismatch(referenceTokens, generated)
        val exactTokens = mismatch == null
        val exactTermination = truncated == referenceTruncated
        return JSONObject()
            .put("decoder_invocations", decoderInvocations)
            .put("decoder_prefix_millis", prefixMillis)
            .put("decoder_tail_millis", tailMillis)
            .put("decoder_core_millis", prefixMillis + tailMillis)
            .put("active_row_projection_millis", projectionMillis)
            .put("wall_millis", elapsedMillis(startedNanos))
            .put("token_count", generated.size)
            .put("token_sha256", tokenDigest(generated))
            .put("first_token_mismatch", mismatch ?: JSONObject.NULL)
            .put("exact_token_match", exactTokens)
            .put("reached_eos", reachedEos)
            .put("truncated", truncated)
            .put("exact_termination_match", exactTermination)
            .put("exact_cpu_result_match", exactTokens && exactTermination)
    }

    /**
     * Uses the stage-routed boundary engine for both GPU proposals and CPU
     * verification. The verifier reuses the engine's CPU tail instead of
     * opening a second copy of the shared decoder model.
     *
     * No token is accepted until [WhisperSpeculativeTokenAcceptance] approves
     * the CPU-selected causal row. Reports contain token IDs and numeric
     * diagnostics only; transcript text is never decoded or persisted.
     */
    private fun runCpuVerifiedBoundarySpeculativeDecode(
        context: Context,
        boundaryGpu: LiteRtDecoderBoundaryDiagnostic,
        encoderState: FloatArray,
        language: WhisperLanguage.Forced,
        referenceTokens: List<Int>,
        referenceTruncated: Boolean,
        blockSize: Int,
    ): JSONObject {
        require(boundaryGpu.stage == WhisperDecoderBoundaryStage.AFTER_1_BLOCK) {
            "Decoder-boundary speculation is validated only after the first block."
        }
        val report = JSONObject()
            .put("block_size", blockSize)
            .put("reference_token_count", referenceTokens.size)
            .put("reference_token_sha256", tokenDigest(referenceTokens))
        val verifiedTokens = mutableListOf<Int>()
        val rounds = JSONArray()
        var reachedEos = false
        var verifierInvocations = 0
        var gpuInvocations = 0
        var totalProposed = 0
        var matchingProposals = 0
        var fullBlockAcceptances = 0
        var cpuCorrections = 0
        var rejectedOrDiscardedProposals = 0
        var gpuMillis = 0L
        var cpuVerifierMillis = 0L
        report.put("combined_memory_before_decode", processMemorySnapshot(context))
        val startedNanos = System.nanoTime()
        while (!reachedEos && verifiedTokens.size < MAX_GENERATED_TOKENS) {
            val remaining = MAX_GENERATED_TOKENS - verifiedTokens.size
            val proposal = boundaryGpu.proposeTokenBlock(
                encoderState = encoderState,
                language = language,
                verifiedTokenPrefix = verifiedTokens,
                maximumTokenCount = minOf(blockSize, remaining),
            )
            require(proposal.tokens.isNotEmpty()) {
                "Decoder-boundary GPU proposer returned an empty block."
            }
            val verification = boundaryGpu.verifyProposedTokenBlock(
                encoderState = encoderState,
                language = language,
                verifiedTokenPrefix = verifiedTokens,
                proposedTokens = proposal.tokens,
            )
            val acceptance = WhisperSpeculativeTokenAcceptance.accept(
                verification = verification,
                eosTokenId = EOS_TOKEN_ID,
            )

            verifierInvocations++
            gpuInvocations += proposal.decoderInvocations
            totalProposed += proposal.tokens.size
            matchingProposals += acceptance.matchingProposalCount
            gpuMillis += proposal.inferenceMillis
            cpuVerifierMillis += verification.inferenceMillis
            if (acceptance.allProposalsAccepted) fullBlockAcceptances++
            if (acceptance.correctedToken != null) cpuCorrections++
            rejectedOrDiscardedProposals +=
                proposal.tokens.size - acceptance.matchingProposalCount
            verifiedTokens += acceptance.acceptedTokens
            reachedEos = acceptance.reachedEos

            val prefixMismatch = firstMismatch(
                reference = referenceTokens.take(verifiedTokens.size),
                candidate = verifiedTokens,
            )
            require(prefixMismatch == null && verifiedTokens.size <= referenceTokens.size) {
                "CPU-verified decoder-boundary prefix diverged from the trusted CPU reference."
            }
            rounds.put(
                JSONObject()
                    .put("round", verifierInvocations - 1)
                    .put("proposed_token_ids", JSONArray(proposal.tokens))
                    .put("proposed_count", proposal.tokens.size)
                    .put(
                        "gpu_selectable_top1_margins",
                        finiteNumberArray(proposal.gpuSelectableTop1Margins),
                    )
                    .put(
                        "cpu_selectable_top1_margins",
                        finiteNumberArray(verification.cpuSelectableTop1Margins),
                    )
                    .put(
                        "proposal_matches_cpu_top1",
                        JSONArray(verification.proposalMatchesCpuTop1),
                    )
                    .put(
                        "matching_proposal_prefix_count",
                        acceptance.matchingProposalCount,
                    )
                    .put("accepted_token_ids", JSONArray(acceptance.acceptedTokens))
                    .put("accepted_count", acceptance.acceptedTokens.size)
                    .put("used_cpu_correction", acceptance.correctedToken != null)
                    .put("reached_eos", reachedEos)
                    .put("verified_token_count", verifiedTokens.size)
                    .put("verified_token_sha256", tokenDigest(verifiedTokens))
                    .put("gpu_inference_millis", proposal.inferenceMillis)
                    .put("cpu_verifier_millis", verification.inferenceMillis),
            )
        }
        val decodeWallMillis = elapsedMillis(startedNanos)
        report.put("combined_memory_after_decode", processMemorySnapshot(context))

        val truncated = !reachedEos && verifiedTokens.size >= MAX_GENERATED_TOKENS
        val mismatch = firstMismatch(referenceTokens, verifiedTokens)
        val exactTokens = mismatch == null
        val exactTermination = truncated == referenceTruncated
        return report
            .put("rounds", rounds)
            .put("verifier_invocations", verifierInvocations)
            .put("gpu_decoder_invocations", gpuInvocations)
            .put("total_proposed_tokens", totalProposed)
            .put("matching_proposal_count", matchingProposals)
            .put(
                "proposal_acceptance_rate",
                matchingProposals.toDouble() / totalProposed.coerceAtLeast(1),
            )
            .put("full_block_acceptance_count", fullBlockAcceptances)
            .put("cpu_correction_count", cpuCorrections)
            .put("rejected_or_discarded_proposals", rejectedOrDiscardedProposals)
            .put(
                "accepted_tokens_per_verifier",
                verifiedTokens.size.toDouble() / verifierInvocations.coerceAtLeast(1),
            )
            .put("gpu_inference_millis", gpuMillis)
            .put("cpu_verifier_millis", cpuVerifierMillis)
            .put("inference_component_millis", gpuMillis + cpuVerifierMillis)
            .put("wall_millis", decodeWallMillis)
            .put("token_ids", JSONArray(verifiedTokens))
            .put("token_count", verifiedTokens.size)
            .put("token_sha256", tokenDigest(verifiedTokens))
            .put("first_token_mismatch", mismatch ?: JSONObject.NULL)
            .put("exact_token_match", exactTokens)
            .put("reached_eos", reachedEos)
            .put("truncated", truncated)
            .put("exact_termination_match", exactTermination)
            .put("exact_cpu_result_match", exactTokens && exactTermination)
    }

    private fun runCpuVerifiedSpeculativeDecode(
        context: Context,
        gpuEngine: LiteRtFullGpuWhisperEngine,
        encoderState: FloatArray,
        language: WhisperLanguage.Forced,
        referenceTokens: List<Int>,
        referenceTruncated: Boolean,
        blockSize: Int,
    ): JSONObject {
        val report = JSONObject()
            .put("block_size", blockSize)
            .put("reference_token_count", referenceTokens.size)
            .put("reference_token_sha256", tokenDigest(referenceTokens))
        val verifiedTokens = mutableListOf<Int>()
        val rounds = JSONArray()
        var reachedEos = false
        var verifierInvocations = 0
        var gpuInvocations = 0
        var totalProposed = 0
        var matchingProposals = 0
        var fullBlockAcceptances = 0
        var cpuCorrections = 0
        var rejectedOrDiscardedProposals = 0
        var gpuMillis = 0L
        var cpuVerifierMillis = 0L
        val cpuVerifierLoadStarted = System.nanoTime()
        val cpuVerifier = LiteRtWhisperEngine.open(context, threadCount = 3)
        report.put(
            "cpu_verifier_model_load_millis",
            elapsedMillis(cpuVerifierLoadStarted),
        )
        val startedNanos = System.nanoTime()
        cpuVerifier.use {
            report.put("combined_memory_before_decode", processMemorySnapshot(context))
            while (!reachedEos && verifiedTokens.size < MAX_GENERATED_TOKENS) {
                val remaining = MAX_GENERATED_TOKENS - verifiedTokens.size
                val proposal = gpuEngine.proposeTokenBlock(
                    encoderState = encoderState,
                    language = language,
                    verifiedTokenPrefix = verifiedTokens,
                    maximumTokenCount = minOf(blockSize, remaining),
                )
                require(proposal.tokens.isNotEmpty()) {
                    "GPU speculative decoder returned an empty block."
                }
                val verification = it.verifyProposedTokenBlock(
                    encoderState = encoderState,
                    language = language,
                    verifiedTokenPrefix = verifiedTokens,
                    proposedTokens = proposal.tokens,
                )
                val acceptance = WhisperSpeculativeTokenAcceptance.accept(
                    verification = verification,
                    eosTokenId = EOS_TOKEN_ID,
                )

                verifierInvocations++
                gpuInvocations += proposal.decoderInvocations
                totalProposed += proposal.tokens.size
                matchingProposals += acceptance.matchingProposalCount
                gpuMillis += proposal.inferenceMillis
                cpuVerifierMillis += verification.inferenceMillis
                if (acceptance.allProposalsAccepted) fullBlockAcceptances++
                if (acceptance.correctedToken != null) cpuCorrections++
                rejectedOrDiscardedProposals +=
                    proposal.tokens.size - acceptance.matchingProposalCount
                verifiedTokens += acceptance.acceptedTokens
                reachedEos = acceptance.reachedEos

                val prefixMismatch = firstMismatch(
                    reference = referenceTokens.take(verifiedTokens.size),
                    candidate = verifiedTokens,
                )
                require(prefixMismatch == null && verifiedTokens.size <= referenceTokens.size) {
                    "CPU-verified speculative prefix diverged from the trusted CPU reference."
                }
                rounds.put(
                    JSONObject()
                        .put("round", verifierInvocations - 1)
                        .put("proposed_count", proposal.tokens.size)
                        .put(
                            "matching_proposal_prefix_count",
                            acceptance.matchingProposalCount,
                        )
                        .put("used_cpu_correction", acceptance.correctedToken != null)
                        .put("reached_eos", reachedEos)
                        .put("verified_token_count", verifiedTokens.size)
                        .put("gpu_inference_millis", proposal.inferenceMillis)
                        .put("cpu_verifier_millis", verification.inferenceMillis),
                )
            }
            report.put("combined_memory_after_decode", processMemorySnapshot(context))
        }

        val truncated = !reachedEos && verifiedTokens.size >= MAX_GENERATED_TOKENS
        val mismatch = firstMismatch(referenceTokens, verifiedTokens)
        val exactTokens = mismatch == null
        val exactTermination = truncated == referenceTruncated
        return report
            .put("rounds", rounds)
            .put("verifier_invocations", verifierInvocations)
            .put("gpu_decoder_invocations", gpuInvocations)
            .put("total_proposed_tokens", totalProposed)
            .put("matching_proposal_count", matchingProposals)
            .put(
                "proposal_acceptance_rate",
                matchingProposals.toDouble() / totalProposed.coerceAtLeast(1),
            )
            .put("full_block_acceptance_count", fullBlockAcceptances)
            .put("cpu_correction_count", cpuCorrections)
            .put("rejected_or_discarded_proposals", rejectedOrDiscardedProposals)
            .put(
                "accepted_tokens_per_verifier",
                verifiedTokens.size.toDouble() / verifierInvocations.coerceAtLeast(1),
            )
            .put("gpu_inference_millis", gpuMillis)
            .put("cpu_verifier_millis", cpuVerifierMillis)
            .put("wall_millis", elapsedMillis(startedNanos))
            .put("token_count", verifiedTokens.size)
            .put("token_sha256", tokenDigest(verifiedTokens))
            .put("first_token_mismatch", mismatch ?: JSONObject.NULL)
            .put("exact_token_match", exactTokens)
            .put("reached_eos", reachedEos)
            .put("truncated", truncated)
            .put("exact_termination_match", exactTermination)
            .put("exact_cpu_result_match", exactTokens && exactTermination)
    }

    private fun processMemorySnapshot(context: Context): JSONObject {
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

    private fun compareFloatArrays(reference: FloatArray, candidate: FloatArray): JSONObject {
        require(reference.size == candidate.size)
        var exactElements = 0
        var finitePairs = 0
        var maxAbsoluteError = 0.0
        var absoluteErrorSum = 0.0
        var squaredErrorSum = 0.0
        var errorMean = 0.0
        var errorSquaredDeviationSum = 0.0
        var referenceMean = 0.0
        var candidateMean = 0.0
        var referenceSquaredDeviationSum = 0.0
        var candidateSquaredDeviationSum = 0.0
        var centeredCrossProductSum = 0.0
        var referenceSquaredSum = 0.0
        var candidateSquaredSum = 0.0
        var dotProduct = 0.0
        reference.indices.forEach { index ->
            val expected = reference[index]
            val actual = candidate[index]
            if (expected.toRawBits() == actual.toRawBits()) exactElements++
            if (!expected.isFinite() || !actual.isFinite()) return@forEach
            finitePairs++
            val difference = actual.toDouble() - expected.toDouble()
            val count = finitePairs.toDouble()
            val errorDelta = difference - errorMean
            errorMean += errorDelta / count
            errorSquaredDeviationSum += errorDelta * (difference - errorMean)
            val expectedDouble = expected.toDouble()
            val actualDouble = actual.toDouble()
            val referenceDelta = expectedDouble - referenceMean
            referenceMean += referenceDelta / count
            val candidateDelta = actualDouble - candidateMean
            candidateMean += candidateDelta / count
            referenceSquaredDeviationSum +=
                referenceDelta * (expectedDouble - referenceMean)
            candidateSquaredDeviationSum +=
                candidateDelta * (actualDouble - candidateMean)
            centeredCrossProductSum +=
                referenceDelta * (actualDouble - candidateMean)
            val absoluteError = abs(difference)
            maxAbsoluteError = maxOf(maxAbsoluteError, absoluteError)
            absoluteErrorSum += absoluteError
            squaredErrorSum += difference * difference
            referenceSquaredSum += expectedDouble * expectedDouble
            candidateSquaredSum += actualDouble * actualDouble
            dotProduct += expectedDouble * actualDouble
        }
        val divisor = finitePairs.coerceAtLeast(1).toDouble()
        val cosineDenominator = sqrt(referenceSquaredSum * candidateSquaredSum)
        val pearsonVarianceProduct =
            referenceSquaredDeviationSum * candidateSquaredDeviationSum
        val pearsonDenominator = sqrt(pearsonVarianceProduct.coerceAtLeast(0.0))
        val centeredRootMeanSquareError =
            sqrt((errorSquaredDeviationSum / divisor).coerceAtLeast(0.0))
        return JSONObject()
            .put("element_count", reference.size)
            .put("finite_pair_count", finitePairs)
            .put("exact_element_count", exactElements)
            .put("max_absolute_error", maxAbsoluteError)
            .put("mean_absolute_error", absoluteErrorSum / divisor)
            .put("root_mean_square_error", sqrt(squaredErrorSum / divisor))
            .put(
                "mean_error",
                if (finitePairs == 0) JSONObject.NULL else errorMean,
            )
            .put(
                "centered_root_mean_square_error",
                if (finitePairs == 0) JSONObject.NULL else centeredRootMeanSquareError,
            )
            .put(
                "error_standard_deviation",
                if (finitePairs == 0) JSONObject.NULL else centeredRootMeanSquareError,
            )
            .put(
                "cosine_similarity",
                if (cosineDenominator == 0.0) JSONObject.NULL else dotProduct / cosineDenominator,
            )
            .put(
                "pearson_correlation",
                if (
                    finitePairs == 0 ||
                    pearsonDenominator == 0.0 ||
                    !pearsonDenominator.isFinite()
                ) {
                    JSONObject.NULL
                } else {
                    centeredCrossProductSum / pearsonDenominator
                },
            )
    }

    private fun compareLogitRows(
        cpu: TeacherForcedRow,
        gpu: WhisperTeacherForcedLogitsRow,
        step: Int,
        referenceToken: Int,
        gpuInferenceMillis: Long,
    ): JSONObject {
        require(cpu.row.logits.size == gpu.logits.size)
        require(referenceToken in cpu.row.logits.indices)
        val cpuRawTop = topFiniteLogit(cpu.row.logits)
        val gpuRawTop = topFiniteLogit(gpu.logits)
        val cpuSelectableTop1 = cpu.row.selectableTop1
        val gpuSelectableTop1 = gpu.selectableTop1
        return JSONObject()
            .put("step", step)
            .put("prompt_position", STANDARD_PROMPT_LENGTH - 1 + step)
            .put("reference_token_id", referenceToken)
            .put("cpu_inference_millis", cpu.inferenceMillis)
            .put("gpu_inference_millis", gpuInferenceMillis)
            .put("cpu_logits_sha256", floatDigest(cpu.row.logits))
            .put("gpu_logits_sha256", floatDigest(gpu.logits))
            .put(
                "exact_logits_match",
                rawFloatArraysEqual(cpu.row.logits, gpu.logits),
            )
            .put(
                "cpu_selectable_top1_token_id",
                cpu.row.selectableTop1?.tokenId ?: JSONObject.NULL,
            )
            .put(
                "cpu_selectable_top2_token_id",
                cpu.row.selectableTop2?.tokenId ?: JSONObject.NULL,
            )
            .put(
                "gpu_selectable_top1_token_id",
                gpu.selectableTop1?.tokenId ?: JSONObject.NULL,
            )
            .put(
                "gpu_selectable_top2_token_id",
                gpu.selectableTop2?.tokenId ?: JSONObject.NULL,
            )
            .put(
                "selectable_top1_match",
                cpuSelectableTop1 != null &&
                    gpuSelectableTop1 != null &&
                    cpuSelectableTop1.tokenId == gpuSelectableTop1.tokenId,
            )
            .put(
                "cpu_selectable_top1_matches_reference",
                cpuSelectableTop1?.tokenId == referenceToken,
            )
            .put(
                "gpu_selectable_top1_matches_reference",
                gpuSelectableTop1?.tokenId == referenceToken,
            )
            .put("cpu_selectable_top1_logit", finiteNumber(cpu.row.selectableTop1?.score))
            .put("cpu_selectable_top2_logit", finiteNumber(cpu.row.selectableTop2?.score))
            .put("gpu_selectable_top1_logit", finiteNumber(gpu.selectableTop1?.score))
            .put("gpu_selectable_top2_logit", finiteNumber(gpu.selectableTop2?.score))
            .put("cpu_selectable_margin", finiteNumber(cpu.row.selectableMargin))
            .put("gpu_selectable_margin", finiteNumber(gpu.selectableMargin))
            .put("cpu_raw_top_token_id", cpuRawTop?.tokenId ?: JSONObject.NULL)
            .put("gpu_raw_top_token_id", gpuRawTop?.tokenId ?: JSONObject.NULL)
            .put("cpu_raw_top_logit", finiteNumber(cpuRawTop?.score))
            .put("gpu_raw_top_logit", finiteNumber(gpuRawTop?.score))
            .put(
                "cpu_reference_token_logit",
                finiteNumber(cpu.row.logits[referenceToken]),
            )
            .put("gpu_reference_token_logit", finiteNumber(gpu.logits[referenceToken]))
            .put(
                "numeric_comparison",
                compareFloatArrays(cpu.row.logits, gpu.logits),
            )
    }

    private fun summarizeTeacherForcedSteps(steps: JSONArray): JSONObject {
        var exactLogitsCount = 0
        var selectableTop1MatchCount = 0
        var gpuTop1MatchesReferenceCount = 0
        var firstSelectableTop1Mismatch: Int? = null
        var firstGpuTop1ReferenceMismatch: Int? = null
        var rootMeanSquareErrorSum = 0.0
        var maximumRootMeanSquareError = 0.0
        var gpuInferenceMillis = 0L

        for (index in 0 until steps.length()) {
            val step = steps.getJSONObject(index)
            if (step.getBoolean("exact_logits_match")) exactLogitsCount++
            if (step.getBoolean("selectable_top1_match")) {
                selectableTop1MatchCount++
            } else if (firstSelectableTop1Mismatch == null) {
                firstSelectableTop1Mismatch = step.getInt("step")
            }
            if (step.getBoolean("gpu_selectable_top1_matches_reference")) {
                gpuTop1MatchesReferenceCount++
            } else if (firstGpuTop1ReferenceMismatch == null) {
                firstGpuTop1ReferenceMismatch = step.getInt("step")
            }
            val rootMeanSquareError = step
                .getJSONObject("numeric_comparison")
                .getDouble("root_mean_square_error")
            rootMeanSquareErrorSum += rootMeanSquareError
            maximumRootMeanSquareError =
                maxOf(maximumRootMeanSquareError, rootMeanSquareError)
            gpuInferenceMillis += step.getLong("gpu_inference_millis")
        }

        val stepCount = steps.length()
        return JSONObject()
            .put("step_count", stepCount)
            .put("exact_logits_count", exactLogitsCount)
            .put("selectable_top1_match_count", selectableTop1MatchCount)
            .put(
                "selectable_top1_match_rate",
                selectableTop1MatchCount.toDouble() / stepCount.coerceAtLeast(1),
            )
            .put(
                "first_selectable_top1_mismatch",
                firstSelectableTop1Mismatch ?: JSONObject.NULL,
            )
            .put(
                "gpu_top1_matches_reference_count",
                gpuTop1MatchesReferenceCount,
            )
            .put(
                "gpu_top1_matches_reference_rate",
                gpuTop1MatchesReferenceCount.toDouble() / stepCount.coerceAtLeast(1),
            )
            .put(
                "first_gpu_top1_reference_mismatch",
                firstGpuTop1ReferenceMismatch ?: JSONObject.NULL,
            )
            .put(
                "mean_root_mean_square_error",
                rootMeanSquareErrorSum / stepCount.coerceAtLeast(1),
            )
            .put("maximum_root_mean_square_error", maximumRootMeanSquareError)
            .put("gpu_inference_millis", gpuInferenceMillis)
    }

    private fun compareTokens(
        referenceTokens: List<Int>,
        candidate: WhisperChunkResult,
    ): JSONObject {
        val mismatch = firstMismatch(referenceTokens, candidate.tokens)
        return JSONObject()
            .put("token_count", candidate.tokens.size)
            .put("token_sha256", tokenDigest(candidate.tokens))
            .put("first_token_mismatch", mismatch ?: JSONObject.NULL)
            .put("exact_token_match", mismatch == null)
            .put("detected_language", candidate.detectedLanguage)
            .put("truncated", candidate.truncated)
            .put("inference_millis", candidate.inferenceMillis)
    }

    private fun firstMismatch(reference: List<Int>, candidate: List<Int>): Int? {
        reference.indices.take(minOf(reference.size, candidate.size)).forEach { index ->
            if (reference[index] != candidate[index]) return index
        }
        return if (reference.size == candidate.size) null else minOf(reference.size, candidate.size)
    }

    private fun tokenDigest(tokens: List<Int>): String {
        val bytes = tokens.joinToString(separator = ",").toByteArray(Charsets.UTF_8)
        return MessageDigest.getInstance("SHA-256")
            .digest(bytes)
            .joinToString(separator = "") { byte -> "%02x".format(byte) }
    }

    private fun floatDigest(values: FloatArray): String {
        val digest = MessageDigest.getInstance("SHA-256")
        values.forEach { value ->
            val bits = value.toRawBits()
            digest.update((bits ushr 24).toByte())
            digest.update((bits ushr 16).toByte())
            digest.update((bits ushr 8).toByte())
            digest.update(bits.toByte())
        }
        return digest.digest().joinToString(separator = "") { byte -> "%02x".format(byte) }
    }

    private fun rawFloatArraysEqual(reference: FloatArray, candidate: FloatArray): Boolean =
        reference.size == candidate.size &&
            reference.indices.all { index ->
                reference[index].toRawBits() == candidate[index].toRawBits()
            }

    private fun topFiniteLogit(logits: FloatArray): TopLogit? {
        var selectedToken = -1
        var selectedScore = Float.NEGATIVE_INFINITY
        logits.forEachIndexed { token, score ->
            if (score.isFinite() && score > selectedScore) {
                selectedToken = token
                selectedScore = score
            }
        }
        return if (selectedToken < 0) null else TopLogit(selectedToken, selectedScore)
    }

    private fun finiteNumber(value: Float?): Any =
        if (value != null && value.isFinite()) value.toDouble() else JSONObject.NULL

    private fun finiteNumberArray(values: List<Float?>): JSONArray =
        JSONArray().also { result ->
            values.forEach { value -> result.put(finiteNumber(value)) }
        }

    private fun elapsedMillis(startedNanos: Long): Long =
        (System.nanoTime() - startedNanos) / 1_000_000L

    private fun requireArgument(value: String?, name: String): String =
        requireNotNull(value?.takeIf(String::isNotBlank)) {
            "Missing instrumentation argument $name."
        }

    private fun requireDiagnosticRunId(value: String?): String {
        val runId = requireArgument(value, ARG_DIAGNOSTIC_RUN_ID)
        require(
            runId.length <= MAX_RUN_ID_CHARACTERS &&
                runId.all { character ->
                    character in 'A'..'Z' ||
                        character in 'a'..'z' ||
                        character in '0'..'9' ||
                        character == '-'
                },
        ) {
            "$ARG_DIAGNOSTIC_RUN_ID must contain only ASCII letters, digits, and hyphens."
        }
        return runId
    }

    private fun resolveTeacherForcedStepCount(
        requested: String?,
        defaultCount: Int,
        referenceTokenCount: Int,
    ): Int {
        require(referenceTokenCount > 0) {
            "Reference chunk must contain at least one token."
        }
        if (requested == null) {
            require(defaultCount in 1..referenceTokenCount) {
                "Reference chunk needs at least $defaultCount tokens for the " +
                    "teacher-forced parity diagnostic."
            }
            return defaultCount
        }
        if (requested.equals(ALL_TEACHER_FORCED_STEPS, ignoreCase = true)) {
            return referenceTokenCount
        }
        val requestedCount = requested.toIntOrNull()
        require(requestedCount != null && requestedCount in 1..referenceTokenCount) {
            "$ARG_TEACHER_FORCED_STEPS must be 'all' or between 1 and " +
                "$referenceTokenCount."
        }
        return requestedCount
    }

    private fun parseGpuProfiles(arguments: Bundle): List<GpuProfile> {
        val hintWaitingForCompletion =
            arguments.getString(ARG_HINT_WAITING)?.toBooleanStrictOrNull() ?: false
        val kernelBatchSize = arguments.getString(ARG_KERNEL_BATCH_SIZE)?.toIntOrNull()
        val requestedProfiles = arguments.getString(ARG_GPU_PROFILES)
            ?.takeIf(String::isNotBlank)
        if (requestedProfiles == null) {
            return listOf(
                GpuProfile(
                    name = CUSTOM_PROFILE,
                    options = GpuProfileOptions(
                        hintWaitingForCompletion = hintWaitingForCompletion,
                        kernelBatchSize = kernelBatchSize,
                        allowSourceQuantizedFcConvOps =
                            arguments.getString(ARG_SOURCE_QUANTIZATION)
                                ?.toBooleanStrictOrNull() ?: true,
                        backend = parseGpuBackend(arguments.getString(ARG_GPU_BACKEND)),
                        bufferStorageType = parseBufferStorage(
                            arguments.getString(ARG_BUFFER_STORAGE),
                        ),
                        preferTextureWeights =
                            arguments.getString(ARG_PREFER_TEXTURE_WEIGHTS)
                                ?.toBooleanStrictOrNull(),
                        benchmarkMode =
                            arguments.getString(ARG_BENCHMARK_MODE)
                                ?.toBooleanStrictOrNull() ?: false,
                    ),
                ),
            )
        }

        val baseOptions = GpuProfileOptions(
            hintWaitingForCompletion = hintWaitingForCompletion,
            kernelBatchSize = kernelBatchSize,
            allowSourceQuantizedFcConvOps = true,
            backend = CompiledModel.GpuOptions.Backend.OPENCL,
            bufferStorageType = CompiledModel.GpuOptions.BufferStorageType.DEFAULT,
            preferTextureWeights = null,
            benchmarkMode = false,
        )
        return requestedProfiles.split(',').map { rawName ->
            val name = rawName.trim().lowercase()
            require(name.isNotEmpty()) {
                "GPU profile names in $ARG_GPU_PROFILES must not be empty."
            }
            GpuProfile(
                name = name,
                options = when (name) {
                    PROFILE_CURRENT -> baseOptions
                    PROFILE_OPENCL_BUFFER -> baseOptions.copy(
                        bufferStorageType =
                            CompiledModel.GpuOptions.BufferStorageType.BUFFER,
                    )
                    PROFILE_OPENCL_TEXTURE_2D -> baseOptions.copy(
                        bufferStorageType =
                            CompiledModel.GpuOptions.BufferStorageType.TEXTURE_2D,
                    )
                    PROFILE_OPENCL_TEXTURE_WEIGHTS -> baseOptions.copy(
                        preferTextureWeights = true,
                    )
                    PROFILE_OPENCL_BUFFER_WEIGHTS -> baseOptions.copy(
                        preferTextureWeights = false,
                    )
                    PROFILE_AUTOMATIC -> baseOptions.copy(
                        backend = CompiledModel.GpuOptions.Backend.AUTOMATIC,
                    )
                    PROFILE_OPENGL -> baseOptions.copy(
                        backend = CompiledModel.GpuOptions.Backend.OPENGL,
                    )
                    PROFILE_BENCHMARK -> baseOptions.copy(benchmarkMode = true)
                    PROFILE_NO_SOURCE_QUANT -> baseOptions.copy(
                        allowSourceQuantizedFcConvOps = false,
                    )
                    else -> error(
                        "Unsupported GPU profile '$rawName'. Supported profiles: " +
                            PREDEFINED_PROFILES.joinToString(),
                    )
                },
            )
        }
    }

    private fun parseGpuBackend(value: String?): CompiledModel.GpuOptions.Backend =
        when (value?.lowercase() ?: "opencl") {
            "automatic" -> CompiledModel.GpuOptions.Backend.AUTOMATIC
            "opencl" -> CompiledModel.GpuOptions.Backend.OPENCL
            "webgpu" -> CompiledModel.GpuOptions.Backend.WEBGPU
            "opengl" -> CompiledModel.GpuOptions.Backend.OPENGL
            else -> error("Unsupported GPU backend '$value'.")
        }

    private fun parseSplitGpuBackend(value: String?): WhisperSplitDecoderGpuBackend =
        when (value?.lowercase() ?: "opencl") {
            "automatic" -> WhisperSplitDecoderGpuBackend.AUTOMATIC
            "opencl" -> WhisperSplitDecoderGpuBackend.OPENCL
            "webgpu" -> WhisperSplitDecoderGpuBackend.WEBGPU
            "opengl" -> WhisperSplitDecoderGpuBackend.OPENGL
            else -> error("Unsupported split GPU backend '$value'.")
        }

    private fun parseBufferStorage(
        value: String?,
    ): CompiledModel.GpuOptions.BufferStorageType =
        when (value?.lowercase() ?: "default") {
            "default" -> CompiledModel.GpuOptions.BufferStorageType.DEFAULT
            "buffer" -> CompiledModel.GpuOptions.BufferStorageType.BUFFER
            "texture2d" -> CompiledModel.GpuOptions.BufferStorageType.TEXTURE_2D
            else -> error("Unsupported GPU buffer storage '$value'.")
        }

    private fun sanitizedErrorMessage(error: Throwable): String? =
        error.message
            ?.lineSequence()
            ?.firstOrNull()
            ?.take(MAX_ERROR_CHARACTERS)

    private data class GpuProfile(
        val name: String,
        val options: GpuProfileOptions,
    )

    private data class GpuProfileOptions(
        val hintWaitingForCompletion: Boolean,
        val kernelBatchSize: Int?,
        val allowSourceQuantizedFcConvOps: Boolean,
        val backend: CompiledModel.GpuOptions.Backend,
        val bufferStorageType: CompiledModel.GpuOptions.BufferStorageType,
        val preferTextureWeights: Boolean?,
        val benchmarkMode: Boolean,
    ) {
        fun toJson(): JSONObject = JSONObject()
            .put("hint_waiting_for_completion", hintWaitingForCompletion)
            .put("kernel_batch_size", kernelBatchSize ?: JSONObject.NULL)
            .put(
                "allow_src_quantized_fc_conv_ops",
                allowSourceQuantizedFcConvOps,
            )
            .put("backend", backend.name.lowercase())
            .put("buffer_storage_type", bufferStorageType.name.lowercase())
            .put(
                "prefer_texture_weights",
                preferTextureWeights ?: JSONObject.NULL,
            )
            .put("benchmark_mode", benchmarkMode)
    }

    private data class CpuDiagnostic(
        val encoderState: FloatArray,
        val teacherForcedRows: List<TeacherForcedRow>,
        val referenceTokens: List<Int>,
        val referenceTruncated: Boolean,
        val referenceDecode: JSONObject?,
    )

    private data class TeacherForcedRow(
        val row: WhisperTeacherForcedLogitsRow,
        val inferenceMillis: Long,
    )

    private data class TopLogit(
        val tokenId: Int,
        val score: Float,
    )

    companion object {
        private const val ARG_MEMO_ID = "memoId"
        private const val ARG_REFERENCE_JOB = "referenceJob"
        private const val ARG_DIAGNOSTIC_RUN_ID = "diagnosticRunId"
        private const val ARG_CHUNK_INDEX = "chunkIndex"
        private const val ARG_GPU_PROFILES = "gpuProfiles"
        private const val ARG_HINT_WAITING = "hintWaitingForCompletion"
        private const val ARG_KERNEL_BATCH_SIZE = "kernelBatchSize"
        private const val ARG_SOURCE_QUANTIZATION = "allowSourceQuantization"
        private const val ARG_CONSTANT_TENSOR_SHARING = "constantTensorSharing"
        private const val ARG_STRICT_GPU_ONLY = "strictGpuOnly"
        private const val ARG_GPU_BACKEND = "gpuBackend"
        private const val ARG_BUFFER_STORAGE = "bufferStorage"
        private const val ARG_PREFER_TEXTURE_WEIGHTS = "preferTextureWeights"
        private const val ARG_BENCHMARK_MODE = "benchmarkMode"
        private const val ARG_RUN_AUTOREGRESSIVE = "runAutoregressive"
        private const val ARG_TEACHER_FORCED_STEPS = "teacherForcedSteps"
        private const val ARG_LIVE_CPU_REFERENCE = "liveCpuReference"
        private const val ARG_SPECULATIVE_BLOCK_SIZE = "speculativeBlockSize"
        private const val ARG_RUN_DECODER_SPLIT = "runDecoderSplit"
        private const val ARG_REVERSE_DECODER_SPLIT = "reverseDecoderSplit"
        private const val ARG_RUN_GPU_ENCODER_CPU_DECODER = "runGpuEncoderCpuDecoder"
        private const val ARG_DECODER_SPLIT_STAGE = "decoderSplitStage"
        private const val CUSTOM_PROFILE = "custom"
        private const val PROFILE_CURRENT = "current"
        private const val PROFILE_OPENCL_BUFFER = "opencl-buffer"
        private const val PROFILE_OPENCL_TEXTURE_2D = "opencl-texture2d"
        private const val PROFILE_OPENCL_TEXTURE_WEIGHTS = "opencl-texture-weights"
        private const val PROFILE_OPENCL_BUFFER_WEIGHTS = "opencl-buffer-weights"
        private const val PROFILE_AUTOMATIC = "automatic"
        private const val PROFILE_OPENGL = "opengl"
        private const val PROFILE_BENCHMARK = "benchmark"
        private const val PROFILE_NO_SOURCE_QUANT = "no-source-quant"
        private val PREDEFINED_PROFILES = listOf(
            PROFILE_CURRENT,
            PROFILE_OPENCL_BUFFER,
            PROFILE_OPENCL_TEXTURE_2D,
            PROFILE_OPENCL_TEXTURE_WEIGHTS,
            PROFILE_OPENCL_BUFFER_WEIGHTS,
            PROFILE_AUTOMATIC,
            PROFILE_OPENGL,
            PROFILE_BENCHMARK,
            PROFILE_NO_SOURCE_QUANT,
        )
        private const val REPORT_DIRECTORY = "gpu-parity"
        private const val MAX_ERROR_CHARACTERS = 300
        private const val MAX_RUN_ID_CHARACTERS = 80
        private const val STANDARD_PROMPT_LENGTH = 4
        private const val TEACHER_FORCED_STEP_COUNT = 3
        private const val ALL_TEACHER_FORCED_STEPS = "all"
        private const val MAX_SPECULATIVE_BLOCK_SIZE = 16
        private const val MAX_BOUNDARY_SPECULATIVE_BLOCK_SIZE = 32
        private const val MAX_GENERATED_TOKENS = 124
        private const val EOS_TOKEN_ID = 50_257
        private const val DECODER_SPLIT_PROFILE = "decoder-split"
        private const val SPLIT_STAGE_POST_LAYER_NORM = "post-layernorm"
        private const val SPLIT_STAGE_PRE_LAYER_NORM = "pre-layernorm"
        private const val SPLIT_STAGE_AFTER_POSITION_ADD = "after-position-add"
        private const val SPLIT_STAGE_AFTER_1_BLOCK = "after-1-block"
        private const val SPLIT_STAGE_AFTER_2_BLOCKS = "after-2-blocks"
        private const val SPLIT_STAGE_AFTER_3_BLOCKS = "after-3-blocks"
        private const val SPLIT_STAGE_AFTER_CROSS_ATTENTION = "after-cross-attention"
        private val SUPPORTED_SPLIT_STAGES = setOf(
            SPLIT_STAGE_POST_LAYER_NORM,
            SPLIT_STAGE_PRE_LAYER_NORM,
            SPLIT_STAGE_AFTER_POSITION_ADD,
            SPLIT_STAGE_AFTER_1_BLOCK,
            SPLIT_STAGE_AFTER_2_BLOCKS,
            SPLIT_STAGE_AFTER_3_BLOCKS,
            SPLIT_STAGE_AFTER_CROSS_ATTENTION,
        )
        private const val STATUS_COMPLETED = "completed"
        private const val STATUS_FAILED = "failed"
    }
}
