package com.example.voicememo.transcription

import android.content.Context
import android.content.res.AssetManager
import android.os.Debug
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.voicememo.VoiceMemoApplication
import com.example.voicememo.data.TranscriptionJobStatus
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.sqrt
import kotlinx.coroutines.runBlocking
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.Interpreter

/**
 * Compares CPU and GPU activations at one prefix boundary of the rewritten
 * production INT8 decoder. The generated prefix model changes only the
 * decoder operator-vector length and its public output tensor.
 *
 * This test never decodes text and never creates or updates application data.
 */
@RunWith(AndroidJUnit4::class)
class RewrittenDecoderPrefixDiagnosticTest {
    @Test
    fun compareRewrittenDecoderPrefix() = runBlocking {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val arguments = InstrumentationRegistry.getArguments()
        val targetContext = instrumentation.targetContext
        val application =
            targetContext.applicationContext as VoiceMemoApplication
        val repository = application.container.memoRepository
        val memoId = requireSafeValue(arguments.getString(ARG_MEMO_ID), ARG_MEMO_ID)
        val referenceJobId =
            requireSafeValue(arguments.getString(ARG_REFERENCE_JOB), ARG_REFERENCE_JOB)
        val runId = requireSafeValue(arguments.getString(ARG_RUN_ID), ARG_RUN_ID)
        val stage = requireSafeValue(arguments.getString(ARG_STAGE), ARG_STAGE)
        val modelVariant =
            requireSafeValue(arguments.getString(ARG_MODEL_VARIANT), ARG_MODEL_VARIANT)
        val boundarySpec = boundarySpec(stage)
        val requestedSteps = parseSteps(arguments.getString(ARG_STEPS))
        val allowSourceQuantization =
            arguments.getString(ARG_ALLOW_SOURCE_QUANTIZATION)
                ?.toBooleanStrictOrNull()
                ?: true
        val isolationId = arguments.getString(ARG_ISOLATION_ID)?.let { value ->
            requireSafeValue(value, ARG_ISOLATION_ID)
        }
        val captureIsolatedKey =
            arguments.getString(ARG_CAPTURE_ISOLATED_KEY)
                ?.toBooleanStrictOrNull()
                ?: false
        val runIsolatedScore =
            arguments.getString(ARG_RUN_ISOLATED_SCORE)
                ?.toBooleanStrictOrNull()
                ?: false
        val captureIsolatedValue =
            arguments.getString(ARG_CAPTURE_ISOLATED_VALUE)
                ?.toBooleanStrictOrNull()
                ?: false
        val runIsolatedValue =
            arguments.getString(ARG_RUN_ISOLATED_VALUE)
                ?.toBooleanStrictOrNull()
                ?: false
        val reportDirectory =
            File(targetContext.cacheDir, REPORT_DIRECTORY).apply { mkdirs() }
        val reportFile = File(reportDirectory, "prefix-$runId.json")
        val temporaryReport = File(reportDirectory, "${reportFile.name}.tmp")
        reportFile.delete()
        temporaryReport.delete()

        val report = JSONObject()
            .put("run_id", runId)
            .put("memo_id", memoId)
            .put("reference_job_id", referenceJobId)
            .put("stage", stage)
            .put("model_variant", modelVariant)
            .put("boundary_shape", boundarySpec.shape)
            .put("boundary_element_count", boundarySpec.elementCount)
            .put("allow_source_quantization", allowSourceQuantization)
            .put("capture_isolated_key", captureIsolatedKey)
            .put("run_isolated_score", runIsolatedScore)
            .put("capture_isolated_value", captureIsolatedValue)
            .put("run_isolated_value", runIsolatedValue)
            .put("persisted_output", false)
            .put("model_sha256", WhisperAssets.MODEL_SHA256)
            .put("requested_steps", JSONArray(requestedSteps))

        try {
            val memo = requireNotNull(repository.getMemo(memoId)) {
                "Unknown memo $memoId."
            }
            val referenceJob =
                requireNotNull(repository.getTranscriptionJob(referenceJobId)) {
                    "Unknown reference job $referenceJobId."
                }
            require(referenceJob.memoId == memoId) {
                "Reference job belongs to a different memo."
            }
            require(referenceJob.status == TranscriptionJobStatus.SUCCEEDED) {
                "Reference job must be complete."
            }
            val referenceChunk = repository.completedTranscriptChunks(referenceJobId)
                .singleOrNull { it.chunkIndex == 0 }
                ?: error("Reference chunk 0 is not complete.")
            require(requestedSteps.all { it < referenceChunk.tokenIds.size }) {
                "Requested prefix step exceeds the reference token count."
            }
            val languageCode = referenceChunk.detectedLanguage
                ?: referenceJob.detectedLanguage
                ?: error("Reference job has no detected language.")
            val language = WhisperLanguage.Forced(languageCode)
            val generationConfig =
                targetContext.assets.open(WhisperAssets.GENERATION_CONFIG).use {
                    WhisperGenerationConfig.load(it)
                }
            val audio = StreamingWavChunkSource.open(File(memo.audioPath)).use { source ->
                source.readChunk(0).samples
            }

            val encoderLoadStarted = System.nanoTime()
            val encoderState = LiteRtWhisperEngine.open(
                targetContext,
                threadCount = 3,
            ).use { engine ->
                report.put(
                    "cpu_encoder_model_load_millis",
                    elapsedMillis(encoderLoadStarted),
                )
                val started = System.nanoTime()
                engine.encodeChunk(audio).also {
                    report.put("cpu_encode_millis", elapsedMillis(started))
                }
            }
            report.put("memory_after_cpu_encode", memorySnapshot())

            val inputs = requestedSteps.map { step ->
                val prompt = WhisperSplitDecoderTensors.forcedPrompt(
                    generationConfig = generationConfig,
                    language = language,
                    referenceTokenPrefix = referenceChunk.tokenIds.take(step),
                )
                PrefixInput(
                    step = step,
                    promptLength = prompt.size,
                    tokens = WhisperSplitDecoderTensors.paddedTokens(
                        prompt = prompt,
                        paddingToken = generationConfig.eosTokenId,
                    ),
                )
            }

            val cpuLoadStarted = System.nanoTime()
            val cpuResults = DecoderPrefixCpuRunner.open(
                instrumentation.context.assets,
                boundarySpec.elementCount,
            ).use { runner ->
                report.put("cpu_prefix_model_load_millis", elapsedMillis(cpuLoadStarted))
                inputs.map { input ->
                    val started = System.nanoTime()
                    PrefixOutput(
                        values = runner.run(encoderState, input.tokens),
                        inferenceMillis = elapsedMillis(started),
                    )
                }
            }
            report.put("memory_after_cpu_prefix", memorySnapshot())

            val isolationDirectory =
                File(reportDirectory, ISOLATION_DIRECTORY).apply { mkdirs() }
            if (captureIsolatedKey) {
                require(stage == "after-self-key-transpose") {
                    "Key capture requires after-self-key-transpose."
                }
                val id = requireNotNull(isolationId) {
                    "$ARG_ISOLATION_ID is required for key capture."
                }
                inputs.indices.forEach { index ->
                    writeFloatFile(
                        isolationFile(isolationDirectory, id, inputs[index].step),
                        cpuResults[index].values,
                    )
                }
                report.put("captured_isolated_key_count", inputs.size)
            }
            if (captureIsolatedValue) {
                require(stage == "after-self-value-reshape") {
                    "Value capture requires after-self-value-reshape."
                }
                val id = requireNotNull(isolationId) {
                    "$ARG_ISOLATION_ID is required for value capture."
                }
                inputs.indices.forEach { index ->
                    writeFloatFile(
                        valueIsolationFile(
                            isolationDirectory,
                            id,
                            inputs[index].step,
                        ),
                        cpuResults[index].values,
                    )
                }
                report.put("captured_isolated_value_count", inputs.size)
            }

            val gpuLoadStarted = System.nanoTime()
            val gpuResults = DecoderPrefixGpuRunner.open(
                instrumentation.context.assets,
                allowSourceQuantization,
            ).use { runner ->
                report.put("gpu_prefix_model_load_millis", elapsedMillis(gpuLoadStarted))
                report.put("memory_after_gpu_open", memorySnapshot())
                inputs.map { input ->
                    val started = System.nanoTime()
                    PrefixOutput(
                        values = runner.run(encoderState, input.tokens),
                        inferenceMillis = elapsedMillis(started),
                    )
                }
            }
            report.put("memory_after_gpu_prefix", memorySnapshot())

            if (stage == "after-self-layer-norm") {
                val isolatedCpu = IsolatedKProjectionCpuRunner.open(
                    instrumentation.context.assets,
                ).use { runner ->
                    cpuResults.map { input ->
                        val started = System.nanoTime()
                        PrefixOutput(
                            values = runner.run(input.values),
                            inferenceMillis = elapsedMillis(started),
                        )
                    }
                }
                val isolatedGpu = IsolatedKProjectionGpuRunner.open(
                    instrumentation.context.assets,
                    allowSourceQuantization,
                ).use { runner ->
                    cpuResults.map { input ->
                        val started = System.nanoTime()
                        PrefixOutput(
                            values = runner.run(input.values),
                            inferenceMillis = elapsedMillis(started),
                        )
                    }
                }
                val hiddenSpec = boundarySpec("after-self-key-projection")
                val isolatedSteps = JSONArray()
                inputs.indices.forEach { index ->
                    val activePosition = inputs[index].promptLength - 1
                    isolatedSteps.put(
                        JSONObject()
                            .put("step", inputs[index].step)
                            .put(
                                "cpu_inference_millis",
                                isolatedCpu[index].inferenceMillis,
                            )
                            .put(
                                "gpu_inference_millis",
                                isolatedGpu[index].inferenceMillis,
                            )
                            .put(
                                "full_boundary",
                                compareArrays(
                                    isolatedCpu[index].values,
                                    isolatedGpu[index].values,
                                ),
                            )
                            .put(
                                "active_row",
                                compareArrays(
                                    hiddenSpec.activeValues(
                                        isolatedCpu[index].values,
                                        activePosition,
                                    ),
                                    hiddenSpec.activeValues(
                                        isolatedGpu[index].values,
                                        activePosition,
                                    ),
                                ),
                            ),
                    )
                }
                report.put(
                    "isolated_k_projection_same_cpu_input",
                    JSONObject()
                        .put("input_backend", "cpu-xnnpack-prefix")
                        .put("weight_type", "int8")
                        .put(
                            "allow_source_quantization",
                            allowSourceQuantization,
                        )
                        .put("steps", isolatedSteps),
                )
            }

            if (stage == "after-self-mask-add") {
                val isolatedCpu = IsolatedSoftmaxCpuRunner.open(
                    instrumentation.context.assets,
                ).use { runner ->
                    cpuResults.map { input ->
                        val started = System.nanoTime()
                        PrefixOutput(
                            values = runner.run(input.values),
                            inferenceMillis = elapsedMillis(started),
                        )
                    }
                }
                val isolatedGpu = IsolatedSoftmaxGpuRunner.open(
                    instrumentation.context.assets,
                ).use { runner ->
                    cpuResults.map { input ->
                        val started = System.nanoTime()
                        PrefixOutput(
                            values = runner.run(input.values),
                            inferenceMillis = elapsedMillis(started),
                        )
                    }
                }
                val isolatedSteps = JSONArray()
                inputs.indices.forEach { index ->
                    val activePosition = inputs[index].promptLength - 1
                    val cpu = isolatedCpu[index]
                    val gpu = isolatedGpu[index]
                    val softmaxSpec = boundarySpec("after-self-softmax")
                    isolatedSteps.put(
                        JSONObject()
                            .put("step", inputs[index].step)
                            .put("cpu_inference_millis", cpu.inferenceMillis)
                            .put("gpu_inference_millis", gpu.inferenceMillis)
                            .put(
                                "full_boundary",
                                compareArrays(cpu.values, gpu.values),
                            )
                            .put(
                                "active_row",
                                compareArrays(
                                    softmaxSpec.activeValues(
                                        cpu.values,
                                        activePosition,
                                    ),
                                    softmaxSpec.activeValues(
                                        gpu.values,
                                        activePosition,
                                    ),
                                ),
                            ),
                    )
                }
                report.put(
                    "isolated_softmax_same_cpu_input",
                    JSONObject()
                        .put("input_backend", "cpu-xnnpack-prefix")
                        .put("steps", isolatedSteps),
                )
            }

            if (stage == "after-self-score-matmul") {
                val isolatedCpu = IsolatedMaskAddCpuRunner.open(
                    instrumentation.context.assets,
                ).use { runner ->
                    cpuResults.map { input ->
                        val started = System.nanoTime()
                        PrefixOutput(
                            values = runner.run(input.values),
                            inferenceMillis = elapsedMillis(started),
                        )
                    }
                }
                val isolatedGpu = IsolatedMaskAddGpuRunner.open(
                    instrumentation.context.assets,
                ).use { runner ->
                    cpuResults.map { input ->
                        val started = System.nanoTime()
                        PrefixOutput(
                            values = runner.run(input.values),
                            inferenceMillis = elapsedMillis(started),
                        )
                    }
                }
                val maskAddSpec = boundarySpec("after-self-mask-add")
                val isolatedSteps = JSONArray()
                inputs.indices.forEach { index ->
                    val activePosition = inputs[index].promptLength - 1
                    val cpu = isolatedCpu[index]
                    val gpu = isolatedGpu[index]
                    isolatedSteps.put(
                        JSONObject()
                            .put("step", inputs[index].step)
                            .put("cpu_inference_millis", cpu.inferenceMillis)
                            .put("gpu_inference_millis", gpu.inferenceMillis)
                            .put(
                                "full_boundary",
                                compareArrays(cpu.values, gpu.values),
                            )
                            .put(
                                "active_row",
                                compareArrays(
                                    maskAddSpec.activeValues(
                                        cpu.values,
                                        activePosition,
                                    ),
                                    maskAddSpec.activeValues(
                                        gpu.values,
                                        activePosition,
                                    ),
                                ),
                            )
                            .put(
                                "unmasked_active_row",
                                compareArrays(
                                    maskAddSpec.unmaskedActiveAttentionValues(
                                        cpu.values,
                                        activePosition,
                                    ),
                                    maskAddSpec.unmaskedActiveAttentionValues(
                                        gpu.values,
                                        activePosition,
                                    ),
                                ),
                            ),
                    )
                }
                report.put(
                    "isolated_mask_add_same_cpu_input",
                    JSONObject()
                        .put("score_backend", "cpu-xnnpack-prefix")
                        .put("mask_backend", "shared-host-causal-mask")
                        .put("steps", isolatedSteps),
                )
            }

            if (runIsolatedScore) {
                require(stage == "after-self-query-reshape") {
                    "Isolated score BMM requires after-self-query-reshape."
                }
                val id = requireNotNull(isolationId) {
                    "$ARG_ISOLATION_ID is required for isolated score BMM."
                }
                val keyFiles = inputs.map { input ->
                    isolationFile(isolationDirectory, id, input.step)
                }
                val keys = keyFiles.map { file ->
                    readFloatFile(file, QUERY_KEY_FLOATS)
                }
                try {
                    val isolatedCpu = IsolatedScoreCpuRunner.open(
                        instrumentation.context.assets,
                    ).use { runner ->
                        inputs.indices.map { index ->
                            val started = System.nanoTime()
                            PrefixOutput(
                                values = runner.run(cpuResults[index].values, keys[index]),
                                inferenceMillis = elapsedMillis(started),
                            )
                        }
                    }
                    val isolatedGpu = IsolatedScoreGpuRunner.open(
                        instrumentation.context.assets,
                    ).use { runner ->
                        inputs.indices.map { index ->
                            val started = System.nanoTime()
                            PrefixOutput(
                                values = runner.run(cpuResults[index].values, keys[index]),
                                inferenceMillis = elapsedMillis(started),
                            )
                        }
                    }
                    val scoreSpec = boundarySpec("after-self-score-matmul")
                    val isolatedSteps = JSONArray()
                    inputs.indices.forEach { index ->
                        val activePosition = inputs[index].promptLength - 1
                        isolatedSteps.put(
                            JSONObject()
                                .put("step", inputs[index].step)
                                .put(
                                    "cpu_inference_millis",
                                    isolatedCpu[index].inferenceMillis,
                                )
                                .put(
                                    "gpu_inference_millis",
                                    isolatedGpu[index].inferenceMillis,
                                )
                                .put(
                                    "full_boundary",
                                    compareArrays(
                                        isolatedCpu[index].values,
                                        isolatedGpu[index].values,
                                    ),
                                )
                                .put(
                                    "active_row",
                                    compareArrays(
                                        scoreSpec.activeValues(
                                            isolatedCpu[index].values,
                                            activePosition,
                                        ),
                                        scoreSpec.activeValues(
                                            isolatedGpu[index].values,
                                            activePosition,
                                        ),
                                    ),
                                ),
                        )
                    }
                    report.put(
                        "isolated_score_same_cpu_inputs",
                        JSONObject()
                            .put("query_backend", "cpu-xnnpack-prefix")
                            .put("key_backend", "cpu-xnnpack-prefix")
                            .put("adj_x", false)
                            .put("adj_y", false)
                            .put("steps", isolatedSteps),
                    )
                } finally {
                    keyFiles.forEach(File::delete)
                }
            }

            if (runIsolatedValue) {
                require(stage == "after-self-softmax") {
                    "Isolated value BMM requires after-self-softmax."
                }
                val id = requireNotNull(isolationId) {
                    "$ARG_ISOLATION_ID is required for isolated value BMM."
                }
                val valueFiles = inputs.map { input ->
                    valueIsolationFile(isolationDirectory, id, input.step)
                }
                val values = valueFiles.map { file ->
                    readFloatFile(file, QUERY_KEY_FLOATS)
                }
                try {
                    val isolatedCpu = IsolatedValueCpuRunner.open(
                        instrumentation.context.assets,
                    ).use { runner ->
                        inputs.indices.map { index ->
                            val started = System.nanoTime()
                            PrefixOutput(
                                values = runner.run(
                                    cpuResults[index].values,
                                    values[index],
                                ),
                                inferenceMillis = elapsedMillis(started),
                            )
                        }
                    }
                    val isolatedGpu = IsolatedValueGpuRunner.open(
                        instrumentation.context.assets,
                    ).use { runner ->
                        inputs.indices.map { index ->
                            val started = System.nanoTime()
                            PrefixOutput(
                                values = runner.run(
                                    cpuResults[index].values,
                                    values[index],
                                ),
                                inferenceMillis = elapsedMillis(started),
                            )
                        }
                    }
                    val valueSpec = boundarySpec("after-self-value-matmul")
                    val isolatedSteps = JSONArray()
                    inputs.indices.forEach { index ->
                        val activePosition = inputs[index].promptLength - 1
                        isolatedSteps.put(
                            JSONObject()
                                .put("step", inputs[index].step)
                                .put(
                                    "cpu_inference_millis",
                                    isolatedCpu[index].inferenceMillis,
                                )
                                .put(
                                    "gpu_inference_millis",
                                    isolatedGpu[index].inferenceMillis,
                                )
                                .put(
                                    "full_boundary",
                                    compareArrays(
                                        isolatedCpu[index].values,
                                        isolatedGpu[index].values,
                                    ),
                                )
                                .put(
                                    "active_row",
                                    compareArrays(
                                        valueSpec.activeValues(
                                            isolatedCpu[index].values,
                                            activePosition,
                                        ),
                                        valueSpec.activeValues(
                                            isolatedGpu[index].values,
                                            activePosition,
                                        ),
                                    ),
                                ),
                        )
                    }
                    report.put(
                        "isolated_value_same_cpu_inputs",
                        JSONObject()
                            .put("probability_backend", "cpu-xnnpack-prefix")
                            .put("value_backend", "cpu-xnnpack-prefix")
                            .put("adj_x", false)
                            .put("adj_y", false)
                            .put("steps", isolatedSteps),
                    )
                } finally {
                    valueFiles.forEach(File::delete)
                }
            }

            val steps = JSONArray()
            inputs.indices.forEach { index ->
                val input = inputs[index]
                val cpu = cpuResults[index]
                val gpu = gpuResults[index]
                require(cpu.values.size == boundarySpec.elementCount)
                require(gpu.values.size == boundarySpec.elementCount)
                val activePosition = input.promptLength - 1
                val stepReport = JSONObject()
                    .put("step", input.step)
                    .put("prompt_length", input.promptLength)
                    .put("cpu_inference_millis", cpu.inferenceMillis)
                    .put("gpu_inference_millis", gpu.inferenceMillis)
                    .put(
                        "full_boundary",
                        compareArrays(cpu.values, gpu.values),
                    )
                    .put(
                        "active_row",
                        compareArrays(
                            boundarySpec.activeValues(cpu.values, activePosition),
                            boundarySpec.activeValues(gpu.values, activePosition),
                        ),
                    )
                if (boundarySpec.layout == DecoderBoundaryLayout.ATTENTION_ROWS) {
                    stepReport.put(
                        "unmasked_active_row",
                        compareArrays(
                            boundarySpec.unmaskedActiveAttentionValues(
                                cpu.values,
                                activePosition,
                            ),
                            boundarySpec.unmaskedActiveAttentionValues(
                                gpu.values,
                                activePosition,
                            ),
                        ),
                    )
                }
                steps.put(stepReport)
            }
            report
                .put("reference_token_count", referenceChunk.tokenIds.size)
                .put("steps", steps)
                .put("status", STATUS_COMPLETED)
        } catch (error: Throwable) {
            report
                .put("status", STATUS_FAILED)
                .put("error_type", error.javaClass.name)
                .put(
                    "error_message",
                    error.message?.lineSequence()?.firstOrNull()?.take(500)
                        ?: error.javaClass.simpleName,
                )
        }

        temporaryReport.writeText(report.toString(2))
        check(temporaryReport.renameTo(reportFile)) {
            "Could not publish rewritten decoder-prefix report."
        }
    }

    private fun compareArrays(reference: FloatArray, candidate: FloatArray): JSONObject {
        require(reference.size == candidate.size)
        var exact = 0
        var finite = 0
        var firstMismatch: Int? = null
        var maximumAbsoluteError = 0.0
        var absoluteErrorSum = 0.0
        var squaredErrorSum = 0.0
        var dotProduct = 0.0
        var referenceSquaredSum = 0.0
        var candidateSquaredSum = 0.0
        reference.indices.forEach { index ->
            val expected = reference[index]
            val actual = candidate[index]
            if (expected.toRawBits() == actual.toRawBits()) {
                exact++
            } else if (firstMismatch == null) {
                firstMismatch = index
            }
            if (!expected.isFinite() || !actual.isFinite()) return@forEach
            finite++
            val difference = actual.toDouble() - expected.toDouble()
            val absoluteError = kotlin.math.abs(difference)
            maximumAbsoluteError = maxOf(maximumAbsoluteError, absoluteError)
            absoluteErrorSum += absoluteError
            squaredErrorSum += difference * difference
            dotProduct += expected.toDouble() * actual.toDouble()
            referenceSquaredSum += expected.toDouble() * expected.toDouble()
            candidateSquaredSum += actual.toDouble() * actual.toDouble()
        }
        val divisor = finite.coerceAtLeast(1).toDouble()
        val cosineDenominator = sqrt(referenceSquaredSum * candidateSquaredSum)
        return JSONObject()
            .put("element_count", reference.size)
            .put("finite_pair_count", finite)
            .put("exact_element_count", exact)
            .put("first_exact_mismatch", firstMismatch ?: JSONObject.NULL)
            .put("maximum_absolute_error", maximumAbsoluteError)
            .put("mean_absolute_error", absoluteErrorSum / divisor)
            .put("root_mean_square_error", sqrt(squaredErrorSum / divisor))
            .put(
                "cosine_similarity",
                if (cosineDenominator == 0.0) {
                    JSONObject.NULL
                } else {
                    dotProduct / cosineDenominator
                },
            )
    }

    private fun parseSteps(raw: String?): List<Int> {
        val result = (raw ?: DEFAULT_STEPS)
            .split(',')
            .map { value ->
                value.toIntOrNull()
                    ?: error("$ARG_STEPS must be comma-separated integers.")
            }
            .distinct()
        require(result.isNotEmpty() && result.all { it in 0..123 }) {
            "$ARG_STEPS must contain values between 0 and 123."
        }
        return result
    }

    private fun requireSafeValue(value: String?, label: String): String {
        val result = value?.trim().orEmpty()
        require(result.matches(Regex("[A-Za-z0-9-]+"))) {
            "$label is missing or invalid."
        }
        return result
    }

    private fun elapsedMillis(startedNanos: Long): Long =
        (System.nanoTime() - startedNanos) / 1_000_000L

    private fun memorySnapshot(): JSONObject {
        val memory = Debug.MemoryInfo()
        Debug.getMemoryInfo(memory)
        return JSONObject()
            .put("total_pss_kib", memory.totalPss)
            .put("native_heap_allocated_bytes", Debug.getNativeHeapAllocatedSize())
    }

    private data class PrefixInput(
        val step: Int,
        val promptLength: Int,
        val tokens: IntArray,
    )

    private data class PrefixOutput(
        val values: FloatArray,
        val inferenceMillis: Long,
    )

    companion object {
        private const val ARG_MEMO_ID = "memoId"
        private const val ARG_REFERENCE_JOB = "referenceJob"
        private const val ARG_RUN_ID = "runId"
        private const val ARG_STAGE = "stage"
        private const val ARG_MODEL_VARIANT = "modelVariant"
        private const val ARG_STEPS = "steps"
        private const val ARG_ALLOW_SOURCE_QUANTIZATION = "allowSourceQuantization"
        private const val ARG_ISOLATION_ID = "isolationId"
        private const val ARG_CAPTURE_ISOLATED_KEY = "captureIsolatedKey"
        private const val ARG_RUN_ISOLATED_SCORE = "runIsolatedScore"
        private const val ARG_CAPTURE_ISOLATED_VALUE = "captureIsolatedValue"
        private const val ARG_RUN_ISOLATED_VALUE = "runIsolatedValue"
        private const val DEFAULT_STEPS = "0,7"
        private const val REPORT_DIRECTORY = "gpu-prefix"
        private const val ISOLATION_DIRECTORY = "isolation"
        private const val STATUS_COMPLETED = "completed"
        private const val STATUS_FAILED = "failed"
    }
}

private data class DecoderBoundarySpec(
    val shape: String,
    val elementCount: Int,
    val layout: DecoderBoundaryLayout,
) {
    fun activeValues(values: FloatArray, activePosition: Int): FloatArray {
        require(values.size == elementCount)
        require(activePosition in 0 until WhisperSplitDecoderTensors.TARGET_LENGTH)
        return when (layout) {
            DecoderBoundaryLayout.HIDDEN_ROWS -> {
                val start = activePosition * WhisperSplitDecoderTensors.ENCODER_WIDTH
                values.copyOfRange(
                    start,
                    start + WhisperSplitDecoderTensors.ENCODER_WIDTH,
                )
            }

            DecoderBoundaryLayout.TRANSPOSED_KEYS -> {
                FloatArray(ATTENTION_HEADS * ATTENTION_HEAD_WIDTH).also { active ->
                    for (head in 0 until ATTENTION_HEADS) {
                        for (channel in 0 until ATTENTION_HEAD_WIDTH) {
                            val destination = head * ATTENTION_HEAD_WIDTH + channel
                            val source =
                                (destination * WhisperSplitDecoderTensors.TARGET_LENGTH) +
                                    activePosition
                            active[destination] = values[source]
                        }
                    }
                }
            }

            DecoderBoundaryLayout.ATTENTION_ROWS -> {
                FloatArray(
                    ATTENTION_HEADS * WhisperSplitDecoderTensors.TARGET_LENGTH,
                ).also { active ->
                    for (head in 0 until ATTENTION_HEADS) {
                        val source =
                            (
                                head * WhisperSplitDecoderTensors.TARGET_LENGTH +
                                    activePosition
                                ) * WhisperSplitDecoderTensors.TARGET_LENGTH
                        values.copyInto(
                            destination = active,
                            destinationOffset =
                                head * WhisperSplitDecoderTensors.TARGET_LENGTH,
                            startIndex = source,
                            endIndex =
                                source + WhisperSplitDecoderTensors.TARGET_LENGTH,
                        )
                    }
                }
            }

            DecoderBoundaryLayout.HEAD_VALUE_ROWS -> {
                FloatArray(ATTENTION_HEADS * ATTENTION_HEAD_WIDTH).also { active ->
                    for (head in 0 until ATTENTION_HEADS) {
                        val source =
                            (
                                head * WhisperSplitDecoderTensors.TARGET_LENGTH +
                                    activePosition
                                ) * ATTENTION_HEAD_WIDTH
                        values.copyInto(
                            destination = active,
                            destinationOffset = head * ATTENTION_HEAD_WIDTH,
                            startIndex = source,
                            endIndex = source + ATTENTION_HEAD_WIDTH,
                        )
                    }
                }
            }
        }
    }

    /**
     * Returns only causally valid keys for the active query, in head-major order.
     *
     * Comparing the complete row after mask addition is misleading because the
     * many masked -1e9 entries dominate cosine and error summaries. This slice
     * isolates the values that can actually influence the active-token softmax.
     */
    fun unmaskedActiveAttentionValues(
        values: FloatArray,
        activePosition: Int,
    ): FloatArray {
        require(layout == DecoderBoundaryLayout.ATTENTION_ROWS)
        require(values.size == elementCount)
        require(activePosition in 0 until WhisperSplitDecoderTensors.TARGET_LENGTH)
        val validKeyCount = activePosition + 1
        return FloatArray(ATTENTION_HEADS * validKeyCount).also { active ->
            for (head in 0 until ATTENTION_HEADS) {
                val source =
                    (
                        head * WhisperSplitDecoderTensors.TARGET_LENGTH +
                            activePosition
                        ) * WhisperSplitDecoderTensors.TARGET_LENGTH
                values.copyInto(
                    destination = active,
                    destinationOffset = head * validKeyCount,
                    startIndex = source,
                    endIndex = source + validKeyCount,
                )
            }
        }
    }
}

private enum class DecoderBoundaryLayout {
    HIDDEN_ROWS,
    TRANSPOSED_KEYS,
    ATTENTION_ROWS,
    HEAD_VALUE_ROWS,
}

private fun boundarySpec(stage: String): DecoderBoundarySpec =
    when (stage) {
        "after-self-key-transpose" ->
            DecoderBoundarySpec(
                shape = "[20,64,128]",
                elementCount = ATTENTION_HEADS * ATTENTION_HEAD_WIDTH *
                    WhisperSplitDecoderTensors.TARGET_LENGTH,
                layout = DecoderBoundaryLayout.TRANSPOSED_KEYS,
            )

        "after-self-score-matmul",
        "after-self-softmax" ->
            DecoderBoundarySpec(
                shape = "[20,128,128]",
                elementCount = ATTENTION_HEADS *
                    WhisperSplitDecoderTensors.TARGET_LENGTH *
                    WhisperSplitDecoderTensors.TARGET_LENGTH,
                layout = DecoderBoundaryLayout.ATTENTION_ROWS,
            )

        "after-self-mask-add" ->
            DecoderBoundarySpec(
                shape = "[1,20,128,128]",
                elementCount = ATTENTION_HEADS *
                    WhisperSplitDecoderTensors.TARGET_LENGTH *
                    WhisperSplitDecoderTensors.TARGET_LENGTH,
                layout = DecoderBoundaryLayout.ATTENTION_ROWS,
            )

        "after-self-value-reshape",
        "after-self-value-matmul",
        "after-self-query-reshape" ->
            DecoderBoundarySpec(
                shape = "[20,128,64]",
                elementCount = ATTENTION_HEADS *
                    WhisperSplitDecoderTensors.TARGET_LENGTH *
                    ATTENTION_HEAD_WIDTH,
                layout = DecoderBoundaryLayout.HEAD_VALUE_ROWS,
            )

        "after-self-context-reshape",
        "after-self-output-projection" ->
            DecoderBoundarySpec(
                shape = "[128,1280]",
                elementCount = WhisperSplitDecoderTensors.DECODER_HIDDEN_FLOATS,
                layout = DecoderBoundaryLayout.HIDDEN_ROWS,
            )

        "after-position-add",
        "after-self-layer-norm",
        "after-self-key-projection",
        "after-self-query-projection",
        "after-self-value-projection",
        "after-self-attention",
        "after-cross-attention",
        "after-1-block",
        "after-2-blocks",
        "after-3-blocks",
        "after-4-blocks",
        "after-final-layer-norm" ->
            DecoderBoundarySpec(
                shape = "[1,128,1280]",
                elementCount = WhisperSplitDecoderTensors.DECODER_HIDDEN_FLOATS,
                layout = DecoderBoundaryLayout.HIDDEN_ROWS,
            )

        else -> error("Unsupported rewritten decoder-prefix stage: $stage.")
    }

private class DecoderPrefixCpuRunner private constructor(
    private val interpreter: Interpreter,
    private val outputElementCount: Int,
) : AutoCloseable {
    private val encoder = directFloatBuffer(ENCODER_FLOATS)
    private val tokens = directIntBuffer(WhisperSplitDecoderTensors.TARGET_LENGTH)
    private val mask = causalMaskBuffer()
    private val output = directFloatBuffer(outputElementCount)

    fun run(encoderState: FloatArray, tokenValues: IntArray): FloatArray {
        encoder.putFloatValues(encoderState)
        tokens.putIntValues(tokenValues)
        mask.rewind()
        output.clear()
        interpreter.runSignature(
            mapOf(
                INPUT_ENCODER to encoder,
                INPUT_TOKENS to tokens,
                INPUT_MASK to mask,
            ),
            mapOf(OUTPUT_BOUNDARY to output),
            SIGNATURE,
        )
        return output.readFloatValues(outputElementCount)
    }

    override fun close() {
        interpreter.close()
    }

    companion object {
        fun open(
            assets: AssetManager,
            outputElementCount: Int,
        ): DecoderPrefixCpuRunner {
            val interpreter = Interpreter(
                mapAsset(assets),
                Interpreter.Options()
                    .setNumThreads(3)
                    .setUseXNNPACK(true)
                    .setCancellable(true),
            )
            return DecoderPrefixCpuRunner(interpreter, outputElementCount)
        }
    }
}

private class DecoderPrefixGpuRunner private constructor(
    private val model: CompiledModel,
    private val encoder: TensorBuffer,
    private val tokens: TensorBuffer,
    private val mask: TensorBuffer,
    private val output: TensorBuffer,
) : AutoCloseable {
    fun run(encoderState: FloatArray, tokenValues: IntArray): FloatArray {
        encoder.writeFloat(encoderState)
        tokens.writeInt(tokenValues)
        model.run(
            mapOf(
                INPUT_ENCODER to encoder,
                INPUT_TOKENS to tokens,
                INPUT_MASK to mask,
            ),
            mapOf(OUTPUT_BOUNDARY to output),
            SIGNATURE,
        )
        return output.readFloat()
    }

    override fun close() {
        listOf(output, mask, tokens, encoder).forEach { runCatching(it::close) }
        model.close()
    }

    companion object {
        fun open(
            assets: AssetManager,
            allowSourceQuantization: Boolean,
        ): DecoderPrefixGpuRunner {
            val options =
                CompiledModel.Options(Accelerator.GPU, Accelerator.CPU).apply {
                    gpuOptions = CompiledModel.GpuOptions(
                        constantTensorSharing = true,
                        allowSrcQuantizedFcConvOps = allowSourceQuantization,
                        precision = CompiledModel.GpuOptions.Precision.FP32,
                        backend = CompiledModel.GpuOptions.Backend.OPENCL,
                    )
                }
            val model = CompiledModel.create(assets, ASSET_PATH, options)
            val allocated = mutableListOf<TensorBuffer>()
            return try {
                fun input(name: String) =
                    model.createInputBuffer(name, SIGNATURE).also(allocated::add)
                fun output(name: String) =
                    model.createOutputBuffer(name, SIGNATURE).also(allocated::add)
                val mask = input(INPUT_MASK)
                mask.writeFloat(WhisperSplitDecoderTensors.causalMask())
                DecoderPrefixGpuRunner(
                    model = model,
                    encoder = input(INPUT_ENCODER),
                    tokens = input(INPUT_TOKENS),
                    mask = mask,
                    output = output(OUTPUT_BOUNDARY),
                )
            } catch (error: Throwable) {
                allocated.asReversed().forEach { runCatching(it::close) }
                model.close()
                throw error
            }
        }
    }
}

private class IsolatedSoftmaxCpuRunner private constructor(
    private val interpreter: Interpreter,
) : AutoCloseable {
    private val input = directFloatBuffer(ATTENTION_SCORE_FLOATS)
    private val output = directFloatBuffer(ATTENTION_SCORE_FLOATS)

    fun run(values: FloatArray): FloatArray {
        require(values.size == ATTENTION_SCORE_FLOATS)
        input.putFloatValues(values)
        output.clear()
        interpreter.runSignature(
            mapOf(SOFTMAX_INPUT to input),
            mapOf(SOFTMAX_OUTPUT to output),
            SOFTMAX_SIGNATURE,
        )
        return output.readFloatValues(ATTENTION_SCORE_FLOATS)
    }

    override fun close() {
        interpreter.close()
    }

    companion object {
        fun open(assets: AssetManager): IsolatedSoftmaxCpuRunner =
            IsolatedSoftmaxCpuRunner(
                Interpreter(
                    mapAsset(assets, SOFTMAX_ASSET_PATH),
                    Interpreter.Options()
                        .setNumThreads(3)
                        .setUseXNNPACK(true)
                        .setCancellable(true),
                ),
            )
    }
}

private class IsolatedSoftmaxGpuRunner private constructor(
    private val model: CompiledModel,
    private val input: TensorBuffer,
    private val output: TensorBuffer,
) : AutoCloseable {
    fun run(values: FloatArray): FloatArray {
        require(values.size == ATTENTION_SCORE_FLOATS)
        input.writeFloat(values)
        model.run(
            mapOf(SOFTMAX_INPUT to input),
            mapOf(SOFTMAX_OUTPUT to output),
            SOFTMAX_SIGNATURE,
        )
        return output.readFloat()
    }

    override fun close() {
        runCatching(output::close)
        runCatching(input::close)
        model.close()
    }

    companion object {
        fun open(assets: AssetManager): IsolatedSoftmaxGpuRunner {
            val options =
                CompiledModel.Options(Accelerator.GPU, Accelerator.CPU).apply {
                    gpuOptions = CompiledModel.GpuOptions(
                        constantTensorSharing = true,
                        allowSrcQuantizedFcConvOps = false,
                        precision = CompiledModel.GpuOptions.Precision.FP32,
                        backend = CompiledModel.GpuOptions.Backend.OPENCL,
                    )
                }
            val model = CompiledModel.create(assets, SOFTMAX_ASSET_PATH, options)
            val input = model.createInputBuffer(SOFTMAX_INPUT, SOFTMAX_SIGNATURE)
            val output = model.createOutputBuffer(SOFTMAX_OUTPUT, SOFTMAX_SIGNATURE)
            return IsolatedSoftmaxGpuRunner(model, input, output)
        }
    }
}

private class IsolatedMaskAddCpuRunner private constructor(
    private val interpreter: Interpreter,
) : AutoCloseable {
    private val scores = directFloatBuffer(ATTENTION_SCORE_FLOATS)
    private val mask = causalMaskBuffer()
    private val output = directFloatBuffer(ATTENTION_SCORE_FLOATS)

    fun run(scoreValues: FloatArray): FloatArray {
        require(scoreValues.size == ATTENTION_SCORE_FLOATS)
        scores.putFloatValues(scoreValues)
        mask.rewind()
        output.clear()
        interpreter.runSignature(
            mapOf(
                MASK_ADD_SCORES to scores,
                MASK_ADD_MASK to mask,
            ),
            mapOf(MASK_ADD_OUTPUT to output),
            MASK_ADD_SIGNATURE,
        )
        return output.readFloatValues(ATTENTION_SCORE_FLOATS)
    }

    override fun close() {
        interpreter.close()
    }

    companion object {
        fun open(assets: AssetManager): IsolatedMaskAddCpuRunner =
            IsolatedMaskAddCpuRunner(
                Interpreter(
                    mapAsset(assets, MASK_ADD_ASSET_PATH),
                    Interpreter.Options()
                        .setNumThreads(3)
                        .setUseXNNPACK(true)
                        .setCancellable(true),
                ),
            )
    }
}

private class IsolatedMaskAddGpuRunner private constructor(
    private val model: CompiledModel,
    private val scores: TensorBuffer,
    private val mask: TensorBuffer,
    private val output: TensorBuffer,
) : AutoCloseable {
    fun run(scoreValues: FloatArray): FloatArray {
        require(scoreValues.size == ATTENTION_SCORE_FLOATS)
        scores.writeFloat(scoreValues)
        model.run(
            mapOf(
                MASK_ADD_SCORES to scores,
                MASK_ADD_MASK to mask,
            ),
            mapOf(MASK_ADD_OUTPUT to output),
            MASK_ADD_SIGNATURE,
        )
        return output.readFloat()
    }

    override fun close() {
        runCatching(output::close)
        runCatching(mask::close)
        runCatching(scores::close)
        model.close()
    }

    companion object {
        fun open(assets: AssetManager): IsolatedMaskAddGpuRunner {
            val options =
                CompiledModel.Options(Accelerator.GPU, Accelerator.CPU).apply {
                    gpuOptions = CompiledModel.GpuOptions(
                        constantTensorSharing = true,
                        allowSrcQuantizedFcConvOps = false,
                        precision = CompiledModel.GpuOptions.Precision.FP32,
                        backend = CompiledModel.GpuOptions.Backend.OPENCL,
                    )
                }
            val model = CompiledModel.create(assets, MASK_ADD_ASSET_PATH, options)
            val scores =
                model.createInputBuffer(MASK_ADD_SCORES, MASK_ADD_SIGNATURE)
            val mask =
                model.createInputBuffer(MASK_ADD_MASK, MASK_ADD_SIGNATURE).also {
                    it.writeFloat(WhisperSplitDecoderTensors.causalMask())
                }
            val output =
                model.createOutputBuffer(MASK_ADD_OUTPUT, MASK_ADD_SIGNATURE)
            return IsolatedMaskAddGpuRunner(model, scores, mask, output)
        }
    }
}

private class IsolatedScoreCpuRunner private constructor(
    private val interpreter: Interpreter,
) : AutoCloseable {
    private val query = directFloatBuffer(QUERY_KEY_FLOATS)
    private val key = directFloatBuffer(QUERY_KEY_FLOATS)
    private val output = directFloatBuffer(ATTENTION_SCORE_FLOATS)

    fun run(queryValues: FloatArray, keyValues: FloatArray): FloatArray {
        require(queryValues.size == QUERY_KEY_FLOATS)
        require(keyValues.size == QUERY_KEY_FLOATS)
        query.putFloatValues(queryValues)
        key.putFloatValues(keyValues)
        output.clear()
        interpreter.runSignature(
            mapOf(
                SCORE_QUERY to query,
                SCORE_KEY to key,
            ),
            mapOf(SCORE_OUTPUT to output),
            SCORE_SIGNATURE,
        )
        return output.readFloatValues(ATTENTION_SCORE_FLOATS)
    }

    override fun close() {
        interpreter.close()
    }

    companion object {
        fun open(assets: AssetManager): IsolatedScoreCpuRunner =
            IsolatedScoreCpuRunner(
                Interpreter(
                    mapAsset(assets, SCORE_ASSET_PATH),
                    Interpreter.Options()
                        .setNumThreads(3)
                        .setUseXNNPACK(true)
                        .setCancellable(true),
                ),
            )
    }
}

private class IsolatedScoreGpuRunner private constructor(
    private val model: CompiledModel,
    private val query: TensorBuffer,
    private val key: TensorBuffer,
    private val output: TensorBuffer,
) : AutoCloseable {
    fun run(queryValues: FloatArray, keyValues: FloatArray): FloatArray {
        require(queryValues.size == QUERY_KEY_FLOATS)
        require(keyValues.size == QUERY_KEY_FLOATS)
        query.writeFloat(queryValues)
        key.writeFloat(keyValues)
        model.run(
            mapOf(
                SCORE_QUERY to query,
                SCORE_KEY to key,
            ),
            mapOf(SCORE_OUTPUT to output),
            SCORE_SIGNATURE,
        )
        return output.readFloat()
    }

    override fun close() {
        runCatching(output::close)
        runCatching(key::close)
        runCatching(query::close)
        model.close()
    }

    companion object {
        fun open(assets: AssetManager): IsolatedScoreGpuRunner {
            val options =
                CompiledModel.Options(Accelerator.GPU, Accelerator.CPU).apply {
                    gpuOptions = CompiledModel.GpuOptions(
                        constantTensorSharing = true,
                        allowSrcQuantizedFcConvOps = false,
                        precision = CompiledModel.GpuOptions.Precision.FP32,
                        backend = CompiledModel.GpuOptions.Backend.OPENCL,
                    )
                }
            val model = CompiledModel.create(assets, SCORE_ASSET_PATH, options)
            val query = model.createInputBuffer(SCORE_QUERY, SCORE_SIGNATURE)
            val key = model.createInputBuffer(SCORE_KEY, SCORE_SIGNATURE)
            val output = model.createOutputBuffer(SCORE_OUTPUT, SCORE_SIGNATURE)
            return IsolatedScoreGpuRunner(model, query, key, output)
        }
    }
}

private class IsolatedValueCpuRunner private constructor(
    private val interpreter: Interpreter,
) : AutoCloseable {
    private val probabilities = directFloatBuffer(ATTENTION_SCORE_FLOATS)
    private val values = directFloatBuffer(QUERY_KEY_FLOATS)
    private val output = directFloatBuffer(QUERY_KEY_FLOATS)

    fun run(
        probabilityValues: FloatArray,
        valueValues: FloatArray,
    ): FloatArray {
        require(probabilityValues.size == ATTENTION_SCORE_FLOATS)
        require(valueValues.size == QUERY_KEY_FLOATS)
        probabilities.putFloatValues(probabilityValues)
        values.putFloatValues(valueValues)
        output.clear()
        interpreter.runSignature(
            mapOf(
                VALUE_PROBABILITIES to probabilities,
                VALUE_VALUES to values,
            ),
            mapOf(VALUE_OUTPUT to output),
            VALUE_SIGNATURE,
        )
        return output.readFloatValues(QUERY_KEY_FLOATS)
    }

    override fun close() {
        interpreter.close()
    }

    companion object {
        fun open(assets: AssetManager): IsolatedValueCpuRunner =
            IsolatedValueCpuRunner(
                Interpreter(
                    mapAsset(assets, VALUE_ASSET_PATH),
                    Interpreter.Options()
                        .setNumThreads(3)
                        .setUseXNNPACK(true)
                        .setCancellable(true),
                ),
            )
    }
}

private class IsolatedValueGpuRunner private constructor(
    private val model: CompiledModel,
    private val probabilities: TensorBuffer,
    private val values: TensorBuffer,
    private val output: TensorBuffer,
) : AutoCloseable {
    fun run(
        probabilityValues: FloatArray,
        valueValues: FloatArray,
    ): FloatArray {
        require(probabilityValues.size == ATTENTION_SCORE_FLOATS)
        require(valueValues.size == QUERY_KEY_FLOATS)
        probabilities.writeFloat(probabilityValues)
        values.writeFloat(valueValues)
        model.run(
            mapOf(
                VALUE_PROBABILITIES to probabilities,
                VALUE_VALUES to values,
            ),
            mapOf(VALUE_OUTPUT to output),
            VALUE_SIGNATURE,
        )
        return output.readFloat()
    }

    override fun close() {
        runCatching(output::close)
        runCatching(values::close)
        runCatching(probabilities::close)
        model.close()
    }

    companion object {
        fun open(assets: AssetManager): IsolatedValueGpuRunner {
            val options =
                CompiledModel.Options(Accelerator.GPU, Accelerator.CPU).apply {
                    gpuOptions = CompiledModel.GpuOptions(
                        constantTensorSharing = true,
                        allowSrcQuantizedFcConvOps = false,
                        precision = CompiledModel.GpuOptions.Precision.FP32,
                        backend = CompiledModel.GpuOptions.Backend.OPENCL,
                    )
                }
            val model = CompiledModel.create(assets, VALUE_ASSET_PATH, options)
            val probabilities =
                model.createInputBuffer(VALUE_PROBABILITIES, VALUE_SIGNATURE)
            val values = model.createInputBuffer(VALUE_VALUES, VALUE_SIGNATURE)
            val output = model.createOutputBuffer(VALUE_OUTPUT, VALUE_SIGNATURE)
            return IsolatedValueGpuRunner(
                model,
                probabilities,
                values,
                output,
            )
        }
    }
}

private class IsolatedKProjectionCpuRunner private constructor(
    private val interpreter: Interpreter,
) : AutoCloseable {
    private val input =
        directFloatBuffer(WhisperSplitDecoderTensors.DECODER_HIDDEN_FLOATS)
    private val output =
        directFloatBuffer(WhisperSplitDecoderTensors.DECODER_HIDDEN_FLOATS)

    fun run(values: FloatArray): FloatArray {
        require(values.size == WhisperSplitDecoderTensors.DECODER_HIDDEN_FLOATS)
        input.putFloatValues(values)
        output.clear()
        interpreter.runSignature(
            mapOf(K_PROJECTION_INPUT to input),
            mapOf(K_PROJECTION_OUTPUT to output),
            K_PROJECTION_SIGNATURE,
        )
        return output.readFloatValues(
            WhisperSplitDecoderTensors.DECODER_HIDDEN_FLOATS,
        )
    }

    override fun close() {
        interpreter.close()
    }

    companion object {
        fun open(assets: AssetManager): IsolatedKProjectionCpuRunner =
            IsolatedKProjectionCpuRunner(
                Interpreter(
                    mapAsset(assets, K_PROJECTION_ASSET_PATH),
                    Interpreter.Options()
                        .setNumThreads(3)
                        .setUseXNNPACK(true)
                        .setCancellable(true),
                ),
            )
    }
}

private class IsolatedKProjectionGpuRunner private constructor(
    private val model: CompiledModel,
    private val input: TensorBuffer,
    private val output: TensorBuffer,
) : AutoCloseable {
    fun run(values: FloatArray): FloatArray {
        require(values.size == WhisperSplitDecoderTensors.DECODER_HIDDEN_FLOATS)
        input.writeFloat(values)
        model.run(
            mapOf(K_PROJECTION_INPUT to input),
            mapOf(K_PROJECTION_OUTPUT to output),
            K_PROJECTION_SIGNATURE,
        )
        return output.readFloat()
    }

    override fun close() {
        runCatching(output::close)
        runCatching(input::close)
        model.close()
    }

    companion object {
        fun open(
            assets: AssetManager,
            allowSourceQuantization: Boolean,
        ): IsolatedKProjectionGpuRunner {
            val options =
                CompiledModel.Options(Accelerator.GPU, Accelerator.CPU).apply {
                    gpuOptions = CompiledModel.GpuOptions(
                        constantTensorSharing = true,
                        allowSrcQuantizedFcConvOps = allowSourceQuantization,
                        precision = CompiledModel.GpuOptions.Precision.FP32,
                        backend = CompiledModel.GpuOptions.Backend.OPENCL,
                    )
                }
            val model =
                CompiledModel.create(assets, K_PROJECTION_ASSET_PATH, options)
            val input =
                model.createInputBuffer(
                    K_PROJECTION_INPUT,
                    K_PROJECTION_SIGNATURE,
                )
            val output =
                model.createOutputBuffer(
                    K_PROJECTION_OUTPUT,
                    K_PROJECTION_SIGNATURE,
                )
            return IsolatedKProjectionGpuRunner(model, input, output)
        }
    }
}

private fun isolationFile(directory: File, isolationId: String, step: Int): File =
    File(directory, "score-$isolationId-step-$step-key.f32")

private fun valueIsolationFile(
    directory: File,
    isolationId: String,
    step: Int,
): File = File(directory, "value-$isolationId-step-$step-values.f32")

private fun writeFloatFile(file: File, values: FloatArray) {
    val bytes = ByteBuffer.allocate(values.size * Float.SIZE_BYTES)
        .order(ByteOrder.LITTLE_ENDIAN)
    bytes.asFloatBuffer().put(values)
    val temporary = File(file.parentFile, "${file.name}.tmp")
    temporary.writeBytes(bytes.array())
    check(temporary.renameTo(file)) {
        "Could not publish isolated score key input."
    }
}

private fun readFloatFile(file: File, expectedElements: Int): FloatArray {
    require(file.isFile && file.length() == expectedElements * Float.SIZE_BYTES.toLong()) {
        "Isolated score key input is missing or has the wrong size."
    }
    val bytes = ByteBuffer.wrap(file.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
    return FloatArray(expectedElements).also { values ->
        bytes.asFloatBuffer().get(values)
    }
}

private const val ASSET_PATH = "whisper-decoder-prefix/decoder-prefix.tflite"
private const val SOFTMAX_ASSET_PATH = "whisper-decoder-prefix/softmax-probe.tflite"
private const val MASK_ADD_ASSET_PATH = "whisper-decoder-prefix/mask-add-probe.tflite"
private const val SCORE_ASSET_PATH = "whisper-decoder-prefix/score-probe.tflite"
private const val VALUE_ASSET_PATH = "whisper-decoder-prefix/value-probe.tflite"
private const val K_PROJECTION_ASSET_PATH =
    "whisper-decoder-prefix/k-projection-probe.tflite"
private const val SIGNATURE = "decode"
private const val SOFTMAX_SIGNATURE = "softmax"
private const val MASK_ADD_SIGNATURE = "mask_add"
private const val SCORE_SIGNATURE = "score"
private const val VALUE_SIGNATURE = "value_matmul"
private const val K_PROJECTION_SIGNATURE = "k_projection"
private const val INPUT_ENCODER = "args_0"
private const val INPUT_TOKENS = "args_1"
private const val INPUT_MASK = "args_2"
private const val OUTPUT_BOUNDARY = "output_0"
private const val SOFTMAX_INPUT = "input_0"
private const val SOFTMAX_OUTPUT = "output_0"
private const val MASK_ADD_SCORES = "scores"
private const val MASK_ADD_MASK = "mask"
private const val MASK_ADD_OUTPUT = "masked_scores"
private const val SCORE_QUERY = "query"
private const val SCORE_KEY = "key"
private const val SCORE_OUTPUT = "scores"
private const val VALUE_PROBABILITIES = "probabilities"
private const val VALUE_VALUES = "values"
private const val VALUE_OUTPUT = "context"
private const val K_PROJECTION_INPUT = "hidden"
private const val K_PROJECTION_OUTPUT = "key"
private const val ENCODER_FLOATS =
    WhisperSplitDecoderTensors.ENCODER_FRAMES * WhisperSplitDecoderTensors.ENCODER_WIDTH
private const val ATTENTION_HEADS = 20
private const val ATTENTION_HEAD_WIDTH = 64
private const val ATTENTION_SCORE_FLOATS =
    ATTENTION_HEADS *
        WhisperSplitDecoderTensors.TARGET_LENGTH *
        WhisperSplitDecoderTensors.TARGET_LENGTH
private const val QUERY_KEY_FLOATS =
    ATTENTION_HEADS *
        WhisperSplitDecoderTensors.TARGET_LENGTH *
        ATTENTION_HEAD_WIDTH

private fun mapAsset(
    assets: AssetManager,
    assetPath: String = ASSET_PATH,
): MappedByteBuffer =
    assets.openFd(assetPath).use { descriptor ->
        FileInputStream(descriptor.fileDescriptor).channel.use { channel ->
            channel.map(
                FileChannel.MapMode.READ_ONLY,
                descriptor.startOffset,
                descriptor.declaredLength,
            )
        }
    }

private fun directFloatBuffer(elements: Int): ByteBuffer =
    ByteBuffer.allocateDirect(elements * Float.SIZE_BYTES)
        .order(ByteOrder.nativeOrder())

private fun directIntBuffer(elements: Int): ByteBuffer =
    ByteBuffer.allocateDirect(elements * Int.SIZE_BYTES)
        .order(ByteOrder.nativeOrder())

private fun causalMaskBuffer(): ByteBuffer =
    directFloatBuffer(
        WhisperSplitDecoderTensors.TARGET_LENGTH *
            WhisperSplitDecoderTensors.TARGET_LENGTH,
    ).also { buffer ->
        buffer.asFloatBuffer().put(WhisperSplitDecoderTensors.causalMask())
        buffer.rewind()
    }

private fun ByteBuffer.putFloatValues(values: FloatArray) {
    clear()
    asFloatBuffer().put(values)
    rewind()
}

private fun ByteBuffer.putIntValues(values: IntArray) {
    clear()
    asIntBuffer().put(values)
    rewind()
}

private fun ByteBuffer.readFloatValues(count: Int): FloatArray {
    rewind()
    return FloatArray(count).also { values ->
        asFloatBuffer().get(values)
        rewind()
    }
}
