package com.example.voicememo.interpretergpuprobe

import android.content.Context
import android.os.Debug
import android.util.Base64
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.security.MessageDigest
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.gpu.GpuDelegateFactory

/**
 * Standalone standard-Interpreter GPU A/B.
 *
 * This test deliberately lives in its own package and process because LiteRT
 * 2.1.x's Maven Interpreter is CPU-only, while the separately published
 * standard GPU delegate remains on the 1.4.x line. The probe opens Voice
 * Memo's uncompressed bundled model through a package context, never accesses
 * recordings, and never writes to Voice Memo.
 */
@RunWith(AndroidJUnit4::class)
class InterpreterGpuAbTest {
    @Test
    fun compareCpuAndOpenClTeacherForcedDecoder() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val arguments = InstrumentationRegistry.getArguments()
        val runId = requiredArgument(arguments.getString(ARG_RUN_ID), ARG_RUN_ID)
        val modelAsset = arguments.getString(ARG_MODEL_ASSET)
            ?.takeIf(String::isNotBlank)
            ?: DEFAULT_MODEL_ASSET
        val modelFileName = arguments.getString(ARG_MODEL_FILE)
            ?.takeIf(String::isNotBlank)
        val mode = arguments.getString(ARG_MODE)
            ?.takeIf(String::isNotBlank)
            ?: MODE_FULL_DECODER
        require(mode == MODE_FULL_DECODER || mode == MODE_BOUNDARY_AFTER_TWO) {
            "Unsupported probe mode: $mode"
        }
        val reportDirectory = File(context.cacheDir, REPORT_DIRECTORY).apply {
            check(exists() || mkdirs()) { "Could not create probe report directory." }
        }
        val reportFile = File(reportDirectory, "$runId.json")
        val temporaryReport = File(reportDirectory, "$runId.json.tmp")
        reportFile.delete()
        temporaryReport.delete()

        val report = JSONObject()
            .put("run_id", runId)
            .put("voice_memo_package", VOICE_MEMO_PACKAGE)
            .put("model_asset", modelAsset)
            .put("model_file", modelFileName ?: JSONObject.NULL)
            .put("mode", mode)
            .put("interpreter_version", "1.4.2")
            .put("gpu_delegate_version", "1.4.2")
            .put("persisted_output", false)
            .put(
                "fixture",
                if (mode == MODE_FULL_DECODER) {
                    "deterministic-mel-russian-prefixes-v1"
                } else {
                    "deterministic-encoder-state-russian-prefixes-v1"
                },
            )

        try {
            val model = if (modelFileName != null) {
                mapPrivateModelFile(context, modelFileName).also {
                    report.put("model_source", "probe-private-file")
                }
            } else {
                val voiceMemoContext = context.createPackageContext(
                    VOICE_MEMO_PACKAGE,
                    Context.CONTEXT_IGNORE_SECURITY,
                )
                mapUncompressedAsset(voiceMemoContext, modelAsset).also {
                    report.put("model_source", "voice-memo-uncompressed-asset")
                }
            }
            report
                .put("model_bytes", model.capacity())
                .put("model_sha256", mappedBufferSha256(model))
                .put("memory_before", memorySnapshot())

            val prefixes = listOf(
                emptyList(),
                listOf(407),
                listOf(407, 5_616),
            )
            if (mode == MODE_FULL_DECODER) {
                val cpuResult = runCpuReference(
                    model = model,
                    prefixes = prefixes,
                )
                report
                    .put("cpu", cpuResult.summary)
                    .put("memory_after_cpu", memorySnapshot())

                System.gc()
                Thread.sleep(GC_SETTLE_MILLIS)

                val gpuResult = runGpuCandidate(
                    model = model,
                    prefixes = prefixes,
                    cpuEncoderState = cpuResult.encoderState,
                    cpuRows = cpuResult.rows,
                )
                report.put("gpu", gpuResult)
            } else {
                val cpuResult = runBoundaryCpuReference(
                    model = model,
                    prefixes = prefixes,
                )
                report
                    .put("cpu", cpuResult.summary)
                    .put("memory_after_cpu", memorySnapshot())

                System.gc()
                Thread.sleep(GC_SETTLE_MILLIS)

                val gpuResult = runBoundaryGpuCandidate(
                    model = model,
                    prefixes = prefixes,
                    encoderState = cpuResult.encoderState,
                    cpuRows = cpuResult.rows,
                )
                report.put("gpu", gpuResult)
            }
            report
                .put("memory_after_gpu", memorySnapshot())
                .put("status", STATUS_COMPLETED)
        } catch (error: Throwable) {
            report
                .put("status", STATUS_FAILED)
                .put("error_type", error.javaClass.name)
                .put("error_message", sanitize(error))
        }

        temporaryReport.writeText(report.toString(2))
        check(temporaryReport.renameTo(reportFile)) {
            "Could not publish Interpreter GPU A/B report."
        }
        Log.i(LOG_TAG, "report=${reportFile.absolutePath} status=${report.optString("status")}")
    }

    private fun runCpuReference(
        model: MappedByteBuffer,
        prefixes: List<List<Int>>,
    ): CpuResult {
        val loadStarted = System.nanoTime()
        val options = Interpreter.Options()
            .setNumThreads(CPU_THREADS)
            .setUseXNNPACK(true)
        return Interpreter(model, options).use { interpreter ->
            val loadMillis = elapsedMillis(loadStarted)
            val executionPlanLength = executionPlanLength(interpreter)
            val encodeStarted = System.nanoTime()
            val encoderState = encode(interpreter, deterministicMel())
            val encodeMillis = elapsedMillis(encodeStarted)
            val rows = mutableListOf<FloatArray>()
            val steps = JSONArray()
            prefixes.forEachIndexed { index, prefix ->
                val started = System.nanoTime()
                val row = decodeActiveRow(interpreter, encoderState, prefix)
                rows += row
                steps.put(
                    rowSummary(
                        step = index,
                        prefix = prefix,
                        row = row,
                        inferenceMillis = elapsedMillis(started),
                    ),
                )
            }
            CpuResult(
                encoderState = encoderState,
                rows = rows,
                summary = JSONObject()
                    .put("model_load_millis", loadMillis)
                    .put("execution_plan_length", executionPlanLength)
                    .put("encode_millis", encodeMillis)
                    .put("encoder_state_sha256", floatSha256(encoderState))
                    .put("teacher_forced_steps", steps),
            )
        }
    }

    private fun runBoundaryCpuReference(
        model: MappedByteBuffer,
        prefixes: List<List<Int>>,
    ): BoundaryCpuResult {
        val loadStarted = System.nanoTime()
        val options = Interpreter.Options()
            .setNumThreads(CPU_THREADS)
            .setUseXNNPACK(true)
        return Interpreter(model, options).use { interpreter ->
            val loadMillis = elapsedMillis(loadStarted)
            val executionPlanLength = executionPlanLength(interpreter)
            val encoderState = deterministicEncoderState()
            val rows = mutableListOf<FloatArray>()
            val steps = JSONArray()
            prefixes.forEachIndexed { index, prefix ->
                val started = System.nanoTime()
                val row = decodeActiveHiddenRow(interpreter, encoderState, prefix)
                rows += row
                steps.put(
                    hiddenRowSummary(
                        step = index,
                        prefix = prefix,
                        row = row,
                        inferenceMillis = elapsedMillis(started),
                    ),
                )
            }
            BoundaryCpuResult(
                encoderState = encoderState,
                rows = rows,
                summary = JSONObject()
                    .put("model_load_millis", loadMillis)
                    .put("execution_plan_length", executionPlanLength)
                    .put("encoder_state_sha256", floatSha256(encoderState))
                    .put("active_hidden_steps", steps)
                    .put(
                        "active_hidden_rows_float32_le_base64",
                        JSONArray(rows.map(::floatRowBase64)),
                    ),
            )
        }
    }

    private fun runGpuCandidate(
        model: MappedByteBuffer,
        prefixes: List<List<Int>>,
        cpuEncoderState: FloatArray,
        cpuRows: List<FloatArray>,
    ): JSONObject {
        val result = JSONObject()
            .put("backend", "standard-interpreter-opencl")
            .put("precision_loss_allowed", false)
            .put("quantized_models_allowed", true)
        try {
            val compatibilitySupported = CompatibilityList().use {
                it.isDelegateSupportedOnThisDevice
            }
            result.put("compatibility_list_supported", compatibilitySupported)

            val delegateOptions = GpuDelegate.Options()
                .setPrecisionLossAllowed(false)
                .setQuantizedModelsAllowed(true)
                .setInferencePreference(
                    GpuDelegate.Options.INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER,
                )
                .setForceBackend(
                    GpuDelegateFactory.Options.GpuBackend.OPENCL,
                )
            val delegate = GpuDelegate(delegateOptions)
            try {
                val interpreterOptions = Interpreter.Options()
                    .setNumThreads(1)
                    .setUseXNNPACK(false)
                    .addDelegate(delegate)
                Log.i(LOG_TAG, "delegate-create-begin backend=opencl")
                val loadStarted = System.nanoTime()
                Interpreter(model, interpreterOptions).use { interpreter ->
                    val loadMillis = elapsedMillis(loadStarted)
                    val delegatedPlanLength = executionPlanLength(interpreter)
                    val encodeStarted = System.nanoTime()
                    val gpuEncoderState = encode(interpreter, deterministicMel())
                    val encoderComparison = compareEncoderStates(
                        cpu = cpuEncoderState,
                        gpu = gpuEncoderState,
                        gpuInferenceMillis = elapsedMillis(encodeStarted),
                    )
                    val steps = JSONArray()
                    prefixes.forEachIndexed { index, prefix ->
                        val started = System.nanoTime()
                        val row = decodeActiveRow(interpreter, gpuEncoderState, prefix)
                        steps.put(
                            compareRows(
                                step = index,
                                prefix = prefix,
                                cpu = cpuRows[index],
                                gpu = row,
                                gpuInferenceMillis = elapsedMillis(started),
                            ),
                        )
                    }
                    result
                        .put("model_load_millis", loadMillis)
                        .put("delegated_execution_plan_length", delegatedPlanLength)
                        .put("encoder_state", encoderComparison)
                        .put("teacher_forced_encoder_source", "gpu")
                        .put(
                            "full_decode_delegation_indicated",
                            delegatedPlanLength == FULLY_DELEGATED_PLAN_LENGTH,
                        )
                        .put("teacher_forced_steps", steps)
                        .put("status", STATUS_COMPLETED)
                    Log.i(
                        LOG_TAG,
                        "delegate-run-end planLength=$delegatedPlanLength",
                    )
                }
            } finally {
                delegate.close()
            }
        } catch (error: Throwable) {
            result
                .put("status", STATUS_FAILED)
                .put("error_type", error.javaClass.name)
                .put("error_message", sanitize(error))
            Log.e(LOG_TAG, "delegate-run-failed", error)
        }
        return result
    }

    private fun runBoundaryGpuCandidate(
        model: MappedByteBuffer,
        prefixes: List<List<Int>>,
        encoderState: FloatArray,
        cpuRows: List<FloatArray>,
    ): JSONObject {
        val result = JSONObject()
            .put("backend", "standard-interpreter-opencl")
            .put("precision_loss_allowed", false)
            .put("quantized_models_allowed", true)
        try {
            val compatibilitySupported = CompatibilityList().use {
                it.isDelegateSupportedOnThisDevice
            }
            result.put("compatibility_list_supported", compatibilitySupported)

            val delegateOptions = GpuDelegate.Options()
                .setPrecisionLossAllowed(false)
                .setQuantizedModelsAllowed(true)
                .setInferencePreference(
                    GpuDelegate.Options.INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER,
                )
                .setForceBackend(
                    GpuDelegateFactory.Options.GpuBackend.OPENCL,
                )
            val delegate = GpuDelegate(delegateOptions)
            try {
                val interpreterOptions = Interpreter.Options()
                    .setNumThreads(1)
                    .setUseXNNPACK(false)
                    .addDelegate(delegate)
                Log.i(LOG_TAG, "boundary-delegate-create-begin backend=opencl")
                val loadStarted = System.nanoTime()
                Interpreter(model, interpreterOptions).use { interpreter ->
                    val loadMillis = elapsedMillis(loadStarted)
                    val delegatedPlanLength = executionPlanLength(interpreter)
                    val steps = JSONArray()
                    prefixes.forEachIndexed { index, prefix ->
                        val started = System.nanoTime()
                        val row = decodeActiveHiddenRow(
                            interpreter,
                            encoderState,
                            prefix,
                        )
                        steps.put(
                            compareHiddenRows(
                                step = index,
                                prefix = prefix,
                                cpu = cpuRows[index],
                                gpu = row,
                                gpuInferenceMillis = elapsedMillis(started),
                            ),
                        )
                    }
                    result
                        .put("model_load_millis", loadMillis)
                        .put("delegated_execution_plan_length", delegatedPlanLength)
                        .put(
                            "full_decode_delegation_indicated",
                            delegatedPlanLength == FULLY_DELEGATED_PLAN_LENGTH,
                        )
                        .put("active_hidden_steps", steps)
                        .put("status", STATUS_COMPLETED)
                    Log.i(
                        LOG_TAG,
                        "boundary-delegate-run-end planLength=$delegatedPlanLength",
                    )
                }
            } finally {
                delegate.close()
            }
        } catch (error: Throwable) {
            result
                .put("status", STATUS_FAILED)
                .put("error_type", error.javaClass.name)
                .put("error_message", sanitize(error))
            Log.e(LOG_TAG, "boundary-delegate-run-failed", error)
        }
        return result
    }

    private fun encode(
        interpreter: Interpreter,
        mel: FloatArray,
    ): FloatArray {
        require(mel.size == MEL_FLOATS)
        val input = directFloatBuffer(MEL_FLOATS).also { buffer ->
            buffer.asFloatBuffer().put(mel)
            buffer.rewind()
        }
        val output = directFloatBuffer(ENCODER_STATE_FLOATS)
        interpreter.runSignature(
            mapOf(ENCODE_INPUT to input),
            mapOf(SIGNATURE_OUTPUT to output),
            ENCODE_SIGNATURE,
        )
        output.rewind()
        return FloatArray(ENCODER_STATE_FLOATS).also { values ->
            output.asFloatBuffer().get(values)
        }
    }

    private fun decodeActiveRow(
        interpreter: Interpreter,
        encoderState: FloatArray,
        prefix: List<Int>,
    ): FloatArray {
        val prompt = intArrayOf(
            DECODER_START_TOKEN,
            RUSSIAN_LANGUAGE_TOKEN,
            TRANSCRIBE_TOKEN,
            NO_TIMESTAMPS_TOKEN,
            *prefix.toIntArray(),
        )
        require(prompt.size <= TARGET_LENGTH)
        val paddedTokens = IntArray(TARGET_LENGTH) { index ->
            if (index < prompt.size) prompt[index] else EOS_TOKEN
        }
        val encoder = directFloatBuffer(ENCODER_STATE_FLOATS).also { buffer ->
            buffer.asFloatBuffer().put(encoderState)
            buffer.rewind()
        }
        val tokens = directIntBuffer(TARGET_LENGTH).also { buffer ->
            buffer.asIntBuffer().put(paddedTokens)
            buffer.rewind()
        }
        val mask = directFloatBuffer(MASK_FLOATS).also { buffer ->
            val floats = buffer.asFloatBuffer()
            repeat(TARGET_LENGTH) { row ->
                repeat(TARGET_LENGTH) { column ->
                    floats.put(if (column <= row) 0.0f else MASKED_LOGIT)
                }
            }
            buffer.rewind()
        }
        val output = directFloatBuffer(DECODER_OUTPUT_FLOATS)
        interpreter.runSignature(
            mapOf(
                DECODE_ENCODER_STATE to encoder,
                DECODE_TOKENS to tokens,
                DECODE_MASK to mask,
            ),
            mapOf(SIGNATURE_OUTPUT to output),
            DECODE_SIGNATURE,
        )
        output.rewind()
        val activeOffset = (prompt.size - 1) * VOCABULARY_SIZE
        output.position(activeOffset * Float.SIZE_BYTES)
        return FloatArray(VOCABULARY_SIZE).also { row ->
            output.slice().order(ByteOrder.nativeOrder()).asFloatBuffer().get(row)
        }
    }

    private fun decodeActiveHiddenRow(
        interpreter: Interpreter,
        encoderState: FloatArray,
        prefix: List<Int>,
    ): FloatArray {
        val prompt = intArrayOf(
            DECODER_START_TOKEN,
            RUSSIAN_LANGUAGE_TOKEN,
            TRANSCRIBE_TOKEN,
            NO_TIMESTAMPS_TOKEN,
            *prefix.toIntArray(),
        )
        require(prompt.size <= TARGET_LENGTH)
        val paddedTokens = IntArray(TARGET_LENGTH) { index ->
            if (index < prompt.size) prompt[index] else EOS_TOKEN
        }
        val encoder = directFloatBuffer(ENCODER_STATE_FLOATS).also { buffer ->
            buffer.asFloatBuffer().put(encoderState)
            buffer.rewind()
        }
        val tokens = directIntBuffer(TARGET_LENGTH).also { buffer ->
            buffer.asIntBuffer().put(paddedTokens)
            buffer.rewind()
        }
        val mask = directFloatBuffer(MASK_FLOATS).also { buffer ->
            val floats = buffer.asFloatBuffer()
            repeat(TARGET_LENGTH) { row ->
                repeat(TARGET_LENGTH) { column ->
                    floats.put(if (column <= row) 0.0f else MASKED_LOGIT)
                }
            }
            buffer.rewind()
        }
        val output = directFloatBuffer(BOUNDARY_OUTPUT_FLOATS)
        interpreter.runSignature(
            mapOf(
                DECODE_ENCODER_STATE to encoder,
                DECODE_TOKENS to tokens,
                DECODE_MASK to mask,
            ),
            mapOf(SIGNATURE_OUTPUT to output),
            DECODE_SIGNATURE,
        )
        output.rewind()
        val activeOffset = (prompt.size - 1) * ENCODER_WIDTH
        output.position(activeOffset * Float.SIZE_BYTES)
        return FloatArray(ENCODER_WIDTH).also { row ->
            output.slice().order(ByteOrder.nativeOrder()).asFloatBuffer().get(row)
        }
    }

    private fun deterministicMel(): FloatArray =
        FloatArray(MEL_FLOATS) { index ->
            val mel = index / MEL_FRAMES
            val frame = index % MEL_FRAMES
            val slow = sin(frame * 0.017 + mel * 0.071)
            val speechLike = cos(frame * 0.043 - mel * 0.113)
            (-0.85 + 0.31 * slow + 0.19 * speechLike).toFloat()
        }

    private fun deterministicEncoderState(): FloatArray =
        FloatArray(ENCODER_STATE_FLOATS) { index ->
            val frame = index / ENCODER_WIDTH
            val channel = index % ENCODER_WIDTH
            val slow = sin(frame * 0.013 + channel * 0.007)
            val cross = cos(frame * 0.031 - channel * 0.011)
            (0.17 * slow + 0.11 * cross).toFloat()
        }

    private fun rowSummary(
        step: Int,
        prefix: List<Int>,
        row: FloatArray,
        inferenceMillis: Long,
    ): JSONObject {
        val top = topTwo(row)
        return JSONObject()
            .put("step", step)
            .put("prefix", JSONArray(prefix))
            .put("inference_millis", inferenceMillis)
            .put("logits_sha256", floatSha256(row))
            .put("top1_token_id", top.first.token)
            .put("top1_logit", top.first.score.toDouble())
            .put("top2_token_id", top.second.token)
            .put("top2_logit", top.second.score.toDouble())
            .put("top1_margin", (top.first.score - top.second.score).toDouble())
    }

    private fun compareRows(
        step: Int,
        prefix: List<Int>,
        cpu: FloatArray,
        gpu: FloatArray,
        gpuInferenceMillis: Long,
    ): JSONObject {
        require(cpu.size == gpu.size)
        var exact = 0
        var maximumAbsoluteError = 0.0
        var absoluteErrorSum = 0.0
        var squaredErrorSum = 0.0
        var cpuSquaredSum = 0.0
        var gpuSquaredSum = 0.0
        var dot = 0.0
        cpu.indices.forEach { index ->
            val expected = cpu[index]
            val actual = gpu[index]
            if (expected.toRawBits() == actual.toRawBits()) exact++
            if (!expected.isFinite() || !actual.isFinite()) return@forEach
            val difference = actual.toDouble() - expected.toDouble()
            val absoluteError = abs(difference)
            maximumAbsoluteError = maxOf(maximumAbsoluteError, absoluteError)
            absoluteErrorSum += absoluteError
            squaredErrorSum += difference * difference
            cpuSquaredSum += expected.toDouble() * expected
            gpuSquaredSum += actual.toDouble() * actual
            dot += expected.toDouble() * actual
        }
        val cpuTop = topTwo(cpu)
        val gpuTop = topTwo(gpu)
        val divisor = cpu.size.toDouble()
        val cosineDenominator = sqrt(cpuSquaredSum * gpuSquaredSum)
        return JSONObject()
            .put("step", step)
            .put("prefix", JSONArray(prefix))
            .put("gpu_inference_millis", gpuInferenceMillis)
            .put("cpu_logits_sha256", floatSha256(cpu))
            .put("gpu_logits_sha256", floatSha256(gpu))
            .put("exact_logits_match", exact == cpu.size)
            .put("exact_element_count", exact)
            .put("element_count", cpu.size)
            .put("cpu_top1_token_id", cpuTop.first.token)
            .put("gpu_top1_token_id", gpuTop.first.token)
            .put("top1_match", cpuTop.first.token == gpuTop.first.token)
            .put("cpu_top1_logit", cpuTop.first.score.toDouble())
            .put("gpu_top1_logit", gpuTop.first.score.toDouble())
            .put("cpu_top1_margin", (cpuTop.first.score - cpuTop.second.score).toDouble())
            .put("gpu_top1_margin", (gpuTop.first.score - gpuTop.second.score).toDouble())
            .put("max_absolute_error", maximumAbsoluteError)
            .put("mean_absolute_error", absoluteErrorSum / divisor)
            .put("root_mean_square_error", sqrt(squaredErrorSum / divisor))
            .put(
                "cosine_similarity",
                if (cosineDenominator == 0.0) JSONObject.NULL else dot / cosineDenominator,
            )
    }

    private fun compareEncoderStates(
        cpu: FloatArray,
        gpu: FloatArray,
        gpuInferenceMillis: Long,
    ): JSONObject {
        require(cpu.size == gpu.size)
        var exact = 0
        var maximumAbsoluteError = 0.0
        var absoluteErrorSum = 0.0
        var squaredErrorSum = 0.0
        var cpuSquaredSum = 0.0
        var gpuSquaredSum = 0.0
        var dot = 0.0
        cpu.indices.forEach { index ->
            val expected = cpu[index]
            val actual = gpu[index]
            if (expected.toRawBits() == actual.toRawBits()) exact++
            if (!expected.isFinite() || !actual.isFinite()) return@forEach
            val difference = actual.toDouble() - expected.toDouble()
            val absoluteError = abs(difference)
            maximumAbsoluteError = maxOf(maximumAbsoluteError, absoluteError)
            absoluteErrorSum += absoluteError
            squaredErrorSum += difference * difference
            cpuSquaredSum += expected.toDouble() * expected
            gpuSquaredSum += actual.toDouble() * actual
            dot += expected.toDouble() * actual
        }
        val divisor = cpu.size.toDouble()
        val cosineDenominator = sqrt(cpuSquaredSum * gpuSquaredSum)
        return JSONObject()
            .put("gpu_inference_millis", gpuInferenceMillis)
            .put("cpu_sha256", floatSha256(cpu))
            .put("gpu_sha256", floatSha256(gpu))
            .put("exact_match", exact == cpu.size)
            .put("exact_element_count", exact)
            .put("element_count", cpu.size)
            .put("max_absolute_error", maximumAbsoluteError)
            .put("mean_absolute_error", absoluteErrorSum / divisor)
            .put("root_mean_square_error", sqrt(squaredErrorSum / divisor))
            .put(
                "cosine_similarity",
                if (cosineDenominator == 0.0) JSONObject.NULL else dot / cosineDenominator,
            )
    }

    private fun hiddenRowSummary(
        step: Int,
        prefix: List<Int>,
        row: FloatArray,
        inferenceMillis: Long,
    ): JSONObject {
        val top = topTwo(row)
        return JSONObject()
            .put("step", step)
            .put("prefix", JSONArray(prefix))
            .put("inference_millis", inferenceMillis)
            .put("hidden_sha256", floatSha256(row))
            .put("argmax_index", top.first.token)
            .put("argmax_value", top.first.score.toDouble())
            .put("second_index", top.second.token)
            .put("second_value", top.second.score.toDouble())
            .put("root_mean_square_value", vectorRootMeanSquare(row))
    }

    private fun compareHiddenRows(
        step: Int,
        prefix: List<Int>,
        cpu: FloatArray,
        gpu: FloatArray,
        gpuInferenceMillis: Long,
    ): JSONObject {
        require(cpu.size == gpu.size)
        var exact = 0
        var maximumAbsoluteError = 0.0
        var absoluteErrorSum = 0.0
        var squaredErrorSum = 0.0
        var cpuSquaredSum = 0.0
        var gpuSquaredSum = 0.0
        var dot = 0.0
        cpu.indices.forEach { index ->
            val expected = cpu[index]
            val actual = gpu[index]
            if (expected.toRawBits() == actual.toRawBits()) exact++
            if (!expected.isFinite() || !actual.isFinite()) return@forEach
            val difference = actual.toDouble() - expected.toDouble()
            val absoluteError = abs(difference)
            maximumAbsoluteError = maxOf(maximumAbsoluteError, absoluteError)
            absoluteErrorSum += absoluteError
            squaredErrorSum += difference * difference
            cpuSquaredSum += expected.toDouble() * expected
            gpuSquaredSum += actual.toDouble() * actual
            dot += expected.toDouble() * actual
        }
        val cpuTop = topTwo(cpu)
        val gpuTop = topTwo(gpu)
        val divisor = cpu.size.toDouble()
        val cosineDenominator = sqrt(cpuSquaredSum * gpuSquaredSum)
        return JSONObject()
            .put("step", step)
            .put("prefix", JSONArray(prefix))
            .put("gpu_inference_millis", gpuInferenceMillis)
            .put("cpu_hidden_sha256", floatSha256(cpu))
            .put("gpu_hidden_sha256", floatSha256(gpu))
            .put("exact_hidden_match", exact == cpu.size)
            .put("exact_element_count", exact)
            .put("element_count", cpu.size)
            .put("cpu_argmax_index", cpuTop.first.token)
            .put("gpu_argmax_index", gpuTop.first.token)
            .put("argmax_match", cpuTop.first.token == gpuTop.first.token)
            .put("cpu_argmax_value", cpuTop.first.score.toDouble())
            .put("gpu_argmax_value", gpuTop.first.score.toDouble())
            .put("cpu_root_mean_square_value", sqrt(cpuSquaredSum / divisor))
            .put("gpu_root_mean_square_value", sqrt(gpuSquaredSum / divisor))
            .put("max_absolute_error", maximumAbsoluteError)
            .put("mean_absolute_error", absoluteErrorSum / divisor)
            .put("root_mean_square_error", sqrt(squaredErrorSum / divisor))
            .put(
                "cosine_similarity",
                if (cosineDenominator == 0.0) JSONObject.NULL else dot / cosineDenominator,
            )
    }

    private fun vectorRootMeanSquare(values: FloatArray): Double =
        sqrt(
            values.sumOf { value ->
                value.toDouble() * value.toDouble()
            } / values.size.toDouble(),
        )

    private fun topTwo(values: FloatArray): Pair<RankedValue, RankedValue> {
        var first = RankedValue(-1, Float.NEGATIVE_INFINITY)
        var second = RankedValue(-1, Float.NEGATIVE_INFINITY)
        values.forEachIndexed { token, score ->
            if (!score.isFinite()) return@forEachIndexed
            if (score > first.score) {
                second = first
                first = RankedValue(token, score)
            } else if (score > second.score) {
                second = RankedValue(token, score)
            }
        }
        check(first.token >= 0 && second.token >= 0) {
            "Decoder logits did not contain two finite values."
        }
        return first to second
    }

    private fun executionPlanLength(interpreter: Interpreter): Int? = runCatching {
        var type: Class<*>? = interpreter.javaClass
        var method: java.lang.reflect.Method? = null
        while (type != null && method == null) {
            method = type.declaredMethods.singleOrNull {
                it.name == "getExecutionPlanLength" && it.parameterCount == 0
            }
            type = type.superclass
        }
        requireNotNull(method).run {
            isAccessible = true
            invoke(interpreter) as Int
        }
    }.getOrNull()

    private fun mapUncompressedAsset(
        context: Context,
        assetPath: String,
    ): MappedByteBuffer {
        val descriptor = try {
            context.assets.openFd(assetPath)
        } catch (error: Throwable) {
            throw IllegalStateException(
                "Voice Memo asset $assetPath must exist and be uncompressed.",
                error,
            )
        }
        return descriptor.use { asset ->
            FileInputStream(asset.fileDescriptor).use { input ->
                input.channel.map(
                    FileChannel.MapMode.READ_ONLY,
                    asset.startOffset,
                    asset.declaredLength,
                )
            }
        }
    }

    private fun mapPrivateModelFile(
        context: Context,
        fileName: String,
    ): MappedByteBuffer {
        require(
            fileName.length <= 96 &&
                fileName.all { it.isLetterOrDigit() || it == '-' || it == '_' || it == '.' },
        ) {
            "$ARG_MODEL_FILE must be a simple file name."
        }
        val model = File(context.filesDir, fileName)
        require(model.isFile) { "Probe-private model file is missing: $fileName" }
        return FileInputStream(model).use { input ->
            input.channel.map(
                FileChannel.MapMode.READ_ONLY,
                0,
                input.channel.size(),
            )
        }
    }

    private fun mappedBufferSha256(buffer: MappedByteBuffer): String {
        val digest = MessageDigest.getInstance("SHA-256")
        val copy = buffer.asReadOnlyBuffer()
        copy.rewind()
        val chunk = ByteArray(1024 * 1024)
        while (copy.hasRemaining()) {
            val count = minOf(copy.remaining(), chunk.size)
            copy.get(chunk, 0, count)
            digest.update(chunk, 0, count)
        }
        return digest.digest().toHex()
    }

    private fun floatSha256(values: FloatArray): String {
        val digest = MessageDigest.getInstance("SHA-256")
        values.forEach { value ->
            val bits = value.toRawBits()
            digest.update((bits ushr 24).toByte())
            digest.update((bits ushr 16).toByte())
            digest.update((bits ushr 8).toByte())
            digest.update(bits.toByte())
        }
        return digest.digest().toHex()
    }

    private fun floatRowBase64(values: FloatArray): String {
        val bytes = ByteBuffer.allocate(values.size * Float.SIZE_BYTES)
            .order(ByteOrder.LITTLE_ENDIAN)
        values.forEach(bytes::putFloat)
        return Base64.encodeToString(bytes.array(), Base64.NO_WRAP)
    }

    private fun ByteArray.toHex(): String =
        joinToString(separator = "") { byte -> "%02x".format(byte) }

    private fun memorySnapshot(): JSONObject {
        val info = Debug.MemoryInfo()
        Debug.getMemoryInfo(info)
        return JSONObject()
            .put("total_pss_kib", info.totalPss)
            .put("total_private_dirty_kib", info.totalPrivateDirty)
            .put("native_heap_bytes", Debug.getNativeHeapAllocatedSize())
    }

    private fun directFloatBuffer(floatCount: Int): ByteBuffer =
        ByteBuffer.allocateDirect(floatCount * Float.SIZE_BYTES)
            .order(ByteOrder.nativeOrder())

    private fun directIntBuffer(intCount: Int): ByteBuffer =
        ByteBuffer.allocateDirect(intCount * Int.SIZE_BYTES)
            .order(ByteOrder.nativeOrder())

    private fun requiredArgument(value: String?, name: String): String {
        val result = requireNotNull(value?.takeIf(String::isNotBlank)) {
            "Missing instrumentation argument $name."
        }
        require(result.length <= 96 && result.all { it.isLetterOrDigit() || it == '-' }) {
            "$name must contain only ASCII letters, digits, and hyphens."
        }
        return result
    }

    private fun elapsedMillis(startedNanos: Long): Long =
        (System.nanoTime() - startedNanos) / 1_000_000L

    private fun sanitize(error: Throwable): String =
        generateSequence(error) { it.cause }
            .mapNotNull { it.message?.lineSequence()?.firstOrNull() }
            .joinToString(" caused by ")
            .take(2_000)
            .ifBlank { error.javaClass.simpleName }

    private data class CpuResult(
        val encoderState: FloatArray,
        val rows: List<FloatArray>,
        val summary: JSONObject,
    )

    private data class BoundaryCpuResult(
        val encoderState: FloatArray,
        val rows: List<FloatArray>,
        val summary: JSONObject,
    )

    private data class RankedValue(
        val token: Int,
        val score: Float,
    )

    companion object {
        private const val ARG_RUN_ID = "runId"
        private const val ARG_MODEL_ASSET = "modelAsset"
        private const val ARG_MODEL_FILE = "modelFile"
        private const val ARG_MODE = "mode"
        private const val MODE_FULL_DECODER = "full-decoder"
        private const val MODE_BOUNDARY_AFTER_TWO = "boundary-after-two"
        private const val VOICE_MEMO_PACKAGE = "com.example.voicememo"
        private const val DEFAULT_MODEL_ASSET =
            "whisper/whisper_large_v3_turbo_30s_i8_explicit_key_transpose.tflite"
        private const val REPORT_DIRECTORY = "interpreter-gpu-ab"
        private const val LOG_TAG = "VoiceMemoInterpGpu"
        private const val STATUS_COMPLETED = "completed"
        private const val STATUS_FAILED = "failed"
        private const val CPU_THREADS = 3
        private const val GC_SETTLE_MILLIS = 500L
        private const val FULLY_DELEGATED_PLAN_LENGTH = 1

        private const val ENCODE_SIGNATURE = "encode"
        private const val ENCODE_INPUT = "args_0"
        private const val DECODE_SIGNATURE = "decode"
        private const val DECODE_ENCODER_STATE = "args_0"
        private const val DECODE_TOKENS = "args_1"
        private const val DECODE_MASK = "args_2"
        private const val SIGNATURE_OUTPUT = "output_0"

        private const val MEL_BINS = 128
        private const val MEL_FRAMES = 3_000
        private const val MEL_FLOATS = MEL_BINS * MEL_FRAMES
        private const val ENCODER_FRAMES = 1_500
        private const val ENCODER_WIDTH = 1_280
        private const val ENCODER_STATE_FLOATS = ENCODER_FRAMES * ENCODER_WIDTH
        private const val TARGET_LENGTH = 128
        private const val VOCABULARY_SIZE = 51_866
        private const val MASK_FLOATS = TARGET_LENGTH * TARGET_LENGTH
        private const val DECODER_OUTPUT_FLOATS = TARGET_LENGTH * VOCABULARY_SIZE
        private const val BOUNDARY_OUTPUT_FLOATS = TARGET_LENGTH * ENCODER_WIDTH
        private const val MASKED_LOGIT = -1.0e9f

        private const val EOS_TOKEN = 50_257
        private const val DECODER_START_TOKEN = 50_258
        private const val RUSSIAN_LANGUAGE_TOKEN = 50_263
        private const val TRANSCRIBE_TOKEN = 50_360
        private const val NO_TIMESTAMPS_TOKEN = 50_364
    }
}
