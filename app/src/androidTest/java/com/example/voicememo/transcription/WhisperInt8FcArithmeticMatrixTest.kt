package com.example.voicememo.transcription

import android.content.res.AssetManager
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.abs
import kotlin.math.sqrt
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.Interpreter

/**
 * Separates scale-vector, zero-point, and accumulation failures in LiteRT's
 * source-quantized GPU FULLY_CONNECTED implementation.
 */
@RunWith(AndroidJUnit4::class)
class WhisperInt8FcArithmeticMatrixTest {
    @Test
    fun compareControlledWeightAndInputVariants() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val runId = requireSafe(
            InstrumentationRegistry.getArguments().getString(ARG_RUN_ID),
        )
        val assets = instrumentation.context.assets
        val outputDirectory =
            File(instrumentation.targetContext.cacheDir, REPORT_DIRECTORY).apply {
                check(isDirectory || mkdirs())
            }
        val realInput = readFloats(assets, "$ASSET_ROOT/real-step-7-input.f32")
        check(realInput.size == ELEMENTS)
        val cases = inputCases(realInput)
        val report = JSONObject()
            .put("run_id", runId)
            .put("active_row", ACTIVE_ROW)
            .put("shape", JSONArray(listOf(1, ROWS, WIDTH)))
            .put("input_cases", JSONArray(cases.map(InputCase::name)))
            .put("models", JSONArray())

        try {
            MODELS.forEach { modelSpec ->
                report.getJSONArray("models").put(
                    runModel(
                        assets = assets,
                        modelSpec = modelSpec,
                        cases = cases,
                    ),
                )
            }
            report.put("status", "completed")
        } catch (error: Throwable) {
            report
                .put("status", "failed")
                .put("error_type", error.javaClass.name)
                .put("error_message", error.message ?: error.toString())
            throw error
        } finally {
            val destination = File(outputDirectory, "arithmetic-$runId.json")
            val temporary = File(outputDirectory, "${destination.name}.tmp")
            temporary.writeText(report.toString(2))
            check(temporary.renameTo(destination))
        }
    }

    private fun runModel(
        assets: AssetManager,
        modelSpec: ModelSpec,
        cases: List<InputCase>,
    ): JSONObject {
        val modelAsset = "$ASSET_ROOT/${modelSpec.name}.tflite"
        val metadata = if (modelSpec.quantized) {
            WeightMetadata(
                scales = expandScales(
                    readFloats(assets, "$ASSET_ROOT/${modelSpec.name}.scales.f32"),
                ),
                rowSums =
                    readInts(assets, "$ASSET_ROOT/${modelSpec.name}.sums.i32"),
            )
        } else {
            null
        }
        val result = JSONObject()
            .put("name", modelSpec.name)
            .put("quantized", modelSpec.quantized)
            .put("cases", JSONArray())

        val cpuLoadStarted = System.nanoTime()
        val cpu = CpuRunner.open(assets, modelAsset)
        result.put("cpu_load_millis", elapsedMillis(cpuLoadStarted))
        val direct = if (modelSpec.quantized) {
            val started = System.nanoTime()
            GpuRunner.open(
                assets = assets,
                modelAsset = modelAsset,
                allowSourceQuantization = false,
            ).also {
                result.put("direct_gpu_load_millis", elapsedMillis(started))
            }
        } else {
            null
        }
        val sourceStarted = System.nanoTime()
        val source = GpuRunner.open(
            assets = assets,
            modelAsset = modelAsset,
            allowSourceQuantization = true,
        )
        result.put("source_gpu_load_millis", elapsedMillis(sourceStarted))

        try {
            cases.forEach { inputCase ->
                val cpuStarted = System.nanoTime()
                val cpuOutput = cpu.run(inputCase.values)
                val cpuMillis = elapsedMillis(cpuStarted)
                val activeCpu = activeRow(cpuOutput)
                val oracle = analyticOutput(modelSpec.name, inputCase.values)

                val caseResult = JSONObject()
                    .put("name", inputCase.name)
                    .put("cpu_inference_millis", cpuMillis)
                    .apply {
                        if (oracle != null) {
                            put("cpu_vs_oracle", compare(oracle, activeCpu, null))
                        }
                    }
                if (direct != null) {
                    val started = System.nanoTime()
                    val output = direct.run(inputCase.values)
                    val activeOutput = activeRow(output)
                    caseResult
                        .put("direct_gpu_inference_millis", elapsedMillis(started))
                        .put(
                            "direct_gpu",
                            compare(activeCpu, activeOutput, metadata),
                        )
                    if (oracle != null) {
                        caseResult.put(
                            "direct_gpu_vs_oracle",
                            compare(oracle, activeOutput, null),
                        )
                    }
                }
                val sourceRunStarted = System.nanoTime()
                val sourceOutput = source.run(inputCase.values)
                val activeSourceOutput = activeRow(sourceOutput)
                caseResult
                    .put(
                        "source_gpu_inference_millis",
                        elapsedMillis(sourceRunStarted),
                    )
                    .put(
                        "source_gpu",
                        compare(activeCpu, activeSourceOutput, metadata),
                    )
                if (oracle != null) {
                    caseResult.put(
                        "source_gpu_vs_oracle",
                        compare(oracle, activeSourceOutput, null),
                    )
                }
                result.getJSONArray("cases").put(caseResult)
            }
        } finally {
            source.close()
            direct?.close()
            cpu.close()
        }
        return result
    }

    private fun compare(
        expected: FloatArray,
        actual: FloatArray,
        metadata: WeightMetadata?,
    ): JSONObject {
        require(expected.size == WIDTH && actual.size == WIDTH)
        val errors = DoubleArray(WIDTH)
        var maxError = 0.0
        var absoluteErrorSum = 0.0
        var squaredErrorSum = 0.0
        var expectedSquaredSum = 0.0
        var actualSquaredSum = 0.0
        var dot = 0.0
        expected.indices.forEach { index ->
            val expectedValue = expected[index].toDouble()
            val actualValue = actual[index].toDouble()
            val error = actualValue - expectedValue
            errors[index] = error
            maxError = maxOf(maxError, abs(error))
            absoluteErrorSum += abs(error)
            squaredErrorSum += error * error
            expectedSquaredSum += expectedValue * expectedValue
            actualSquaredSum += actualValue * actualValue
            dot += expectedValue * actualValue
        }
        val denominator = sqrt(expectedSquaredSum * actualSquaredSum)
        return JSONObject()
            .put("maximum_absolute_error", maxError)
            .put("mean_absolute_error", absoluteErrorSum / WIDTH)
            .put("root_mean_square_error", sqrt(squaredErrorSum / WIDTH))
            .put(
                "cosine_similarity",
                if (denominator == 0.0) JSONObject.NULL else dot / denominator,
            )
            .apply {
                if (metadata != null) {
                    val absoluteErrors = DoubleArray(WIDTH) { abs(errors[it]) }
                    put(
                        "absolute_error_scale_correlation",
                        finiteOrNull(
                            pearson(
                                absoluteErrors,
                                DoubleArray(WIDTH) { metadata.scales[it].toDouble() },
                            ),
                        ),
                    )
                    put(
                        "absolute_error_weight_sum_correlation",
                        finiteOrNull(
                            pearson(
                                absoluteErrors,
                                DoubleArray(WIDTH) {
                                    abs(metadata.rowSums[it].toDouble())
                                },
                            ),
                        ),
                    )
                    put(
                        "signed_error_weight_sum_correlation",
                        finiteOrNull(
                            pearson(
                                errors,
                                DoubleArray(WIDTH) {
                                    metadata.rowSums[it].toDouble()
                                },
                            ),
                        ),
                    )
                    put(
                        "signed_error_by_weight_sum",
                        linearRegression(
                            independent = DoubleArray(WIDTH) {
                                metadata.rowSums[it].toDouble()
                            },
                            dependent = errors,
                        ),
                    )
                    put(
                        "absolute_error_by_scale",
                        linearRegression(
                            independent = DoubleArray(WIDTH) {
                                metadata.scales[it].toDouble()
                            },
                            dependent = absoluteErrors,
                        ),
                    )
                    put(
                        "absolute_error_channel_correlation",
                        finiteOrNull(
                            pearson(
                                absoluteErrors,
                                DoubleArray(WIDTH) { it.toDouble() },
                            ),
                        ),
                    )
                }
            }
    }

    private fun inputCases(real: FloatArray): List<InputCase> {
        fun active(generator: (Int) -> Float): FloatArray =
            FloatArray(ELEMENTS).also { values ->
                val start = ACTIVE_ROW * WIDTH
                repeat(WIDTH) { channel ->
                    values[start + channel] = generator(channel)
                }
            }

        return listOf(
            InputCase("real-step-7", real),
            InputCase("quiet-real-step-7", FloatArray(ELEMENTS) { real[it] * 0.01f }),
            InputCase("zero", FloatArray(ELEMENTS)),
            InputCase("constant-positive", active { 0.25f }),
            InputCase("constant-negative", active { -0.25f }),
            InputCase("impulse", active { if (it == 0) 1f else 0f }),
            InputCase("alternating", active { if (it % 2 == 0) 0.25f else -0.25f }),
            InputCase("ramp", active { -1f + 2f * it / (WIDTH - 1) }),
        )
    }

    private fun analyticOutput(
        modelName: String,
        input: FloatArray,
    ): FloatArray? {
        val rowStart = ACTIVE_ROW * WIDTH
        return when (modelName) {
            "synthetic-diagonal-scale-ramp" ->
                FloatArray(WIDTH) { channel ->
                    val scale = 0.001f + channel * 0.000001f
                    input[rowStart + channel] * 127f * scale
                }
            "synthetic-accumulation-ladder" -> {
                val prefix = DoubleArray(WIDTH + 1)
                repeat(WIDTH) { channel ->
                    prefix[channel + 1] =
                        prefix[channel] + input[rowStart + channel]
                }
                val widths =
                    intArrayOf(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1280)
                FloatArray(WIDTH) { outputChannel ->
                    val count = widths[outputChannel % widths.size]
                    (prefix[count] * 0.01).toFloat()
                }
            }
            else -> null
        }
    }

    private fun activeRow(values: FloatArray): FloatArray {
        val start = ACTIVE_ROW * WIDTH
        return values.copyOfRange(start, start + WIDTH)
    }

    private fun expandScales(values: FloatArray): FloatArray = when (values.size) {
        1 -> FloatArray(WIDTH) { values.single() }
        WIDTH -> values
        else -> error("Unexpected scale count ${values.size}.")
    }

    private fun pearson(left: DoubleArray, right: DoubleArray): Double {
        require(left.size == right.size && left.isNotEmpty())
        val leftMean = left.average()
        val rightMean = right.average()
        var numerator = 0.0
        var leftSquared = 0.0
        var rightSquared = 0.0
        left.indices.forEach { index ->
            val leftDelta = left[index] - leftMean
            val rightDelta = right[index] - rightMean
            numerator += leftDelta * rightDelta
            leftSquared += leftDelta * leftDelta
            rightSquared += rightDelta * rightDelta
        }
        val denominator = sqrt(leftSquared * rightSquared)
        return if (denominator == 0.0) Double.NaN else numerator / denominator
    }

    private fun linearRegression(
        independent: DoubleArray,
        dependent: DoubleArray,
    ): Any {
        require(independent.size == dependent.size && independent.isNotEmpty())
        val independentMean = independent.average()
        val dependentMean = dependent.average()
        var covariance = 0.0
        var independentVariance = 0.0
        independent.indices.forEach { index ->
            val delta = independent[index] - independentMean
            covariance += delta * (dependent[index] - dependentMean)
            independentVariance += delta * delta
        }
        if (independentVariance == 0.0) return JSONObject.NULL
        val slope = covariance / independentVariance
        return JSONObject()
            .put("slope", slope)
            .put("intercept", dependentMean - slope * independentMean)
    }

    private fun finiteOrNull(value: Double): Any =
        if (value.isFinite()) value else JSONObject.NULL

    private fun elapsedMillis(started: Long): Long =
        (System.nanoTime() - started) / 1_000_000L

    private fun requireSafe(value: String?): String {
        val result = value.orEmpty()
        require(result.matches(Regex("[A-Za-z0-9-]+"))) {
            "$ARG_RUN_ID is missing or invalid."
        }
        return result
    }

    private data class InputCase(val name: String, val values: FloatArray)

    private data class ModelSpec(val name: String, val quantized: Boolean)

    private data class WeightMetadata(
        val scales: FloatArray,
        val rowSums: IntArray,
    )

    companion object {
        private const val ARG_RUN_ID = "runId"
        private const val REPORT_DIRECTORY = "gpu-int8-fc-arithmetic"
        private const val ASSET_ROOT = "whisper-int8-fc-matrix"
        private const val ROWS = 128
        private const val WIDTH = 1280
        private const val ELEMENTS = ROWS * WIDTH
        private const val ACTIVE_ROW = 10
        private const val SIGNATURE = "k_projection"
        private const val INPUT = "hidden"
        private const val OUTPUT = "key"
        private val MODELS = listOf(
            ModelSpec("original-per-channel", quantized = true),
            ModelSpec("original-explicit-zero-points", quantized = true),
            ModelSpec("requantized-per-tensor", quantized = true),
            ModelSpec("requantized-uniform-per-channel", quantized = true),
            ModelSpec("synthetic-diagonal-scale-ramp", quantized = true),
            ModelSpec("synthetic-accumulation-ladder", quantized = true),
            ModelSpec("float32-original-control", quantized = false),
        )
    }

    private class CpuRunner private constructor(
        private val interpreter: Interpreter,
    ) : AutoCloseable {
        private val input = matrixDirectFloatBuffer()
        private val output = matrixDirectFloatBuffer()

        fun run(values: FloatArray): FloatArray {
            input.putMatrixFloats(values)
            output.clear()
            interpreter.runSignature(
                mapOf(INPUT to input),
                mapOf(OUTPUT to output),
                SIGNATURE,
            )
            return output.readMatrixFloats()
        }

        override fun close() = interpreter.close()

        companion object {
            fun open(assets: AssetManager, modelAsset: String): CpuRunner =
                CpuRunner(
                    Interpreter(
                        mapMatrixAsset(assets, modelAsset),
                        Interpreter.Options()
                            .setNumThreads(3)
                            .setUseXNNPACK(true),
                    ),
                )
        }
    }

    private class GpuRunner private constructor(
        private val model: CompiledModel,
        private val input: TensorBuffer,
        private val output: TensorBuffer,
    ) : AutoCloseable {
        fun run(values: FloatArray): FloatArray {
            input.writeFloat(values)
            model.run(
                mapOf(INPUT to input),
                mapOf(OUTPUT to output),
                SIGNATURE,
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
                modelAsset: String,
                allowSourceQuantization: Boolean,
            ): GpuRunner {
                val options =
                    CompiledModel.Options(Accelerator.GPU, Accelerator.CPU).apply {
                        gpuOptions = CompiledModel.GpuOptions(
                            constantTensorSharing = true,
                            allowSrcQuantizedFcConvOps = allowSourceQuantization,
                            precision = CompiledModel.GpuOptions.Precision.FP32,
                            backend = CompiledModel.GpuOptions.Backend.OPENCL,
                        )
                    }
                val model = CompiledModel.create(assets, modelAsset, options)
                return try {
                    GpuRunner(
                        model = model,
                        input = model.createInputBuffer(INPUT, SIGNATURE),
                        output = model.createOutputBuffer(OUTPUT, SIGNATURE),
                    )
                } catch (error: Throwable) {
                    model.close()
                    throw error
                }
            }
        }
    }
}

private fun mapMatrixAsset(
    assets: AssetManager,
    path: String,
): MappedByteBuffer =
    assets.openFd(path).use { descriptor ->
        FileInputStream(descriptor.fileDescriptor).channel.use { channel ->
            channel.map(
                FileChannel.MapMode.READ_ONLY,
                descriptor.startOffset,
                descriptor.declaredLength,
            )
        }
    }

private fun readFloats(assets: AssetManager, path: String): FloatArray =
    assets.open(path).use { input ->
        val bytes = input.readBytes()
        check(bytes.size % Float.SIZE_BYTES == 0)
        ByteBuffer.wrap(bytes)
            .order(ByteOrder.LITTLE_ENDIAN)
            .asFloatBuffer()
            .let { buffer -> FloatArray(buffer.remaining()).also(buffer::get) }
    }

private fun readInts(assets: AssetManager, path: String): IntArray =
    assets.open(path).use { input ->
        val bytes = input.readBytes()
        check(bytes.size % Int.SIZE_BYTES == 0)
        ByteBuffer.wrap(bytes)
            .order(ByteOrder.LITTLE_ENDIAN)
            .asIntBuffer()
            .let { buffer -> IntArray(buffer.remaining()).also(buffer::get) }
    }

private fun matrixDirectFloatBuffer(): ByteBuffer =
    ByteBuffer.allocateDirect(128 * 1280 * Float.SIZE_BYTES)
        .order(ByteOrder.nativeOrder())

private fun ByteBuffer.putMatrixFloats(values: FloatArray) {
    clear()
    asFloatBuffer().put(values)
    rewind()
}

private fun ByteBuffer.readMatrixFloats(): FloatArray {
    rewind()
    return FloatArray(128 * 1280).also { values ->
        asFloatBuffer().get(values)
        rewind()
    }
}
