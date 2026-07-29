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
import kotlin.math.sqrt
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.Interpreter

/**
 * Standalone reproduction for the first Whisper decoder block's per-channel
 * INT8 K projection. It has no dependency on recordings, Room, or app state.
 */
@RunWith(AndroidJUnit4::class)
class WhisperInt8FullyConnectedRegressionTest {
    @Test
    fun compareExactWhisperProjection() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val arguments = InstrumentationRegistry.getArguments()
        val context = instrumentation.targetContext
        val runId = requireSafe(arguments.getString(ARG_RUN_ID))
        val benchmarkMode =
            arguments.getString(ARG_BENCHMARK_MODE)?.toBooleanStrictOrNull() ?: false
        val fixture = instrumentation.context.assets.open(FIXTURE_ASSET).use { input ->
            input.readBytes()
        }
        check(fixture.size == INPUT_FLOATS * Float.SIZE_BYTES) {
            "INT8 FC fixture has ${fixture.size} bytes; expected " +
                "${INPUT_FLOATS * Float.SIZE_BYTES}."
        }
        val input = ByteBuffer.wrap(fixture)
            .order(ByteOrder.LITTLE_ENDIAN)
            .asFloatBuffer()
            .let { floats ->
                FloatArray(INPUT_FLOATS).also(floats::get)
            }

        val reportDirectory = File(context.cacheDir, REPORT_DIRECTORY).apply { mkdirs() }
        val report = JSONObject()
            .put("run_id", runId)
            .put("model_sha256", MODEL_SHA256)
            .put("fixture_sha256", FIXTURE_SHA256)
            .put("shape", JSONArray(listOf(1, ROWS, WIDTH)))
            .put("active_row", ACTIVE_ROW)
            .put("benchmark_mode", benchmarkMode)
            .put("variants", JSONArray())

        try {
            val cpuStarted = System.nanoTime()
            val cpu = CpuProjection.open(instrumentation.context.assets).use { runner ->
                runner.run(input)
            }
            report.put("cpu_inference_millis", elapsedMillis(cpuStarted))

            val variants = listOf(
                Variant(
                    name = "direct-int8-fc",
                    allowSourceQuantization = false,
                    preferTextureWeights = null,
                ),
                Variant(
                    name = "source-quantized-default",
                    allowSourceQuantization = true,
                    preferTextureWeights = null,
                ),
                Variant(
                    name = "source-quantized-buffer-weights",
                    allowSourceQuantization = true,
                    preferTextureWeights = false,
                ),
                Variant(
                    name = "source-quantized-texture-weights",
                    allowSourceQuantization = true,
                    preferTextureWeights = true,
                ),
                Variant(
                    name = "source-quantized-buffer-storage",
                    allowSourceQuantization = true,
                    preferTextureWeights = null,
                    bufferStorageType =
                        CompiledModel.GpuOptions.BufferStorageType.BUFFER,
                ),
                Variant(
                    name = "source-quantized-texture-storage",
                    allowSourceQuantization = true,
                    preferTextureWeights = null,
                    bufferStorageType =
                        CompiledModel.GpuOptions.BufferStorageType.TEXTURE_2D,
                ),
            )
            val variantReports = report.getJSONArray("variants")
            variants.forEach { variant ->
                val cacheDirectory = File(
                    reportDirectory,
                    "program-cache/${variant.name}",
                ).apply {
                    deleteRecursively()
                    check(mkdirs()) { "Could not create GPU program cache directory." }
                }
                val loadStarted = System.nanoTime()
                val gpuRunner = GpuProjection.open(
                    assets = instrumentation.context.assets,
                    variant = variant,
                    benchmarkMode = benchmarkMode,
                    cacheDirectory = cacheDirectory,
                )
                val loadMillis = elapsedMillis(loadStarted)
                val variantReport = JSONObject()
                    .put("name", variant.name)
                    .put(
                        "allow_source_quantized_fc_conv_ops",
                        variant.allowSourceQuantization,
                    )
                    .put(
                        "prefer_texture_weights",
                        variant.preferTextureWeights ?: JSONObject.NULL,
                    )
                    .put(
                        "buffer_storage_type",
                        variant.bufferStorageType.name,
                    )
                    .put("model_load_millis", loadMillis)
                gpuRunner.use { runner ->
                    val inferenceStarted = System.nanoTime()
                    val gpu = runner.run(input)
                    val inferenceMillis = elapsedMillis(inferenceStarted)
                    variantReport
                        .put("inference_millis", inferenceMillis)
                        .put("full", compare(cpu, gpu))
                        .put(
                            "active_row",
                            compare(
                                activeRow(cpu),
                                activeRow(gpu),
                            ),
                        )
                }
                val cacheFiles =
                    cacheDirectory.walkTopDown().filter(File::isFile).toList()
                variantReport
                    .put("program_cache_files", cacheFiles.size)
                    .put(
                        "program_cache_bytes",
                        cacheFiles.sumOf(File::length),
                    )
                    .put(
                        "program_cache_sha256",
                        if (cacheFiles.size == 1) {
                            sha256(cacheFiles.single())
                        } else {
                            JSONObject.NULL
                        },
                    )
                    .put(
                        "program_cache_markers",
                        JSONArray(findKernelMarkers(cacheDirectory)),
                    )
                variantReports.put(variantReport)
            }
            report.put("status", "completed")
        } catch (error: Throwable) {
            report
                .put("status", "failed")
                .put("error_type", error.javaClass.name)
                .put("error_message", error.message ?: error.toString())
            throw error
        } finally {
            val destination = File(reportDirectory, "int8-fc-$runId.json")
            val temporary = File(reportDirectory, "${destination.name}.tmp")
            temporary.writeText(report.toString(2))
            check(temporary.renameTo(destination)) {
                "Could not publish the INT8 FC regression report."
            }
        }
    }

    private fun activeRow(values: FloatArray): FloatArray {
        val start = ACTIVE_ROW * WIDTH
        return values.copyOfRange(start, start + WIDTH)
    }

    private fun compare(expected: FloatArray, actual: FloatArray): JSONObject {
        require(expected.size == actual.size)
        var exact = 0
        var firstMismatch = -1
        var maximumAbsoluteError = 0.0
        var absoluteErrorSum = 0.0
        var squaredErrorSum = 0.0
        var expectedSquaredSum = 0.0
        var actualSquaredSum = 0.0
        var dot = 0.0
        expected.indices.forEach { index ->
            if (
                java.lang.Float.floatToRawIntBits(expected[index]) ==
                java.lang.Float.floatToRawIntBits(actual[index])
            ) {
                exact++
            } else if (firstMismatch < 0) {
                firstMismatch = index
            }
            val expectedValue = expected[index].toDouble()
            val actualValue = actual[index].toDouble()
            val difference = actualValue - expectedValue
            val absoluteError = kotlin.math.abs(difference)
            maximumAbsoluteError = maxOf(maximumAbsoluteError, absoluteError)
            absoluteErrorSum += absoluteError
            squaredErrorSum += difference * difference
            expectedSquaredSum += expectedValue * expectedValue
            actualSquaredSum += actualValue * actualValue
            dot += expectedValue * actualValue
        }
        val count = expected.size.toDouble()
        val denominator = sqrt(expectedSquaredSum * actualSquaredSum)
        return JSONObject()
            .put("element_count", expected.size)
            .put("exact_element_count", exact)
            .put("first_exact_mismatch", if (firstMismatch < 0) JSONObject.NULL else firstMismatch)
            .put("maximum_absolute_error", maximumAbsoluteError)
            .put("mean_absolute_error", absoluteErrorSum / count)
            .put("root_mean_square_error", sqrt(squaredErrorSum / count))
            .put(
                "cosine_similarity",
                if (denominator == 0.0) JSONObject.NULL else dot / denominator,
            )
    }

    private fun elapsedMillis(started: Long): Long =
        (System.nanoTime() - started) / 1_000_000L

    private fun findKernelMarkers(directory: File): List<String> {
        val markers = listOf(
            "ConvWaveMatrixMali",
            "ConvWaveMatrix",
            "ConvWaveMemory",
            "ConvGeneric",
            "FullyConnected",
            "fully_connected_int8",
            "WaveMatrixMAC",
            "dot_4x8packed",
        )
        val found = linkedSetOf<String>()
        directory.walkTopDown().filter(File::isFile).forEach { file ->
            val bytes = runCatching { file.readBytes() }.getOrNull() ?: return@forEach
            markers.forEach { marker ->
                if (bytes.indexOfSubsequence(marker.encodeToByteArray()) >= 0) {
                    found += marker
                }
            }
        }
        return found.toList()
    }

    private fun sha256(file: File): String {
        val digest = java.security.MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    private fun requireSafe(value: String?): String {
        val result = value.orEmpty()
        require(result.matches(Regex("[A-Za-z0-9-]+"))) {
            "$ARG_RUN_ID is missing or invalid."
        }
        return result
    }

    private data class Variant(
        val name: String,
        val allowSourceQuantization: Boolean,
        val preferTextureWeights: Boolean?,
        val bufferStorageType: CompiledModel.GpuOptions.BufferStorageType =
            CompiledModel.GpuOptions.BufferStorageType.DEFAULT,
    )

    companion object {
        private const val ARG_RUN_ID = "runId"
        private const val ARG_BENCHMARK_MODE = "benchmarkMode"
        private const val REPORT_DIRECTORY = "gpu-int8-fc"
        private const val MODEL_ASSET =
            "whisper-decoder-prefix/k-projection-probe.tflite"
        private const val FIXTURE_ASSET =
            "whisper-decoder-prefix/k-projection-step-7-input.f32"
        private const val MODEL_SHA256 =
            "7a30331ccb89e332edb176587d149f8d852e5b224864247defadae37db940b1d"
        private const val FIXTURE_SHA256 =
            "105b86c9c20463e3920d871f0f2a653a9da88ce8739572f489e077754f6c29cb"
        private const val ROWS = 128
        private const val WIDTH = 1280
        private const val INPUT_FLOATS = ROWS * WIDTH
        private const val ACTIVE_ROW = 10
        private const val SIGNATURE = "k_projection"
        private const val INPUT = "hidden"
        private const val OUTPUT = "key"
    }

    private class CpuProjection private constructor(
        private val interpreter: Interpreter,
    ) : AutoCloseable {
        private val input = directFloatBuffer(INPUT_FLOATS)
        private val output = directFloatBuffer(INPUT_FLOATS)

        fun run(values: FloatArray): FloatArray {
            input.putFloats(values)
            output.clear()
            interpreter.runSignature(
                mapOf(INPUT to input),
                mapOf(OUTPUT to output),
                SIGNATURE,
            )
            return output.readFloats(INPUT_FLOATS)
        }

        override fun close() = interpreter.close()

        companion object {
            fun open(assets: AssetManager): CpuProjection =
                CpuProjection(
                    Interpreter(
                        mapAsset(assets, MODEL_ASSET),
                        Interpreter.Options()
                            .setNumThreads(3)
                            .setUseXNNPACK(true),
                    ),
                )
        }
    }

    private class GpuProjection private constructor(
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
                variant: Variant,
                benchmarkMode: Boolean,
                cacheDirectory: File,
            ): GpuProjection {
                val options =
                    CompiledModel.Options(Accelerator.GPU, Accelerator.CPU).apply {
                        gpuOptions = CompiledModel.GpuOptions(
                            constantTensorSharing = true,
                            allowSrcQuantizedFcConvOps =
                                variant.allowSourceQuantization,
                            precision = CompiledModel.GpuOptions.Precision.FP32,
                            backend = CompiledModel.GpuOptions.Backend.OPENCL,
                            preferTextureWeights = variant.preferTextureWeights,
                            bufferStorageType = variant.bufferStorageType,
                            serializationDir = cacheDirectory.absolutePath,
                            serializeProgramCache = true,
                            modelCacheKey =
                                if (benchmarkMode) {
                                    "${variant.name}\"\n" +
                                        "benchmark_mode = true\n" +
                                        "model_cache_key = \"${variant.name}"
                                } else {
                                    variant.name
                                },
                        )
                    }
                val model = CompiledModel.create(assets, MODEL_ASSET, options)
                return try {
                    GpuProjection(
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

private fun ByteArray.indexOfSubsequence(needle: ByteArray): Int {
    if (needle.isEmpty()) return 0
    if (needle.size > size) return -1
    val lastStart = size - needle.size
    for (start in 0..lastStart) {
        var matches = true
        for (offset in needle.indices) {
            if (this[start + offset] != needle[offset]) {
                matches = false
                break
            }
        }
        if (matches) return start
    }
    return -1
}

private fun mapAsset(
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

private fun directFloatBuffer(elements: Int): ByteBuffer =
    ByteBuffer.allocateDirect(elements * Float.SIZE_BYTES)
        .order(ByteOrder.nativeOrder())

private fun ByteBuffer.putFloats(values: FloatArray) {
    clear()
    asFloatBuffer().put(values)
    rewind()
}

private fun ByteBuffer.readFloats(count: Int): FloatArray {
    rewind()
    return FloatArray(count).also { values ->
        asFloatBuffer().get(values)
        rewind()
    }
}
