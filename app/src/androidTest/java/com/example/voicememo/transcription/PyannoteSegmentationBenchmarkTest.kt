package com.example.voicememo.transcription

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.os.Build
import android.os.PowerManager
import android.os.Process
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.channels.FileChannel
import java.security.MessageDigest
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.Interpreter

/**
 * Isolated speed and parity benchmark for pyannote segmentation-3.0 (speaker activity per
 * 17 ms frame, 10 s windows). Compares LiteRT and ONNX Runtime on staged public audio.
 *
 * The host runner (`scripts/run-pyannote-segmentation-benchmark.py`) must build the target
 * with `-PisolatedBenchmarkPackage=true`. Models, audio and reports live only under that
 * separate package's private storage; nothing is persisted in the app's data.
 */
@RunWith(AndroidJUnit4::class)
class PyannoteSegmentationBenchmarkTest {
    @Test
    fun runBenchmark() {
        val arguments = InstrumentationRegistry.getArguments()
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        check(context.packageName == BENCHMARK_PACKAGE) {
            "Refusing to run against non-isolated package ${context.packageName}."
        }
        val runId = requireSafe(arguments.getString("runId"), "runId")
        val engineName = requireSafe(arguments.getString("engine"), "engine")
        val threads = requireSafe(arguments.getString("threads"), "threads").toInt()
        val hopSamples = requireSafe(arguments.getString("hopMillis"), "hopMillis").toInt() * SAMPLE_RATE / 1_000
        require(threads in 1..8 && hopSamples in SAMPLE_RATE / 2..WINDOW_SAMPLES) { "Invalid threads or hop." }
        val modelFile = ParakeetBenchmarkSupport.stagedFile(
            context, DIRECTORY, arguments.getString("modelName").orEmpty(),
            if (engineName == ENGINE_ONNX) ".onnx" else ".tflite",
        )
        val audioFile = ParakeetBenchmarkSupport.stagedFile(
            context, DIRECTORY, arguments.getString("audioName").orEmpty(), ".wav",
        )

        val reportDirectory = File(context.cacheDir, DIRECTORY).apply { mkdirs() }
        val reportFile = File(reportDirectory, "$runId.json")
        val temporaryReport = File(reportDirectory, "$runId.json.tmp")
        val outputFile = File(reportDirectory, "$runId.f32")
        listOf(reportFile, temporaryReport, outputFile).forEach(File::delete)
        val report = JSONObject()
            .put("schema_version", 1)
            .put("run_id", runId)
            .put("package_name", context.packageName)
            .put("engine", engineName)
            .put("threads", threads)
            .put("hop_samples", hopSamples)
            .put("model_file_name", modelFile.name)
            .put("model_sha256", sha256(modelFile))
            .put("audio_file_name", audioFile.name)
            .put("audio_sha256", sha256(audioFile))
            .put("device", deviceJson())
            .put("status", "running")

        fun publish() {
            temporaryReport.writeText(report.toString(1))
            reportFile.delete()
            check(temporaryReport.renameTo(reportFile)) { "Could not publish the report." }
        }
        publish()
        try {
            report.put("result", benchmark(context, engineName, modelFile, audioFile, threads, hopSamples, outputFile))
            report.put("status", "completed")
        } catch (error: Throwable) {
            report.put("status", "failed")
                .put("error_type", error.javaClass.name)
                .put("error_message", error.message?.lineSequence()?.firstOrNull()?.take(500))
            throw error
        } finally {
            publish()
        }
    }

    private fun benchmark(
        context: Context,
        engineName: String,
        modelFile: File,
        audioFile: File,
        threads: Int,
        hopSamples: Int,
        outputFile: File,
    ): JSONObject {
        val audio = ParakeetBenchmarkSupport.readPcm16Wav(audioFile)
        val starts = (0 until maxOf(1, audio.size - WINDOW_SAMPLES + hopSamples) step hopSamples).toList()
        val thermalBefore = thermalStatus(context)
        val memoryBefore = ParakeetBenchmarkSupport.memory()

        val loadStarted = System.nanoTime()
        val engine = when (engineName) {
            ENGINE_COMPILED -> CompiledEngine(modelFile, cpuOptions(threads))
            ENGINE_GPU_FP16 -> CompiledEngine(modelFile, gpuOptions(CompiledModel.GpuOptions.Precision.FP16))
            ENGINE_GPU_FP32 -> CompiledEngine(modelFile, gpuOptions(CompiledModel.GpuOptions.Precision.FP32))
            ENGINE_GPU_FP16_CPU_FALLBACK ->
                CompiledEngine(modelFile, gpuOptions(CompiledModel.GpuOptions.Precision.FP16, cpuFallbackThreads = threads))
            ENGINE_INTERPRETER -> InterpreterEngine(modelFile, threads, resetState = true)
            ENGINE_INTERPRETER_KEEP_STATE -> InterpreterEngine(modelFile, threads, resetState = false)
            ENGINE_ONNX -> OnnxEngine(modelFile, threads)
            else -> error("Unsupported engine: $engineName")
        }
        val loadMillis = (System.nanoTime() - loadStarted) / 1e6
        val memoryLoaded = ParakeetBenchmarkSupport.memory()

        val chunkMillis = JSONArray()
        val classes = JSONArray()
        val digest = MessageDigest.getInstance("SHA-256")
        val window = FloatArray(WINDOW_SAMPLES)
        val bytes = ByteBuffer.allocate(OUTPUT_FLOATS * Float.SIZE_BYTES).order(ByteOrder.LITTLE_ENDIAN)
        val cpuStarted = Process.getElapsedCpuTime()
        val wallStarted = System.nanoTime()
        engine.use {
            outputFile.outputStream().buffered().use { output ->
                for (start in starts) {
                    val end = minOf(audio.size, start + WINDOW_SAMPLES)
                    for (index in 0 until WINDOW_SAMPLES) {
                        window[index] = if (start + index < end) audio[start + index] else 0f
                    }
                    val chunkStarted = System.nanoTime()
                    val logProbabilities = engine.run(window)
                    chunkMillis.put((System.nanoTime() - chunkStarted) / 1e6)
                    check(logProbabilities.size == OUTPUT_FLOATS) { "Unexpected output size ${logProbabilities.size}." }

                    bytes.clear()
                    bytes.asFloatBuffer().put(logProbabilities)
                    output.write(bytes.array())
                    digest.update(bytes.array())
                    val best = StringBuilder(FRAMES)
                    for (frame in 0 until FRAMES) {
                        var bestClass = 0
                        for (candidate in 1 until CLASSES) {
                            if (logProbabilities[frame * CLASSES + candidate] > logProbabilities[frame * CLASSES + bestClass]) {
                                bestClass = candidate
                            }
                        }
                        best.append(bestClass)
                    }
                    classes.put(best.toString())
                }
            }
        }
        val wallMillis = (System.nanoTime() - wallStarted) / 1e6
        val audioSeconds = audio.size.toDouble() / SAMPLE_RATE
        return JSONObject()
            .put("audio_seconds", audioSeconds)
            .put("chunks", starts.size)
            .put("model_load_ms", loadMillis)
            .put("inference_wall_ms", wallMillis)
            .put("inference_cpu_ms", Process.getElapsedCpuTime() - cpuStarted)
            .put("real_time_factor", wallMillis / 1_000.0 / audioSeconds)
            .put("chunk_ms", chunkMillis)
            .put("frame_classes", classes)
            .put("output_sha256", digest.digest().joinToString("") { "%02x".format(it) })
            .put("output_file_name", outputFile.name)
            .put("memory_before", memoryBefore)
            .put("memory_after_load", memoryLoaded)
            .put("memory_after_run", ParakeetBenchmarkSupport.memory())
            .put("thermal_status_before", thermalBefore)
            .put("thermal_status_after", thermalStatus(context))
    }

    private interface Engine : AutoCloseable {
        fun run(window: FloatArray): FloatArray
    }

    private fun cpuOptions(threads: Int) = CompiledModel.Options(Accelerator.CPU).apply {
        cpuOptions = CompiledModel.CpuOptions(threads, null, null)
    }

    /**
     * GPU only by default, so a model the GPU cannot take fails instead of silently
     * running on the CPU. No program cache: the load time is the cold compile.
     */
    private fun gpuOptions(precision: CompiledModel.GpuOptions.Precision, cpuFallbackThreads: Int? = null) =
        (if (cpuFallbackThreads == null) CompiledModel.Options(Accelerator.GPU)
        else CompiledModel.Options(Accelerator.GPU, Accelerator.CPU)).apply {
            gpuOptions = CompiledModel.GpuOptions(precision = precision)
            if (cpuFallbackThreads != null) cpuOptions = CompiledModel.CpuOptions(cpuFallbackThreads, null, null)
        }

    /** LiteRT CompiledModel: CPU/XNNPACK as the production Parakeet decoder runs, or GPU. */
    private class CompiledEngine(modelFile: File, options: CompiledModel.Options) : Engine {
        private val model = CompiledModel.create(modelFile.absolutePath, options)
        private val inputs = model.createInputBuffers()
        private val outputs = model.createOutputBuffers()

        override fun run(window: FloatArray): FloatArray {
            inputs[0].writeFloat(window)
            model.run(inputs, outputs)
            return outputs[0].readFloat()
        }

        override fun close() {
            inputs.forEach { it.close() }
            outputs.forEach { it.close() }
            model.close()
        }
    }

    /**
     * Classic interpreter. Fused sequence-LSTM ops keep their hidden state in variable
     * tensors, so the state has to be cleared before every independent window.
     */
    private class InterpreterEngine(modelFile: File, threads: Int, private val resetState: Boolean) : Engine {
        private val interpreter = Interpreter(
            FileInputStream(modelFile).channel.use { it.map(FileChannel.MapMode.READ_ONLY, 0L, it.size()) },
            Interpreter.Options().setNumThreads(threads).setUseXNNPACK(true),
        )
        private val input = directFloats(WINDOW_SAMPLES)
        private val output = directFloats(OUTPUT_FLOATS)

        override fun run(window: FloatArray): FloatArray {
            if (resetState) interpreter.resetVariableTensors()
            input.rewind()
            input.asFloatBuffer().put(window)
            output.rewind()
            interpreter.run(input, output)
            output.rewind()
            return FloatArray(OUTPUT_FLOATS).also { output.asFloatBuffer().get(it) }
        }

        override fun close() = interpreter.close()
    }

    /** ONNX Runtime CPU, the runtime the app already uses for Silero VAD. */
    private class OnnxEngine(modelFile: File, threads: Int) : Engine {
        private val environment = OrtEnvironment.getEnvironment()
        private val session = OrtSession.SessionOptions().use { options ->
            options.setInterOpNumThreads(1)
            options.setIntraOpNumThreads(threads)
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            environment.createSession(modelFile.absolutePath, options)
        }
        private val input: FloatBuffer = directFloats(WINDOW_SAMPLES).asFloatBuffer()
        private val tensor = OnnxTensor.createTensor(environment, input, longArrayOf(1, 1, WINDOW_SAMPLES.toLong()))

        override fun run(window: FloatArray): FloatArray {
            input.rewind()
            input.put(window)
            input.rewind()
            return session.run(mapOf("x" to tensor)).use { result ->
                val buffer = (result[0] as OnnxTensor).floatBuffer
                FloatArray(buffer.remaining()).also(buffer::get)
            }
        }

        override fun close() {
            tensor.close()
            session.close()
        }
    }

    private companion object {
        private const val BENCHMARK_PACKAGE = "com.example.voicememo.benchmark"
        private const val DIRECTORY = "pyannote-seg-benchmark"
        private const val ENGINE_COMPILED = "litert-compiled"
        private const val ENGINE_GPU_FP16 = "litert-gpu-fp16"
        private const val ENGINE_GPU_FP32 = "litert-gpu-fp32"
        private const val ENGINE_GPU_FP16_CPU_FALLBACK = "litert-gpu-fp16-cpu-fallback"
        private const val ENGINE_INTERPRETER = "litert-interpreter"
        private const val ENGINE_INTERPRETER_KEEP_STATE = "litert-interpreter-keep-state"
        private const val ENGINE_ONNX = "onnx"
        private const val SAMPLE_RATE = 16_000
        private const val WINDOW_SAMPLES = 160_000
        private const val FRAMES = 589
        private const val CLASSES = 7
        private const val OUTPUT_FLOATS = FRAMES * CLASSES
        private val SAFE_VALUE = Regex("[A-Za-z0-9-]{1,128}")

        private fun requireSafe(value: String?, label: String): String =
            value?.takeIf(SAFE_VALUE::matches) ?: error("$label is missing or invalid.")

        private fun directFloats(count: Int): ByteBuffer =
            ByteBuffer.allocateDirect(count * Float.SIZE_BYTES).order(ByteOrder.nativeOrder())

        private fun sha256(file: File): String {
            val digest = MessageDigest.getInstance("SHA-256")
            file.inputStream().buffered().use { input ->
                val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                while (true) {
                    val count = input.read(buffer)
                    if (count < 0) break
                    digest.update(buffer, 0, count)
                }
            }
            return digest.digest().joinToString("") { "%02x".format(it) }
        }

        private fun thermalStatus(context: Context): Any =
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                context.getSystemService(PowerManager::class.java)?.currentThermalStatus ?: JSONObject.NULL
            } else {
                JSONObject.NULL
            }

        private fun deviceJson(): JSONObject = JSONObject()
            .put("manufacturer", Build.MANUFACTURER)
            .put("model", Build.MODEL)
            .put("soc", if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) Build.SOC_MODEL else "")
            .put("android_release", Build.VERSION.RELEASE)
            .put("sdk", Build.VERSION.SDK_INT)
    }
}
