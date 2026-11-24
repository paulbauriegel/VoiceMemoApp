package com.example.voicememo.transcription

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.voicememo.transcription.ParakeetBenchmarkSupport.FRAME_SECONDS
import com.example.voicememo.transcription.ParakeetBenchmarkSupport.SAMPLE_RATE
import com.example.voicememo.transcription.ParakeetBenchmarkSupport.TimedPiece
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.math.max
import kotlin.math.min

/**
 * Diagnostic benchmark: Parakeet TDT 0.6B v3 exported by scripts/export-parakeet-litert-30s.py,
 * 30 s windows at a 25 s stride with the same gap repair as the ggml benchmark. Features follow
 * the Transformers/NeMo frontend exactly.
 *
 * Model layouts:
 * - single file (`*_step.tflite`): signatures `encode` and `decode_step` (LSTM + joint per frame).
 * - split (`*_encoder.tflite` + `*_decoder.tflite`): `encode` in one file; `predict` (LSTM, run
 *   only after a non-blank emission) and `joint` (one frame, argmax in-graph) in the other.
 *
 * Backends: `cpu` runs everything on XNNPACK. `gpu` runs `encode` on the GPU (CPU fallback for
 * unsupported ops) and the decoder on the CPU.
 */
@RunWith(AndroidJUnit4::class)
class ParakeetLiteRtBenchmarkTest {
    @Test
    fun transcribeStagedAudio() {
        val arguments = InstrumentationRegistry.getArguments()
        val modelName = arguments.getString(ARG_MODEL_NAME)
        assumeTrue("Parakeet LiteRT model not requested.", modelName != null)
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val audioNames = requireNotNull(arguments.getString(ARG_AUDIO_NAMES)) {
            "Pass -e $ARG_AUDIO_NAMES name1.wav,name2.wav"
        }.split(',').filter { it.isNotBlank() }
        val backend = arguments.getString(ARG_BACKEND) ?: "cpu"
        require(backend == "cpu" || backend == "gpu") { "Backend must be cpu or gpu." }
        val gpuPrecision = arguments.getString(ARG_GPU_PRECISION) ?: "fp32"
        require(gpuPrecision == "fp32" || gpuPrecision == "fp16") { "GPU precision must be fp32 or fp16." }
        val threads = arguments.getString(ARG_THREADS)?.toInt() ?: DEFAULT_THREADS
        val gapRepair = arguments.getString(ARG_GAP_REPAIR)?.toBoolean() ?: true
        val repairMode = arguments.getString(ARG_REPAIR_MODE) ?: "tight"
        require(repairMode == "tight" || repairMode == "full") { "Repair mode must be tight or full." }
        val runId = arguments.getString(ARG_RUN_ID) ?: System.currentTimeMillis().toString()
        val gpuCache = arguments.getString(ARG_GPU_CACHE)?.toBoolean() ?: true
        val modelFile = ParakeetBenchmarkSupport.stagedFile(context, MODEL_DIRECTORY, requireNotNull(modelName), ".tflite")
        val split = modelFile.name.endsWith(ENCODER_SUFFIX)
        val decoderFile = if (split) {
            ParakeetBenchmarkSupport.stagedFile(
                context,
                MODEL_DIRECTORY,
                modelFile.name.removeSuffix(ENCODER_SUFFIX) + DECODER_SUFFIX,
                ".tflite",
            )
        } else {
            null
        }
        val reportDirectory = File(context.cacheDir, REPORT_DIRECTORY).apply { mkdirs() }

        val before = ParakeetBenchmarkSupport.memory()
        val loadStarted = System.nanoTime()
        val cpu = CompiledModel.Options(Accelerator.CPU).apply {
            cpuOptions = CompiledModel.CpuOptions(threads, null, null)
        }
        val gpu = CompiledModel.Options(Accelerator.GPU, Accelerator.CPU).apply {
            gpuOptions = CompiledModel.GpuOptions(
                precision = if (gpuPrecision == "fp16") CompiledModel.GpuOptions.Precision.FP16
                else CompiledModel.GpuOptions.Precision.FP32,
                constantTensorSharing = true,
                serializationDir = if (gpuCache) File(context.cacheDir, GPU_CACHE_DIRECTORY).apply { mkdirs() }.absolutePath else null,
                modelCacheKey = if (gpuCache) "${modelFile.name}-${modelFile.length()}-$gpuPrecision" else null,
                serializeProgramCache = if (gpuCache) true else null,
            )
        }
        val gpuCacheWarm = gpuCache && File(context.cacheDir, GPU_CACHE_DIRECTORY).list()?.isNotEmpty() == true
        val encoderOptions = if (backend == "gpu") gpu else cpu
        val engine: Engine = if (decoderFile != null) {
            SplitEngine.open(modelFile, decoderFile, encoderOptions, cpu)
        } else {
            StepEngine.open(modelFile, backend == "gpu", encoderOptions, cpu)
        }
        val loadMillis = (System.nanoTime() - loadStarted) / 1_000_000
        val report = JSONObject()
            .put("run_id", runId)
            .put("engine", "litert")
            .put("layout", if (split) "split" else "step")
            .put("model", modelFile.name)
            .put("model_bytes", modelFile.length() + (decoderFile?.length() ?: 0L))
            .put("backend", if (backend == "gpu") "gpu-encode/cpu-decode ($gpuPrecision)" else "cpu")
            .put("threads", threads)
            .put("gap_repair", gapRepair)
            .put("repair_mode", repairMode)
            .put("window_seconds", ParakeetBenchmarkSupport.WINDOW_SECONDS)
            .put("stride_seconds", ParakeetBenchmarkSupport.STRIDE_SECONDS)
            .put("model_load_ms", loadMillis)
            .put("gpu_cache", gpuCache)
            .put("gpu_cache_warm_at_start", gpuCacheWarm)
            .put("memory_before", before)
            .put("memory_after_load", ParakeetBenchmarkSupport.memory())
        val samples = JSONArray()
        var peakPssKib = 0L
        var dumpedFeatures = false
        var frontendMillis = 0L
        val decoder = ParakeetBenchmarkSupport.WindowDecoder { audio, start, end, stats ->
            val started = System.nanoTime()
            val window = audio.slice(start, end)
            val features = ParakeetFrontend.extract(window, 0, window.size)
            frontendMillis += (System.nanoTime() - started) / 1_000_000
            if (!dumpedFeatures) {
                // First window of the first sample, for host-side frontend parity.
                File(reportDirectory, "features-$runId.f32").writeBytes(
                    ByteBuffer.allocate(features.values.size * 4).order(ByteOrder.LITTLE_ENDIAN)
                        .also { buffer -> features.values.forEach(buffer::putFloat) }.array(),
                )
                dumpedFeatures = true
            }
            val tokens = engine.transcribe(features)
            stats.decodes++
            stats.decodeMillis += (System.nanoTime() - started) / 1_000_000
            val offset = start.toDouble() / SAMPLE_RATE
            tokens.map { token ->
                TimedPiece(
                    token.id,
                    "",
                    offset + token.frame * FRAME_SECONDS,
                    offset + (token.frame + token.duration) * FRAME_SECONDS,
                )
            }
        }
        try {
            for (name in audioNames) {
                val audio = ParakeetBenchmarkSupport.readPcm16Wav(
                    ParakeetBenchmarkSupport.stagedFile(context, AUDIO_DIRECTORY, name, ".wav"),
                )
                samples.put(
                    ParakeetBenchmarkSupport.transcribeSample(name, audio, decoder, gapRepair, repairMode == "tight") {
                        peakPssKib = max(peakPssKib, ParakeetBenchmarkSupport.memory().getLong("total_pss_kib"))
                    }
                        .put("frontend_ms", frontendMillis)
                        .put("encode_ms", engine.timing.encodeMillis)
                        .put("decode_step_ms", engine.timing.decodeMillis)
                        .put("predict_calls", engine.timing.predictCalls)
                        .put("joint_calls", engine.timing.jointCalls)
                        .put("memory_loaded", ParakeetBenchmarkSupport.memory()),
                )
                frontendMillis = 0
                engine.timing.reset()
            }
        } finally {
            engine.close()
        }
        report.put("samples", samples).put("peak_total_pss_kib", peakPssKib).put("memory_after", ParakeetBenchmarkSupport.memory())
        File(reportDirectory, "parakeet-$runId.json").writeText(report.toString(1))
    }

    private data class Token(val id: Int, val frame: Int, val duration: Int)

    private class Timing {
        var encodeMillis = 0L
        var decodeMillis = 0L
        var predictCalls = 0
        var jointCalls = 0

        fun reset() {
            encodeMillis = 0
            decodeMillis = 0
            predictCalls = 0
            jointCalls = 0
        }
    }

    private interface Engine : AutoCloseable {
        val timing: Timing
        fun transcribe(features: ParakeetFrontend.Features): List<Token>
    }

    /** `encode` signature shared by both layouts. */
    private class Encoder(private val model: CompiledModel) : AutoCloseable {
        private val inputs = listOf(model.createInputBuffer("args_0", ENCODE), model.createInputBuffer("args_1", ENCODE))
        private val outputs = listOf(model.createOutputBuffer("output_0", ENCODE))

        fun encode(features: ParakeetFrontend.Features): FloatArray {
            inputs[0].writeFloat(features.values)
            inputs[1].writeInt(features.mask)
            model.run(inputs, outputs, ENCODE)
            return outputs[0].readFloat()
        }

        override fun close() {
            (inputs + outputs).forEach { runCatching(it::close) }
        }
    }

    /** Single file: LSTM + joint on every visited frame, full logits read back. */
    private class StepEngine(
        private val encoderModel: CompiledModel,
        private val stepModel: CompiledModel,
    ) : Engine {
        override val timing = Timing()
        private val encoder = Encoder(encoderModel)
        private val stepInputs = listOf("args_0", "args_1", "args_2", "args_3").map { stepModel.createInputBuffer(it, STEP) }
        private val stepOutputs = listOf("output_0", "output_1", "output_2").map { stepModel.createOutputBuffer(it, STEP) }

        override fun transcribe(features: ParakeetFrontend.Features): List<Token> {
            var started = System.nanoTime()
            val encoded = encoder.encode(features)
            timing.encodeMillis += (System.nanoTime() - started) / 1_000_000
            started = System.nanoTime()
            val valid = validFrames(encoded, features)
            var hidden = FloatArray(STATE_SIZE)
            var cell = FloatArray(STATE_SIZE)
            var last = BLANK
            val frame = FloatArray(ENCODED_FEATURES)
            val out = greedy(valid) { time ->
                encoded.copyInto(frame, 0, time * ENCODED_FEATURES, (time + 1) * ENCODED_FEATURES)
                stepInputs[0].writeInt(intArrayOf(last))
                stepInputs[1].writeFloat(hidden)
                stepInputs[2].writeFloat(cell)
                stepInputs[3].writeFloat(frame)
                stepModel.run(stepInputs, stepOutputs, STEP)
                timing.jointCalls++
                val logits = stepOutputs[0].readFloat()
                val token = argmax(logits, 0, VOCAB)
                val durationIndex = argmax(logits, VOCAB, VOCAB + DURATIONS.size) - VOCAB
                if (token != BLANK) {
                    last = token
                    hidden = stepOutputs[1].readFloat()
                    cell = stepOutputs[2].readFloat()
                }
                token to durationIndex
            }
            timing.decodeMillis += (System.nanoTime() - started) / 1_000_000
            return out
        }

        override fun close() {
            encoder.close()
            (stepInputs + stepOutputs).forEach { runCatching(it::close) }
            stepModel.close()
            if (encoderModel !== stepModel) encoderModel.close()
        }

        companion object {
            fun open(file: File, separateEncoder: Boolean, encoderOptions: CompiledModel.Options, cpu: CompiledModel.Options): StepEngine {
                val stepModel = CompiledModel.create(file.absolutePath, cpu)
                if (!separateEncoder) return StepEngine(stepModel, stepModel)
                val encoderModel = try {
                    CompiledModel.create(file.absolutePath, encoderOptions)
                } catch (failure: Throwable) {
                    stepModel.close()
                    throw failure
                }
                return StepEngine(encoderModel, stepModel)
            }
        }
    }

    /**
     * Split files: `predict` runs once at the start and after each emission; `joint` returns
     * (token, duration index) as two ints, so no logits are copied back.
     */
    private class SplitEngine(
        private val encoderModel: CompiledModel,
        private val decoderModel: CompiledModel,
    ) : Engine {
        override val timing = Timing()
        private val encoder = Encoder(encoderModel)
        private val predictInputs = listOf("args_0", "args_1", "args_2").map { decoderModel.createInputBuffer(it, PREDICT) }
        private val predictOutputs = listOf("output_0", "output_1", "output_2").map { decoderModel.createOutputBuffer(it, PREDICT) }
        private val jointInputs = listOf("args_0", "args_1").map { decoderModel.createInputBuffer(it, JOINT) }
        private val jointOutputs = listOf("output_0", "output_1").map { decoderModel.createOutputBuffer(it, JOINT) }
        private val tokenInput = IntArray(1)

        override fun transcribe(features: ParakeetFrontend.Features): List<Token> {
            var started = System.nanoTime()
            val encoded = encoder.encode(features)
            timing.encodeMillis += (System.nanoTime() - started) / 1_000_000
            started = System.nanoTime()
            val valid = validFrames(encoded, features)
            // Decoder output for the current label context, and the state *after* consuming it.
            var hidden = FloatArray(STATE_SIZE)
            var cell = FloatArray(STATE_SIZE)
            var decoderOutput = predict(BLANK, hidden, cell).also { (_, h, c) -> hidden = h; cell = c }.first
            jointInputs[1].writeFloat(decoderOutput)
            val frame = FloatArray(ENCODED_FEATURES)
            val out = greedy(valid) { time ->
                encoded.copyInto(frame, 0, time * ENCODED_FEATURES, (time + 1) * ENCODED_FEATURES)
                jointInputs[0].writeFloat(frame)
                decoderModel.run(jointInputs, jointOutputs, JOINT)
                timing.jointCalls++
                val token = jointOutputs[0].readInt()[0]
                val durationIndex = jointOutputs[1].readInt()[0]
                if (token != BLANK) {
                    val (output, h, c) = predict(token, hidden, cell)
                    decoderOutput = output
                    hidden = h
                    cell = c
                    jointInputs[1].writeFloat(decoderOutput)
                }
                token to durationIndex
            }
            timing.decodeMillis += (System.nanoTime() - started) / 1_000_000
            return out
        }

        private fun predict(token: Int, hidden: FloatArray, cell: FloatArray): Triple<FloatArray, FloatArray, FloatArray> {
            tokenInput[0] = token
            predictInputs[0].writeInt(tokenInput)
            predictInputs[1].writeFloat(hidden)
            predictInputs[2].writeFloat(cell)
            decoderModel.run(predictInputs, predictOutputs, PREDICT)
            timing.predictCalls++
            return Triple(predictOutputs[0].readFloat(), predictOutputs[1].readFloat(), predictOutputs[2].readFloat())
        }

        override fun close() {
            encoder.close()
            (predictInputs + predictOutputs + jointInputs + jointOutputs).forEach { runCatching(it::close) }
            decoderModel.close()
            encoderModel.close()
        }

        companion object {
            fun open(encoderFile: File, decoderFile: File, encoderOptions: CompiledModel.Options, cpu: CompiledModel.Options): SplitEngine {
                val decoderModel = CompiledModel.create(decoderFile.absolutePath, cpu)
                val encoderModel = try {
                    CompiledModel.create(encoderFile.absolutePath, encoderOptions)
                } catch (failure: Throwable) {
                    decoderModel.close()
                    throw failure
                }
                return SplitEngine(encoderModel, decoderModel)
            }
        }
    }

    private companion object {
        const val ARG_MODEL_NAME = "parakeet_model_name"
        const val ARG_AUDIO_NAMES = "parakeet_audio_names"
        const val ARG_BACKEND = "parakeet_backend"
        const val ARG_GPU_PRECISION = "parakeet_gpu_precision"
        const val ARG_THREADS = "parakeet_threads"
        const val ARG_GAP_REPAIR = "parakeet_gap_repair"
        const val ARG_REPAIR_MODE = "parakeet_repair_mode"
        const val ARG_RUN_ID = "run_id"
        const val ARG_GPU_CACHE = "parakeet_gpu_cache"
        const val GPU_CACHE_DIRECTORY = "parakeet-litert-gpu-cache"
        const val MODEL_DIRECTORY = "parakeet-litert"
        const val AUDIO_DIRECTORY = "gpu-audio"
        const val REPORT_DIRECTORY = "parakeet-litert"
        const val ENCODER_SUFFIX = "_encoder.tflite"
        const val DECODER_SUFFIX = "_decoder.tflite"
        const val DEFAULT_THREADS = 4
        const val ENCODE = "encode"
        const val STEP = "decode_step"
        const val PREDICT = "predict"
        const val JOINT = "joint"
        const val ENCODED_FEATURES = 640
        const val STATE_SIZE = 2 * 640
        const val SUBSAMPLING = 8
        const val BLANK = 8_192
        const val VOCAB = 8_193
        const val MAX_SYMBOLS_PER_STEP = 10
        val DURATIONS = intArrayOf(0, 1, 2, 3, 4)

        fun validFrames(encoded: FloatArray, features: ParakeetFrontend.Features): Int =
            min(encoded.size / ENCODED_FEATURES, (features.validMelFrames - 1) / SUBSAMPLING + 1)

        /**
         * TDT greedy loop shared by both layouts. [step] evaluates the joint for one frame with
         * the current label context and returns (token, duration index); it is responsible for
         * advancing the label context when the token is not blank.
         */
        inline fun greedy(valid: Int, step: (time: Int) -> Pair<Int, Int>): List<Token> {
            var time = 0
            var symbols = 0
            val out = mutableListOf<Token>()
            while (time < valid) {
                val (token, durationIndex) = step(time)
                var duration = DURATIONS[durationIndex]
                if (token != BLANK) {
                    out += Token(token, time, duration)
                    symbols++
                    if (duration == 0 && symbols >= MAX_SYMBOLS_PER_STEP) duration = 1
                } else if (duration == 0) {
                    duration = 1
                }
                if (duration > 0) symbols = 0
                time += duration
            }
            return out
        }

        fun argmax(values: FloatArray, start: Int, endExclusive: Int): Int {
            var best = start
            for (index in start + 1 until endExclusive) if (values[index] > values[best]) best = index
            return best
        }
    }
}
