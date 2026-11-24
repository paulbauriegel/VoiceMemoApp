package com.example.voicememo.transcription

import android.content.Context
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.min

/**
 * Parakeet TDT 0.6B v3 on LiteRT for one window of up to 30 seconds.
 *
 * Split export: the encoder file holds `encode` (GPU, F32_F16 with CPU fallback for the few
 * unsupported ops); the decoder file holds `predict` (LSTM, run only after an emission) and
 * `joint` (one frame, argmax in-graph, returns token and duration index). Greedy TDT decoding
 * follows NeMo: blank 8192, durations 0-4, at most 10 symbols per frame.
 *
 * Each chunk is decoded once and then gap-repaired ([ParakeetGapRepair]). Results carry
 * word-level timed segments relative to the chunk start; windows are joined by
 * [ParakeetTranscriptMerger].
 */
internal class LiteRtParakeetEngine private constructor(
    private val encoderModel: CompiledModel,
    private val decoderModel: CompiledModel,
    private val tokenizer: ParakeetTokenizer,
    val usesGpu: Boolean,
) : WhisperEngine {
    private val closed = AtomicBoolean(false)
    private val inferenceLock = Any()
    private val encodeInputs = listOf(
        encoderModel.createInputBuffer("args_0", ENCODE),
        encoderModel.createInputBuffer("args_1", ENCODE),
    )
    private val encodeOutputs = listOf(encoderModel.createOutputBuffer("output_0", ENCODE))
    private val predictInputs = listOf("args_0", "args_1", "args_2").map {
        decoderModel.createInputBuffer(it, PREDICT)
    }
    private val predictOutputs = listOf("output_0", "output_1", "output_2").map {
        decoderModel.createOutputBuffer(it, PREDICT)
    }
    private val jointInputs = listOf("args_0", "args_1").map { decoderModel.createInputBuffer(it, JOINT) }
    private val jointOutputs = listOf("output_0", "output_1").map { decoderModel.createOutputBuffer(it, JOINT) }
    private val tokenInput = IntArray(1)
    private val frame = FloatArray(ENCODED_FEATURES)

    override fun transcribeChunk(
        audio: FloatArray,
        language: WhisperLanguage,
        shouldCancel: () -> Boolean,
        onTokenProgress: (generated: Int, limit: Int) -> Unit,
    ): WhisperChunkResult = synchronized(inferenceLock) {
        checkCancellation(shouldCancel)
        require(audio.size <= ParakeetAssets.MAX_SAMPLES) {
            "Parakeet accepts at most 30 seconds of 16 kHz PCM."
        }
        val started = System.nanoTime()
        val tokens = if (audio.isEmpty()) {
            emptyList()
        } else {
            val initial = decode(audio, 0, audio.size, shouldCancel)
            onTokenProgress(PROGRESS_DECODED, PROGRESS_LIMIT)
            ParakeetGapRepair.repair(audio, initial) { start, end ->
                decode(audio, start, end, shouldCancel)
            }.tokens
        }
        onTokenProgress(PROGRESS_LIMIT, PROGRESS_LIMIT)
        val ids = tokens.map(ParakeetTimedToken::id)
        WhisperChunkResult(
            tokens = ids,
            text = tokenizer.decode(ids),
            detectedLanguage = (language as? WhisperLanguage.Forced)?.code ?: "und",
            inferenceMillis = (System.nanoTime() - started) / 1_000_000L,
            truncated = false,
            warnings = if (language is WhisperLanguage.Forced) {
                listOf(
                    "Parakeet detects language automatically; the selected language is kept " +
                        "only as transcript metadata.",
                )
            } else {
                emptyList()
            },
            timedSegments = tokenizer.words(tokens),
        )
    }

    override fun decodeTokens(tokens: List<Int>): String = tokenizer.decode(tokens)

    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        // A running decode notices the flag at its next frame and throws; wait for it.
        synchronized(inferenceLock) {
            (encodeInputs + encodeOutputs + predictInputs + predictOutputs + jointInputs + jointOutputs)
                .asReversed()
                .forEach { runCatching(it::close) }
            decoderModel.close()
            encoderModel.close()
        }
    }

    /** Decodes audio[start, end) and returns tokens with times relative to audio[0]. */
    private fun decode(
        audio: FloatArray,
        start: Int,
        end: Int,
        shouldCancel: () -> Boolean,
    ): List<ParakeetTimedToken> {
        val features = ParakeetFrontend.extract(audio, start, end)
        checkCancellation(shouldCancel)
        encodeInputs[0].writeFloat(features.values)
        encodeInputs[1].writeInt(features.mask)
        encoderModel.run(encodeInputs, encodeOutputs, ENCODE)
        checkCancellation(shouldCancel)
        val encoded = encodeOutputs[0].readFloat()
        val valid = min(encoded.size / ENCODED_FEATURES, (features.validMelFrames - 1) / SUBSAMPLING + 1)
        val offsetMillis = start.toLong() * 1_000L / ParakeetAssets.SAMPLE_RATE

        var hidden = FloatArray(STATE_SIZE)
        var cell = FloatArray(STATE_SIZE)
        predict(ParakeetAssets.BLANK_TOKEN_ID, hidden, cell).let { (output, h, c) ->
            jointInputs[1].writeFloat(output)
            hidden = h
            cell = c
        }
        val out = mutableListOf<ParakeetTimedToken>()
        var time = 0
        var symbols = 0
        var steps = 0
        while (time < valid) {
            if (++steps % CANCEL_CHECK_STEPS == 0) checkCancellation(shouldCancel)
            encoded.copyInto(frame, 0, time * ENCODED_FEATURES, (time + 1) * ENCODED_FEATURES)
            jointInputs[0].writeFloat(frame)
            decoderModel.run(jointInputs, jointOutputs, JOINT)
            val token = jointOutputs[0].readInt()[0]
            var duration = DURATIONS[jointOutputs[1].readInt()[0]]
            if (token != ParakeetAssets.BLANK_TOKEN_ID) {
                out += ParakeetTimedToken(
                    id = token,
                    startMillis = offsetMillis + time * ParakeetAssets.FRAME_MILLIS,
                    endMillis = offsetMillis + (time + duration) * ParakeetAssets.FRAME_MILLIS,
                )
                symbols++
                if (duration == 0 && symbols >= MAX_SYMBOLS_PER_STEP) duration = 1
                val (output, h, c) = predict(token, hidden, cell)
                jointInputs[1].writeFloat(output)
                hidden = h
                cell = c
            } else if (duration == 0) {
                duration = 1
            }
            if (duration > 0) symbols = 0
            time += duration
        }
        return out
    }

    private fun predict(token: Int, hidden: FloatArray, cell: FloatArray): Triple<FloatArray, FloatArray, FloatArray> {
        tokenInput[0] = token
        predictInputs[0].writeInt(tokenInput)
        predictInputs[1].writeFloat(hidden)
        predictInputs[2].writeFloat(cell)
        decoderModel.run(predictInputs, predictOutputs, PREDICT)
        return Triple(predictOutputs[0].readFloat(), predictOutputs[1].readFloat(), predictOutputs[2].readFloat())
    }

    private fun checkCancellation(shouldCancel: () -> Boolean) {
        if (closed.get() || shouldCancel()) throw WhisperCancelledException()
    }

    companion object {
        private const val TAG = "LiteRtParakeetEngine"
        private const val ENCODE = "encode"
        private const val PREDICT = "predict"
        private const val JOINT = "joint"
        private const val ENCODED_FEATURES = 640
        private const val STATE_SIZE = 2 * 640
        private const val SUBSAMPLING = 8
        private const val MAX_SYMBOLS_PER_STEP = 10
        private const val CANCEL_CHECK_STEPS = 32
        private const val PROGRESS_DECODED = 80
        private const val PROGRESS_LIMIT = 100
        private const val CPU_THREADS = 4
        private const val GPU_CACHE_DIRECTORY = "parakeet-litert-gpu-cache"
        private val DURATIONS = intArrayOf(0, 1, 2, 3, 4)

        fun open(context: Context, backend: WhisperBackendRequest): LiteRtParakeetEngine {
            val applicationContext = context.applicationContext
            val files = ParakeetModelFiles.verified(applicationContext)
            val tokenizer = applicationContext.assets.open(ParakeetAssets.TOKENIZER).use(ParakeetTokenizer::load)
            val cpu = CompiledModel.Options(Accelerator.CPU).apply {
                cpuOptions = CompiledModel.CpuOptions(CPU_THREADS, null, null)
            }
            val decoderModel = CompiledModel.create(files.decoder.absolutePath, cpu)
            try {
                var usesGpu = false
                val encoderModel = if (backend == WhisperBackendRequest.GPU) {
                    runCatching {
                        CompiledModel.create(files.encoder.absolutePath, gpuOptions(applicationContext, files.encoder))
                    }.onFailure {
                        Log.w(TAG, "Parakeet GPU encoder could not start; using the CPU.", it)
                    }.getOrNull()?.also { usesGpu = true }
                } else {
                    null
                } ?: CompiledModel.create(files.encoder.absolutePath, cpu)
                return try {
                    LiteRtParakeetEngine(encoderModel, decoderModel, tokenizer, usesGpu)
                } catch (failure: Throwable) {
                    encoderModel.close()
                    throw failure
                }
            } catch (failure: Throwable) {
                decoderModel.close()
                throw failure
            }
        }

        /**
         * FP16 precision executes as F32_F16 with the patched LiteRT runtime. The compiled GPU
         * programs are cached (~160 MB), which cuts the load from ~26 s to ~15 s.
         */
        private fun gpuOptions(context: Context, encoder: File) =
            CompiledModel.Options(Accelerator.GPU, Accelerator.CPU).apply {
                val cache = File(context.cacheDir, GPU_CACHE_DIRECTORY).apply { mkdirs() }
                gpuOptions = CompiledModel.GpuOptions(
                    precision = CompiledModel.GpuOptions.Precision.FP16,
                    constantTensorSharing = true,
                    serializationDir = cache.absolutePath,
                    modelCacheKey = "${encoder.name}-${encoder.length()}-fp16",
                    serializeProgramCache = true,
                )
            }
    }
}

/** Opens the Parakeet engine for [WhisperEngineCache] and reports which backend it got. */
internal fun openParakeetEngineSet(
    context: Context,
    requestedBackend: WhisperBackendRequest,
): WhisperEngineCache.EngineSet {
    val engine = LiteRtParakeetEngine.open(context, requestedBackend)
    val selection = when {
        requestedBackend == WhisperBackendRequest.CPU -> WhisperBackendSelection.cpuRequested()
        engine.usesGpu -> WhisperBackendSelection.gpuSelected()
        else -> WhisperBackendSelection.gpuFallback(
            actual = WhisperRuntimeBackend.CPU_XNNPACK,
            stage = WhisperBackendFallbackStage.OPEN,
            reason = "The Parakeet GPU encoder could not start; using CPU/XNNPACK.",
        )
    }
    return WhisperEngineCache.EngineSet(listOf(engine), selection)
}
