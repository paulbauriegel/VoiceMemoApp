package com.example.voicememo.transcription

import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Benchmark-only implementation of the pinned Qwen3-ASR 0.6B LiteRT graph.
 *
 * The model stays outside the production APK and is loaded from an explicitly
 * supplied app-private cache path.
 */
internal class LiteRtQwenAsrBenchmarkEngine private constructor(
    private val model: CompiledModel,
    private val tokenizer: WhisperTokenizer,
    private val featureExtractor: QwenAsrFeatureExtractor,
    private val encodeInputs: List<TensorBuffer>,
    private val encodeOutputs: List<TensorBuffer>,
    private val decodeInputs: List<TensorBuffer>,
    private val decodeOutputs: List<TensorBuffer>,
) : WhisperEngine {
    private val closed = AtomicBoolean(false)

    override fun transcribeChunk(
        audio: FloatArray,
        language: WhisperLanguage,
        shouldCancel: () -> Boolean,
        onTokenProgress: (generated: Int, limit: Int) -> Unit,
    ): WhisperChunkResult {
        check(!closed.get()) { "Qwen3-ASR benchmark engine is closed." }
        require(audio.size <= QwenAsrFeatureExtractor.MAX_SAMPLES)
        checkCancellation(shouldCancel)
        val started = System.nanoTime()

        encodeInputs[0].writeFloat(featureExtractor.extract(audio))
        model.run(encodeInputs, encodeOutputs, ENCODE_SIGNATURE)
        checkCancellation(shouldCancel)

        val tokens = IntArray(MAX_DECODE_TOKENS)
        tokens[0] = DECODE_START_TOKEN_ID
        var skipPrompt = true
        val generated = mutableListOf<Int>()
        for (index in 0 until MAX_DECODE_TOKENS - 1) {
            checkCancellation(shouldCancel)
            decodeInputs[TOKEN_INPUT_INDEX].writeInt(tokens)
            model.run(
                listOf(
                    encodeOutputs[0],
                    decodeInputs[TOKEN_INPUT_INDEX],
                    decodeInputs[MASK_INPUT_INDEX],
                ),
                decodeOutputs,
                DECODE_SIGNATURE,
            )
            val logits = decodeOutputs[0].readFloat()
            val rowStart = index * VOCABULARY_SIZE
            val token = argmax(logits, rowStart, rowStart + VOCABULARY_SIZE) - rowStart
            if (token == DECODE_STOP_TOKEN_ID) break
            tokens[index + 1] = token
            if (skipPrompt) {
                skipPrompt = token != DECODE_SKIP_UNTIL_TOKEN_ID
            } else {
                generated += token
                onTokenProgress(generated.size, MAX_DECODE_TOKENS - 1)
            }
        }

        return WhisperChunkResult(
            tokens = generated,
            text = tokenizer.decode(generated),
            detectedLanguage = (language as? WhisperLanguage.Forced)?.code ?: "und",
            inferenceMillis = (System.nanoTime() - started) / 1_000_000L,
            truncated = tokens.last() != 0 && tokens.last() != DECODE_STOP_TOKEN_ID,
            warnings = if (language is WhisperLanguage.Forced) {
                listOf(
                    "Qwen3-ASR detects language automatically; the selected language is " +
                        "retained only as benchmark metadata.",
                )
            } else {
                emptyList()
            },
        )
    }

    override fun decodeTokens(tokens: List<Int>): String = tokenizer.decode(tokens)

    override fun close() {
        if (closed.compareAndSet(false, true)) {
            (encodeInputs + encodeOutputs + decodeInputs + decodeOutputs)
                .distinct()
                .asReversed()
                .forEach { runCatching(it::close) }
            model.close()
        }
    }

    private fun checkCancellation(shouldCancel: () -> Boolean) {
        if (closed.get() || shouldCancel()) throw WhisperCancelledException()
    }

    private fun argmax(values: FloatArray, start: Int, endExclusive: Int): Int {
        var best = start
        for (index in start + 1 until endExclusive) {
            if (values[index] > values[best]) best = index
        }
        return best
    }

    companion object {
        private const val ENCODE_SIGNATURE = "encode"
        private const val DECODE_SIGNATURE = "decode"
        private const val TOKEN_INPUT_INDEX = 1
        private const val MASK_INPUT_INDEX = 2
        private const val MAX_DECODE_TOKENS = 64
        private const val MASK_SIDE = 134
        private const val VOCABULARY_SIZE = 151_936
        private const val DECODE_START_TOKEN_ID = 198
        private const val DECODE_STOP_TOKEN_ID = 151_645
        private const val DECODE_SKIP_UNTIL_TOKEN_ID = 151_704

        fun open(
            modelFile: File,
            tokenizerFile: File,
            backend: WhisperBackendRequest,
        ): LiteRtQwenAsrBenchmarkEngine {
            require(modelFile.isFile) { "Missing Qwen model: $modelFile" }
            require(tokenizerFile.isFile) { "Missing Qwen tokenizer: $tokenizerFile" }
            val options = when (backend) {
                WhisperBackendRequest.CPU -> CompiledModel.Options(Accelerator.CPU)
                WhisperBackendRequest.GPU ->
                    CompiledModel.Options(Accelerator.GPU, Accelerator.CPU).apply {
                        gpuOptions = CompiledModel.GpuOptions(
                            precision = CompiledModel.GpuOptions.Precision.FP32,
                            constantTensorSharing = true,
                        )
                    }
            }
            val model = CompiledModel.create(modelFile.absolutePath, options)
            return try {
                val decodeInputs = model.createInputBuffers(DECODE_SIGNATURE)
                decodeInputs[MASK_INPUT_INDEX].writeFloat(causalMask())
                LiteRtQwenAsrBenchmarkEngine(
                    model = model,
                    tokenizer = tokenizerFile.inputStream().buffered().use(WhisperTokenizer::load),
                    featureExtractor = QwenAsrFeatureExtractor(),
                    encodeInputs = model.createInputBuffers(ENCODE_SIGNATURE),
                    encodeOutputs = model.createOutputBuffers(ENCODE_SIGNATURE),
                    decodeInputs = decodeInputs,
                    decodeOutputs = model.createOutputBuffers(DECODE_SIGNATURE),
                )
            } catch (failure: Throwable) {
                model.close()
                throw failure
            }
        }

        private fun causalMask(): FloatArray {
            val masked = -0.7f * Float.MAX_VALUE
            return FloatArray(MASK_SIDE * MASK_SIDE) { index ->
                val row = index / MASK_SIDE
                val column = index % MASK_SIDE
                if (column <= row) 0.0f else masked
            }
        }
    }
}
