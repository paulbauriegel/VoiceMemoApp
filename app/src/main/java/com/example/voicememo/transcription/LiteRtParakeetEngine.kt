package com.example.voicememo.transcription

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import java.util.concurrent.atomic.AtomicBoolean

/**
 * LiteRT implementation of the pinned stateful Parakeet TDT graph.
 *
 * The decode loop follows Google's LiteRT speech-recognition sample: it begins
 * with the 64-token stateless signature, then switches to `decode_1` while
 * swapping the recurrent state buffers.
 */
internal class LiteRtParakeetEngine private constructor(
    private val model: CompiledModel,
    private val tokenizer: ParakeetTokenizer,
    private val featureExtractor: ParakeetFeatureExtractor,
    private val encodeInputs: List<TensorBuffer>,
    private val encodeOutputs: List<TensorBuffer>,
    private val decodeInputs: List<TensorBuffer>,
    private val decodeOutputs: List<TensorBuffer>,
    private val decodeOneInputs: List<TensorBuffer>,
    private val decodeOneOutputs: List<TensorBuffer>,
) : WhisperEngine {
    private val closed = AtomicBoolean(false)
    private val inferenceLock = Any()

    override fun transcribeChunk(
        audio: FloatArray,
        language: WhisperLanguage,
        shouldCancel: () -> Boolean,
        onTokenProgress: (generated: Int, limit: Int) -> Unit,
    ): WhisperChunkResult = synchronized(inferenceLock) {
        check(!closed.get()) { "Parakeet engine is closed." }
        require(audio.size <= ParakeetAssets.MAX_SAMPLES) {
            "Parakeet accepts at most five seconds of 16 kHz PCM."
        }
        checkCancellation(shouldCancel)
        val started = System.nanoTime()

        encodeInputs[0].writeFloat(featureExtractor.extract(audio))
        model.run(encodeInputs, encodeOutputs, ENCODE_SIGNATURE)
        checkCancellation(shouldCancel)

        val decoded = decode(shouldCancel, onTokenProgress)
        val tokenIds = decoded.map(DecodedToken::id)
        val text = tokenizer.decode(tokenIds)
        WhisperChunkResult(
            tokens = tokenIds,
            text = text,
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
            timedSegments = decoded.map { token ->
                WhisperTimedSegment(
                    startMillis = token.timeIndex * ParakeetAssets.INPUT_SECONDS * 1_000L /
                        ENCODED_TIME_STEPS,
                    endMillis = (token.timeIndex + 1) *
                        ParakeetAssets.INPUT_SECONDS * 1_000L / ENCODED_TIME_STEPS,
                    text = tokenizer.decode(listOf(token.id)),
                )
            },
        )
    }

    override fun decodeTokens(tokens: List<Int>): String = tokenizer.decode(tokens)

    override fun close() {
        if (closed.compareAndSet(false, true)) {
            (
                encodeInputs + encodeOutputs + decodeInputs + decodeOutputs +
                    decodeOneInputs + decodeOneOutputs
                ).distinct().asReversed().forEach { buffer ->
                runCatching(buffer::close)
            }
            model.close()
        }
    }

    private fun decode(
        shouldCancel: () -> Boolean,
        onTokenProgress: (generated: Int, limit: Int) -> Unit,
    ): List<DecodedToken> {
        val maximumTimeIndex = encodeOutputs[0].readFloat().size / ENCODED_FEATURES
        val tokenSlots = decodeInputs[1].readInt().size
        val logitsPerToken =
            decodeOutputs[0].readFloat().size / tokenSlots / maximumTimeIndex
        require(logitsPerToken == ParakeetAssets.BLANK_TOKEN_ID + 1 + DURATION_COUNT) {
            "Unexpected Parakeet logits width: $logitsPerToken."
        }

        val stateSize = decodeInputs[2].readFloat().size
        val zeroState = FloatArray(stateSize)
        val stateBuffers = listOf(
            listOf(decodeInputs[2], decodeInputs[3]),
            listOf(decodeOutputs[1], decodeOutputs[2]),
        )
        var inputStateIndex = 0
        stateBuffers[inputStateIndex].forEach { it.writeFloat(zeroState) }

        var signature = DECODE_SIGNATURE
        var inferenceTokenSlots = tokenSlots
        var tokenBuffer = decodeInputs[1]
        var tokenInput = IntArray(tokenSlots)
        tokenInput[0] = ParakeetAssets.BLANK_TOKEN_ID
        var logitsBuffer = decodeOutputs[0]
        var tokenIndex = 0
        var timeIndex = 0
        val generated = mutableListOf<DecodedToken>()

        while (timeIndex < maximumTimeIndex) {
            checkCancellation(shouldCancel)
            tokenBuffer.writeInt(tokenInput)
            model.run(
                buildList {
                    addAll(encodeOutputs)
                    add(tokenBuffer)
                    addAll(stateBuffers[inputStateIndex])
                },
                listOf(logitsBuffer) + stateBuffers[1 - inputStateIndex],
                signature,
            )
            checkCancellation(shouldCancel)

            val logits = logitsBuffer.readFloat()
            val rowStart =
                timeIndex * inferenceTokenSlots * logitsPerToken + tokenIndex * logitsPerToken
            val tokenEnd = rowStart + logitsPerToken - DURATION_COUNT
            val rowEnd = rowStart + logitsPerToken
            val tokenId = argmax(logits, rowStart, tokenEnd) - rowStart
            if (tokenId != ParakeetAssets.BLANK_TOKEN_ID) {
                generated += DecodedToken(tokenId, timeIndex)
                onTokenProgress(generated.size, TOKEN_PROGRESS_LIMIT)
                if (inferenceTokenSlots > 1) {
                    tokenIndex++
                    if (tokenIndex == inferenceTokenSlots - 1) {
                        signature = DECODE_ONE_SIGNATURE
                        inferenceTokenSlots = 1
                        tokenBuffer = decodeOneInputs[1]
                        tokenInput = IntArray(1)
                        logitsBuffer = decodeOneOutputs[0]
                        tokenIndex = 0
                    }
                }
                tokenInput[tokenIndex] = tokenId
            }

            val duration = argmax(logits, tokenEnd, rowEnd) - tokenEnd
            timeIndex += if (duration == 0 && tokenId == ParakeetAssets.BLANK_TOKEN_ID) {
                1
            } else {
                duration
            }
            if (inferenceTokenSlots == 1) inputStateIndex = 1 - inputStateIndex
        }
        return generated
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

    private data class DecodedToken(
        val id: Int,
        val timeIndex: Int,
    )

    companion object {
        private const val ENCODE_SIGNATURE = "encode"
        private const val DECODE_SIGNATURE = "decode"
        private const val DECODE_ONE_SIGNATURE = "decode_1"
        private const val ENCODED_FEATURES = 1_024
        private const val ENCODED_TIME_STEPS = 63
        private const val DURATION_COUNT = 5
        private const val TOKEN_PROGRESS_LIMIT = 512

        fun open(
            context: Context,
            backend: WhisperBackendRequest,
        ): LiteRtParakeetEngine {
            val applicationContext = context.applicationContext
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
            val model = CompiledModel.create(
                applicationContext.assets,
                ParakeetAssets.MODEL,
                options,
            )
            return try {
                LiteRtParakeetEngine(
                    model = model,
                    tokenizer = applicationContext.assets.open(ParakeetAssets.TOKENIZER).use(
                        ParakeetTokenizer::load,
                    ),
                    featureExtractor = ParakeetFeatureExtractor(),
                    encodeInputs = model.createInputBuffers(ENCODE_SIGNATURE),
                    encodeOutputs = model.createOutputBuffers(ENCODE_SIGNATURE),
                    decodeInputs = model.createInputBuffers(DECODE_SIGNATURE),
                    decodeOutputs = model.createOutputBuffers(DECODE_SIGNATURE),
                    decodeOneInputs = model.createInputBuffers(DECODE_ONE_SIGNATURE),
                    decodeOneOutputs = model.createOutputBuffers(DECODE_ONE_SIGNATURE),
                )
            } catch (failure: Throwable) {
                model.close()
                throw failure
            }
        }
    }
}
