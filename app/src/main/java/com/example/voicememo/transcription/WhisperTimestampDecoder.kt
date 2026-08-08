package com.example.voicememo.transcription

import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.roundToInt

/**
 * Timestamp-token rules from OpenAI Whisper's ApplyTimestampRules filter.
 *
 * The pinned large-v3 Turbo vocabulary starts timestamps immediately after
 * <|notimestamps|>. Each timestamp step represents 20 ms within the current
 * 30-second inference window.
 */
internal object WhisperTimestampTokenSelector {
    fun select(
        vocabularySize: Int,
        generatedTokens: List<Int>,
        config: WhisperGenerationConfig,
        firstGeneratedToken: Boolean,
        logitAt: (tokenId: Int) -> Float,
        failure: (String) -> RuntimeException = { WhisperModelContractException(it) },
    ): Int {
        val timestampBegin = config.timestampBeginTokenId
        val lastWasTimestamp = generatedTokens.lastOrNull()?.let { it >= timestampBegin } == true
        val penultimateWasTimestamp =
            generatedTokens.size < 2 || generatedTokens[generatedTokens.lastIndex - 1] >= timestampBegin
        val timestamps = generatedTokens.filter { it >= timestampBegin }
        val minimumTimestamp = timestamps.lastOrNull()?.let { lastTimestamp ->
            if (lastWasTimestamp && !penultimateWasTimestamp) {
                lastTimestamp
            } else {
                lastTimestamp + 1
            }
        }

        fun isAllowed(tokenId: Int): Boolean {
            if (tokenId !in 0 until vocabularySize) return false
            if (tokenId == config.noTimestampsTokenId) return false
            if (tokenId in config.suppressTokens) return false
            if (firstGeneratedToken && tokenId in config.beginSuppressTokens) return false

            val isTimestamp = tokenId >= timestampBegin
            val isOtherSpecial =
                tokenId >= config.decoderStartTokenId &&
                    tokenId < timestampBegin &&
                    tokenId != config.eosTokenId
            if (isOtherSpecial) return false

            if (firstGeneratedToken) {
                if (!isTimestamp) return false
                val lastAllowed = config.maxInitialTimestampIndex?.let(timestampBegin::plus)
                if (lastAllowed != null && tokenId > lastAllowed) return false
            }

            if (lastWasTimestamp) {
                if (penultimateWasTimestamp && isTimestamp) return false
                if (!penultimateWasTimestamp && !isTimestamp && tokenId != config.eosTokenId) {
                    return false
                }
            }
            if (isTimestamp && minimumTimestamp != null && tokenId < minimumTimestamp) {
                return false
            }
            return true
        }

        var bestTextToken = -1
        var bestTextScore = Float.NEGATIVE_INFINITY
        var bestTimestampToken = -1
        var bestTimestampScore = Float.NEGATIVE_INFINITY
        for (tokenId in 0 until vocabularySize) {
            if (!isAllowed(tokenId)) continue
            val score = logitAt(tokenId)
            if (!score.isFinite()) continue
            if (tokenId >= timestampBegin) {
                if (score > bestTimestampScore) {
                    bestTimestampToken = tokenId
                    bestTimestampScore = score
                }
            } else if (score > bestTextScore) {
                bestTextToken = tokenId
                bestTextScore = score
            }
        }

        if (bestTextToken < 0 && bestTimestampToken < 0) {
            throw failure("Decoder produced no finite timestamp-aware logits.")
        }

        val timestampLogProbabilityMass = if (bestTimestampToken >= 0) {
            var scaledProbabilitySum = 0.0
            for (tokenId in timestampBegin until vocabularySize) {
                if (!isAllowed(tokenId)) continue
                val score = logitAt(tokenId)
                if (score.isFinite()) {
                    scaledProbabilitySum += exp((score - bestTimestampScore).toDouble())
                }
            }
            bestTimestampScore.toDouble() + ln(scaledProbabilitySum)
        } else {
            Double.NEGATIVE_INFINITY
        }

        // Normalization cancels from both sides, so raw-logit log-sum-exp is
        // equivalent to comparing the normalized log probabilities.
        if (
            bestTimestampToken >= 0 &&
            timestampLogProbabilityMass > bestTextScore.toDouble()
        ) {
            return bestTimestampToken
        }
        return when {
            bestTextToken < 0 -> bestTimestampToken
            bestTimestampToken < 0 -> bestTextToken
            bestTimestampScore > bestTextScore -> bestTimestampToken
            else -> bestTextToken
        }
    }
}

internal data class WhisperSeekAdvance(
    /** Tokens to keep. An unfinished trailing segment is excluded. */
    val keptTokenCount: Int,
    /**
     * Position inside the decoded audio where generation stopped and decoding
     * must resume, or null when the decoder consumed the whole payload.
     */
    val advanceMillis: Long?,
)

internal object WhisperTimestampTokens {
    const val DEFAULT_TIMESTAMP_BEGIN_TOKEN_ID = 50_365
    const val MILLISECONDS_PER_TOKEN = 20L
    private val timestampMarkerPattern = Regex("""<\|\d{1,3}\.\d{2}\|>""")

    fun containsTimestamp(tokens: List<Int>, timestampBeginTokenId: Int): Boolean =
        tokens.any { it >= timestampBeginTokenId }

    fun withoutTimestamps(
        tokens: List<Int>,
        timestampBeginTokenId: Int,
    ): List<Int> = tokens.filter { it < timestampBeginTokenId }

    fun withoutTimestampMarkers(text: String): String =
        timestampMarkerPattern.replace(text, "").trim()

    fun shift(
        tokens: List<Int>,
        timestampBeginTokenId: Int,
        offsetMillis: Long,
    ): List<Int> {
        val tokenOffset = (offsetMillis.toDouble() / MILLISECONDS_PER_TOKEN)
            .roundToInt()
        return tokens.map { token ->
            if (token >= timestampBeginTokenId) token + tokenOffset else token
        }
    }

    /**
     * OpenAI Whisper's long-form seek rule. The timestamp grammar allows EOS
     * after any closed segment, so an early stop is legitimate decoder output:
     * a single trailing timestamp means the payload was fully consumed, while
     * a decode whose last closed segment ends in a consecutive timestamp pair
     * stopped at that pair — tokens after it belong to an unfinished segment
     * and decoding must resume at the pair's timestamp.
     */
    fun seekAdvance(tokens: List<Int>, timestampBeginTokenId: Int): WhisperSeekAdvance {
        val fullyConsumed = WhisperSeekAdvance(tokens.size, null)
        if (tokens.size < 2) return fullyConsumed
        val singleTimestampEnding =
            tokens[tokens.size - 2] < timestampBeginTokenId &&
                tokens[tokens.size - 1] >= timestampBeginTokenId
        if (singleTimestampEnding) return fullyConsumed
        var lastPairSecondIndex = -1
        for (index in 0 until tokens.size - 1) {
            if (
                tokens[index] >= timestampBeginTokenId &&
                tokens[index + 1] >= timestampBeginTokenId
            ) {
                lastPairSecondIndex = index + 1
            }
        }
        if (lastPairSecondIndex < 0) return fullyConsumed
        val advanceMillis =
            (tokens[lastPairSecondIndex - 1] - timestampBeginTokenId).toLong() *
                MILLISECONDS_PER_TOKEN
        return WhisperSeekAdvance(
            keptTokenCount = lastPairSecondIndex,
            advanceMillis = advanceMillis,
        )
    }

    fun parseSegments(
        tokens: List<Int>,
        timestampBeginTokenId: Int,
        tokenizer: WhisperTokenizer,
        audioDurationMillis: Long,
    ): List<WhisperTimedSegment> {
        val result = mutableListOf<WhisperTimedSegment>()
        val textTokens = mutableListOf<Int>()
        var startMillis: Long? = null

        fun timestampMillis(token: Int): Long =
            (token - timestampBeginTokenId).toLong() * MILLISECONDS_PER_TOKEN

        fun closeSegment(endMillis: Long) {
            val start = startMillis ?: return
            val text = tokenizer.decode(textTokens).trim()
            val boundedStart = start.coerceIn(0L, audioDurationMillis)
            val boundedEnd = endMillis.coerceIn(boundedStart, audioDurationMillis)
            if (text.isNotEmpty() && boundedEnd > boundedStart) {
                result += WhisperTimedSegment(
                    startMillis = boundedStart,
                    endMillis = boundedEnd,
                    text = text,
                )
            }
            textTokens.clear()
        }

        for (token in tokens) {
            if (token >= timestampBeginTokenId) {
                val timestamp = timestampMillis(token)
                if (startMillis == null) {
                    startMillis = timestamp
                } else if (textTokens.isNotEmpty()) {
                    closeSegment(timestamp)
                    startMillis = timestamp
                } else {
                    // The second token in a consecutive timestamp pair is the
                    // start boundary for the following segment.
                    startMillis = timestamp
                }
            } else if (startMillis != null) {
                textTokens += token
            }
        }
        if (textTokens.isNotEmpty()) {
            closeSegment(audioDurationMillis)
        }
        return result
    }
}
