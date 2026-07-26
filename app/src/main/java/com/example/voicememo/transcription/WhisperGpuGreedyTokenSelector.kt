package com.example.voicememo.transcription

/**
 * Greedy selection for the GPU decoder.
 *
 * Keeping this separate from the LiteRT lifecycle makes the numerical safety
 * rules testable without allocating the 755 MB model. Non-finite rows are a
 * backend failure, not EOS: the runtime fallback can then retry the same audio
 * chunk with the CPU engine instead of accepting a silently truncated result.
 */
internal object WhisperGpuGreedyTokenSelector {
    fun selectLanguageToken(
        logits: FloatArray,
        position: Int,
        vocabularySize: Int,
        languageTokenIds: Iterable<Int>,
    ): Int {
        var selected = -1
        var selectedScore = Float.NEGATIVE_INFINITY
        for (tokenId in languageTokenIds) {
            val score = logit(logits, position, tokenId, vocabularySize)
            if (score.isFinite() && score > selectedScore) {
                selected = tokenId
                selectedScore = score
            }
        }
        if (selected < 0) {
            throw WhisperGpuBackendException(
                "GPU decoder produced no finite language logits.",
            )
        }
        return selected
    }

    fun selectTextToken(
        logits: FloatArray,
        position: Int,
        vocabularySize: Int,
        decoderStartTokenId: Int,
        eosTokenId: Int,
        suppressTokens: Set<Int>,
        beginSuppressTokens: Set<Int>,
        firstGeneratedToken: Boolean,
    ): Int {
        var selected = -1
        var selectedScore = Float.NEGATIVE_INFINITY
        for (tokenId in 0 until vocabularySize) {
            val isSpecial = tokenId >= decoderStartTokenId && tokenId != eosTokenId
            val isConfiguredSuppressed = tokenId in suppressTokens
            val isBeginningSuppressed =
                firstGeneratedToken && tokenId in beginSuppressTokens
            if (isSpecial || isConfiguredSuppressed || isBeginningSuppressed) continue

            val score = logit(logits, position, tokenId, vocabularySize)
            if (score.isFinite() && score > selectedScore) {
                selected = tokenId
                selectedScore = score
            }
        }
        if (selected < 0) {
            throw WhisperGpuBackendException(
                "GPU decoder produced no finite text logits at position $position.",
            )
        }
        return selected
    }

    private fun logit(
        logits: FloatArray,
        position: Int,
        token: Int,
        vocabularySize: Int,
    ): Float {
        val floatIndex = position * vocabularySize + token
        if (floatIndex !in logits.indices) {
            throw WhisperGpuBackendException(
                "GPU decoder returned ${logits.size} logits; expected at least " +
                    "${(position + 1) * vocabularySize}.",
            )
        }
        return logits[floatIndex]
    }
}
