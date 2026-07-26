package com.example.voicememo.transcription

/**
 * One diagnostic decoder row plus the two tokens that the production greedy
 * policy could select. Token text is deliberately absent.
 */
internal data class WhisperTeacherForcedLogitsRow(
    val logits: FloatArray,
    val selectableTop1: WhisperLogitCandidate?,
    val selectableTop2: WhisperLogitCandidate?,
) {
    val selectableMargin: Float?
        get() {
            val first = selectableTop1?.score ?: return null
            val second = selectableTop2?.score ?: return null
            if (!first.isFinite() || !second.isFinite()) return null
            return first - second
        }
}

internal data class WhisperLogitCandidate(
    val tokenId: Int,
    val score: Float,
)

/**
 * Mirrors the eligibility and ascending-ID tie breaking used by production
 * greedy decoding while retaining the runner-up for parity diagnostics.
 */
internal object WhisperTeacherForcedTokenRanking {
    fun forCpu(
        logits: FloatArray,
        generationConfig: WhisperGenerationConfig,
        firstGeneratedToken: Boolean,
    ): WhisperTeacherForcedLogitsRow = rank(
        logits = logits,
        generationConfig = generationConfig,
        firstGeneratedToken = firstGeneratedToken,
        scoreIsUsable = { score -> !score.isNaN() },
    )

    fun forGpu(
        logits: FloatArray,
        generationConfig: WhisperGenerationConfig,
        firstGeneratedToken: Boolean,
    ): WhisperTeacherForcedLogitsRow = rank(
        logits = logits,
        generationConfig = generationConfig,
        firstGeneratedToken = firstGeneratedToken,
        scoreIsUsable = { score -> score.isFinite() },
    )

    private inline fun rank(
        logits: FloatArray,
        generationConfig: WhisperGenerationConfig,
        firstGeneratedToken: Boolean,
        scoreIsUsable: (Float) -> Boolean,
    ): WhisperTeacherForcedLogitsRow {
        var top1: WhisperLogitCandidate? = null
        var top2: WhisperLogitCandidate? = null
        logits.forEachIndexed { tokenId, score ->
            val isSpecial = tokenId >= generationConfig.decoderStartTokenId &&
                tokenId != generationConfig.eosTokenId
            val isConfiguredSuppressed = tokenId in generationConfig.suppressTokens
            val isBeginningSuppressed =
                firstGeneratedToken && tokenId in generationConfig.beginSuppressTokens
            if (
                isSpecial ||
                isConfiguredSuppressed ||
                isBeginningSuppressed ||
                !scoreIsUsable(score)
            ) {
                return@forEachIndexed
            }

            val candidate = WhisperLogitCandidate(tokenId = tokenId, score = score)
            val currentTop1 = top1
            val currentTop2 = top2
            when {
                currentTop1 == null || score > currentTop1.score -> {
                    top2 = currentTop1
                    top1 = candidate
                }

                currentTop2 == null || score > currentTop2.score -> top2 = candidate
            }
        }
        return WhisperTeacherForcedLogitsRow(
            logits = logits,
            selectableTop1 = top1,
            selectableTop2 = top2,
        )
    }
}
