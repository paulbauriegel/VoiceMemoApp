package com.example.voicememo.transcription

/**
 * Diagnostic-only GPU proposals. The tokens are never persisted or decoded to
 * text; a reference CPU decoder must verify them before they can be accepted.
 */
internal data class WhisperGpuProposedTokenBlock(
    val tokens: List<Int>,
    val decoderInvocations: Int,
    val inferenceMillis: Long,
    /** Selectable top-1 minus top-2 logit for each proposed token. */
    val gpuSelectableTop1Margins: List<Float?> = emptyList(),
) {
    init {
        require(
            gpuSelectableTop1Margins.isEmpty() ||
                gpuSelectableTop1Margins.size == tokens.size,
        )
    }
}

/**
 * CPU selections for each causal row in one proposed block. Rows after the
 * first mismatch are intentionally ignored because they depend on an
 * unverified GPU token.
 */
internal data class WhisperCpuVerifiedTokenBlock(
    val proposedTokens: List<Int>,
    val cpuSelectedTokens: List<Int>,
    val inferenceMillis: Long,
    /** Selectable top-1 minus top-2 logit for each CPU verifier row. */
    val cpuSelectableTop1Margins: List<Float?> = emptyList(),
    /** Whether each proposal equals the CPU-selected top-1 token for that row. */
    val proposalMatchesCpuTop1: List<Boolean> = emptyList(),
) {
    init {
        require(proposedTokens.isNotEmpty())
        require(proposedTokens.size == cpuSelectedTokens.size)
        require(
            cpuSelectableTop1Margins.isEmpty() ||
                cpuSelectableTop1Margins.size == proposedTokens.size,
        )
        require(
            proposalMatchesCpuTop1.isEmpty() ||
                proposalMatchesCpuTop1.size == proposedTokens.size,
        )
    }
}

internal data class WhisperSpeculativeAcceptance(
    /** CPU-approved non-EOS tokens to append to the verified prefix. */
    val acceptedTokens: List<Int>,
    val matchingProposalCount: Int,
    val correctedToken: Int?,
    val reachedEos: Boolean,
    val allProposalsAccepted: Boolean,
)

/**
 * Applies the exactness rule for speculative greedy decoding.
 *
 * Matching GPU proposals are accepted in order. At the first mismatch, the
 * CPU-selected correction is accepted and all later rows are discarded. This
 * guarantees the resulting prefix is the same prefix a purely CPU greedy
 * decoder would have produced.
 */
internal object WhisperSpeculativeTokenAcceptance {
    fun accept(
        verification: WhisperCpuVerifiedTokenBlock,
        eosTokenId: Int,
    ): WhisperSpeculativeAcceptance {
        val accepted = mutableListOf<Int>()
        var matchingProposalCount = 0
        var correctedToken: Int? = null
        var reachedEos = false

        for (index in verification.proposedTokens.indices) {
            val proposed = verification.proposedTokens[index]
            val cpuSelected = verification.cpuSelectedTokens[index]
            if (proposed == cpuSelected) {
                matchingProposalCount++
                if (cpuSelected == eosTokenId) {
                    reachedEos = true
                    break
                }
                accepted += cpuSelected
            } else {
                correctedToken = cpuSelected
                if (cpuSelected == eosTokenId) {
                    reachedEos = true
                } else {
                    accepted += cpuSelected
                }
                return WhisperSpeculativeAcceptance(
                    acceptedTokens = accepted,
                    matchingProposalCount = matchingProposalCount,
                    correctedToken = correctedToken,
                    reachedEos = reachedEos,
                    allProposalsAccepted = false,
                )
            }
        }

        return WhisperSpeculativeAcceptance(
            acceptedTokens = accepted,
            matchingProposalCount = matchingProposalCount,
            correctedToken = correctedToken,
            reachedEos = reachedEos,
            allProposalsAccepted =
                matchingProposalCount == verification.proposedTokens.size,
        )
    }
}
