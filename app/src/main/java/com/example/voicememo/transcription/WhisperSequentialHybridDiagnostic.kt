package com.example.voicememo.transcription

import java.security.MessageDigest

/**
 * Pure comparison logic for the diagnostic GPU-encoder/CPU-decoder lane.
 *
 * This type intentionally accepts token IDs rather than [WhisperChunkResult],
 * so transcript text cannot accidentally be copied into a parity report.
 */
internal object WhisperSequentialHybridDiagnostic {
    fun compare(
        referenceTokens: List<Int>,
        referenceTruncated: Boolean,
        candidateTokens: List<Int>,
        candidateTruncated: Boolean,
        timings: Timings,
    ): Result {
        val copiedTokens = candidateTokens.toList()
        val firstMismatch = firstMismatch(referenceTokens, copiedTokens)
        val exactTokenMatch = firstMismatch == null
        val exactTerminationMatch = referenceTruncated == candidateTruncated
        return Result(
            tokenIds = copiedTokens,
            tokenSha256 = tokenDigest(copiedTokens),
            firstTokenMismatch = firstMismatch,
            exactTokenMatch = exactTokenMatch,
            truncated = candidateTruncated,
            exactTerminationMatch = exactTerminationMatch,
            exactCpuResultMatch = exactTokenMatch && exactTerminationMatch,
            timings = timings,
        )
    }

    private fun firstMismatch(reference: List<Int>, candidate: List<Int>): Int? {
        repeat(minOf(reference.size, candidate.size)) { index ->
            if (reference[index] != candidate[index]) return index
        }
        return if (reference.size == candidate.size) {
            null
        } else {
            minOf(reference.size, candidate.size)
        }
    }

    private fun tokenDigest(tokens: List<Int>): String {
        val bytes = tokens.joinToString(separator = ",").toByteArray(Charsets.UTF_8)
        return MessageDigest.getInstance("SHA-256")
            .digest(bytes)
            .joinToString(separator = "") { byte -> "%02x".format(byte) }
    }

    internal data class Timings(
        val gpuEncodeMillis: Long,
        val gpuModelCloseMillis: Long,
        val cpuDecoderModelLoadMillis: Long,
        val cpuDecodeMillis: Long,
    ) {
        init {
            require(gpuEncodeMillis >= 0L)
            require(gpuModelCloseMillis >= 0L)
            require(cpuDecoderModelLoadMillis >= 0L)
            require(cpuDecodeMillis >= 0L)
        }

        val inferenceComponentMillis: Long
            get() = gpuEncodeMillis + cpuDecodeMillis

        val sequentialComponentMillis: Long
            get() =
                inferenceComponentMillis +
                    gpuModelCloseMillis +
                    cpuDecoderModelLoadMillis
    }

    internal data class Result(
        val tokenIds: List<Int>,
        val tokenSha256: String,
        val firstTokenMismatch: Int?,
        val exactTokenMatch: Boolean,
        val truncated: Boolean,
        val exactTerminationMatch: Boolean,
        val exactCpuResultMatch: Boolean,
        val timings: Timings,
    )
}
