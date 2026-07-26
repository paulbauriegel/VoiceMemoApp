package com.example.voicememo.transcription

import org.junit.Assert.assertEquals
import org.junit.Assert.fail
import org.junit.Test

class WhisperGpuGreedyTokenSelectorTest {
    @Test
    fun textSelection_appliesSuppressionAndKeepsLowestTokenOnTie() {
        val logits = logits(
            position = 2,
            vocabularySize = 8,
            scores = mapOf(
                0 to 7.0f,
                1 to 100.0f,
                2 to 90.0f,
                3 to 7.0f,
                5 to 6.0f,
                7 to 80.0f,
            ),
        )

        val selected = WhisperGpuGreedyTokenSelector.selectTextToken(
            logits = logits,
            position = 2,
            vocabularySize = 8,
            decoderStartTokenId = 6,
            eosTokenId = 5,
            suppressTokens = setOf(1),
            beginSuppressTokens = setOf(2),
            firstGeneratedToken = true,
        )

        assertEquals(0, selected)
    }

    @Test
    fun textSelection_onlyAppliesBeginningSuppressionToFirstToken() {
        val logits = logits(
            position = 0,
            vocabularySize = 8,
            scores = mapOf(0 to 1.0f, 2 to 9.0f),
        )

        val selected = WhisperGpuGreedyTokenSelector.selectTextToken(
            logits = logits,
            position = 0,
            vocabularySize = 8,
            decoderStartTokenId = 6,
            eosTokenId = 5,
            suppressTokens = emptySet(),
            beginSuppressTokens = setOf(2),
            firstGeneratedToken = false,
        )

        assertEquals(2, selected)
    }

    @Test
    fun languageSelection_ignoresNonFiniteScores() {
        val logits = logits(
            position = 1,
            vocabularySize = 6,
            scores = mapOf(
                1 to Float.NaN,
                2 to Float.POSITIVE_INFINITY,
                3 to 4.0f,
                4 to 3.0f,
            ),
        )

        assertEquals(
            3,
            WhisperGpuGreedyTokenSelector.selectLanguageToken(
                logits = logits,
                position = 1,
                vocabularySize = 6,
                languageTokenIds = listOf(1, 2, 3, 4),
            ),
        )
    }

    @Test
    fun allNonFiniteTextScores_throwInsteadOfSilentlyReturningEos() {
        val logits = floatArrayOf(
            Float.NaN,
            Float.POSITIVE_INFINITY,
            Float.NEGATIVE_INFINITY,
            Float.NaN,
        )

        assertGpuFailure {
            WhisperGpuGreedyTokenSelector.selectTextToken(
                logits = logits,
                position = 0,
                vocabularySize = 4,
                decoderStartTokenId = 3,
                eosTokenId = 2,
                suppressTokens = emptySet(),
                beginSuppressTokens = emptySet(),
                firstGeneratedToken = true,
            )
        }
    }

    @Test
    fun allNonFiniteLanguageScores_throwInsteadOfSelectingUnknownLanguage() {
        val logits = floatArrayOf(
            0.0f,
            Float.NaN,
            Float.POSITIVE_INFINITY,
            Float.NEGATIVE_INFINITY,
        )

        assertGpuFailure {
            WhisperGpuGreedyTokenSelector.selectLanguageToken(
                logits = logits,
                position = 0,
                vocabularySize = 4,
                languageTokenIds = listOf(1, 2, 3),
            )
        }
    }

    private fun logits(
        position: Int,
        vocabularySize: Int,
        scores: Map<Int, Float>,
    ): FloatArray {
        val values = FloatArray((position + 1) * vocabularySize) { -100.0f }
        scores.forEach { (token, score) ->
            values[position * vocabularySize + token] = score
        }
        return values
    }

    private fun assertGpuFailure(block: () -> Unit) {
        try {
            block()
            fail("Expected an invalid GPU-logit failure.")
        } catch (_: WhisperGpuBackendException) {
            // Expected: the runtime wrapper retries the same chunk on CPU.
        }
    }
}
