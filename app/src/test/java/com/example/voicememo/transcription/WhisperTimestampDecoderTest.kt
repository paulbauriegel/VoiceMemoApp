package com.example.voicememo.transcription

import org.junit.Assert.assertEquals
import org.junit.Test

class WhisperTimestampDecoderTest {
    private val config = WhisperGenerationConfig(
        eosTokenId = 4,
        decoderStartTokenId = 5,
        transcribeTokenId = 7,
        noTimestampsTokenId = 9,
        languageToId = mapOf("en" to 6),
        suppressTokens = emptySet(),
        beginSuppressTokens = emptySet(),
        maxInitialTimestampIndex = 2,
    )

    @Test
    fun firstGeneratedToken_isTimestampAndRespectsInitialLimit() {
        val scores = FloatArray(16) { -10f }.apply {
            this[3] = 200f
            this[12] = 5f
            this[13] = 100f
        }

        assertEquals(12, select(scores, generated = emptyList(), first = true))
    }

    @Test
    fun timestampPairsAlternateWithTextAndNeverMoveBackward() {
        val afterStart = FloatArray(16) { -10f }.apply {
            this[3] = 8f
            this[11] = 100f
        }
        assertEquals(3, select(afterStart, generated = listOf(10)))

        val afterText = FloatArray(16) { -10f }.apply {
            this[10] = 100f
            this[11] = 7f
        }
        assertEquals(11, select(afterText, generated = listOf(10, 3)))

        val afterEnd = FloatArray(16) { -10f }.apply {
            this[3] = 100f
            this[11] = 6f
            this[4] = 5f
        }
        assertEquals(11, select(afterEnd, generated = listOf(10, 3, 11)))
    }

    @Test
    fun aggregateTimestampProbabilityCanForceBoundary() {
        val scores = FloatArray(16) { -20f }.apply {
            this[3] = 0.5f
            for (token in 11 until size) this[token] = 0f
        }

        assertEquals(11, select(scores, generated = listOf(10, 3)))
    }

    @Test
    fun parserBuildsSegmentsFromConsecutiveTimestampPairs() {
        val tokenizer = WhisperTokenizer.fromTokens(
            mapOf(
                1 to "Hello",
                2 to "Ġworld",
            ),
        )

        val segments = WhisperTimestampTokens.parseSegments(
            tokens = listOf(10, 1, 12, 12, 2, 15),
            timestampBeginTokenId = 10,
            tokenizer = tokenizer,
            audioDurationMillis = 1_000,
        )

        assertEquals(
            listOf(
                WhisperTimedSegment(0, 40, "Hello"),
                WhisperTimedSegment(40, 100, "world"),
            ),
            segments,
        )
    }

    @Test
    fun shiftOffsetsOnlyTimestampTokens() {
        assertEquals(
            listOf(710, 1, 712),
            WhisperTimestampTokens.shift(
                tokens = listOf(10, 1, 12),
                timestampBeginTokenId = 10,
                offsetMillis = 14_000,
            ),
        )
    }

    @Test
    fun transcriptTextFiltersTimestampTokensAndExistingMarkers() {
        val tokenizer = WhisperTokenizer.fromTokens(
            mapOf(
                1 to "Hello",
                2 to "Ġworld",
                10 to "<|0.00|>",
                12 to "<|0.04|>",
            ),
        )
        val textOnlyTokens = WhisperTimestampTokens.withoutTimestamps(
            tokens = listOf(10, 1, 2, 12),
            timestampBeginTokenId = 10,
        )

        assertEquals("Hello world", tokenizer.decode(textOnlyTokens))
        assertEquals(
            "Hello world",
            WhisperTimestampTokens.withoutTimestampMarkers(
                "<|0.00|>Hello world<|0.04|>",
            ),
        )
    }

    private fun select(
        scores: FloatArray,
        generated: List<Int>,
        first: Boolean = false,
    ): Int = WhisperTimestampTokenSelector.select(
        vocabularySize = scores.size,
        generatedTokens = generated,
        config = config,
        firstGeneratedToken = first,
        logitAt = scores::get,
    )
}
