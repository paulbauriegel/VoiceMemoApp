package com.example.voicememo.transcription

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ParakeetTranscriptMergerTest {
    private fun word(text: String, startMillis: Long) = WhisperTimedSegment(startMillis, startMillis + 300, text)

    private fun words(startMillis: Long, vararg texts: String, stepMillis: Long = 500) =
        texts.mapIndexed { index, text -> word(text, startMillis + index * stepMillis) }

    @Test
    fun alignedRunSwitchesWindowsAtTheAnchorNearestTheOverlapMiddle() {
        // Window 0 covers 0-30 s, window 1 covers 25-55 s; the overlap middle is 27.5 s.
        val first = words(25_000, "we", "went", "to", "the", "old", "harbour", "late")
        // The newer window hears the final word differently and continues.
        val second = words(25_100, "we", "went", "to", "the", "old", "harbor", "later", "on")
        val merged = ParakeetTranscriptMerger.merge(
            listOf(
                ParakeetTranscriptMerger.Window(0, 30_000, words(0, "start") + first),
                ParakeetTranscriptMerger.Window(25_000, 55_000, second),
            ),
        )
        assertEquals("start we went to the old harbor later on", ParakeetTranscriptMerger.text(merged))
    }

    @Test
    fun withoutAnAnchorTheMergeCutsWholeWordsAtTheOverlapMiddle() {
        val merged = ParakeetTranscriptMerger.merge(
            listOf(
                ParakeetTranscriptMerger.Window(0, 30_000, listOf(word("alpha", 26_000), word("beta", 28_000))),
                ParakeetTranscriptMerger.Window(25_000, 55_000, listOf(word("gamma", 26_500), word("delta", 29_000))),
            ),
        )
        assertEquals("alpha delta", ParakeetTranscriptMerger.text(merged))
    }

    @Test
    fun anEmptyFirstWindowKeepsTheWholeNextWindow() {
        val merged = ParakeetTranscriptMerger.merge(
            listOf(
                ParakeetTranscriptMerger.Window(0, 30_000, emptyList()),
                ParakeetTranscriptMerger.Window(25_000, 55_000, listOf(word("early", 25_500), word("late", 40_000))),
            ),
        )
        assertEquals("early late", ParakeetTranscriptMerger.text(merged))
    }

    @Test
    fun normalizationIgnoresCaseAndPunctuation() {
        assertEquals("don't", ParakeetTranscriptMerger.normalize("Don't,"))
        assertEquals("", ParakeetTranscriptMerger.normalize("—"))
    }

    @Test
    fun linesBreakAtSentenceEndsAndPauses() {
        val lines = ParakeetTranscriptMerger.lines(
            listOf(
                word("Hello", 0),
                word("there.", 400),
                word("Next", 900),
                word("bit", 1_300),
                word("after", 4_000),
            ),
        )
        assertEquals(listOf("Hello there.", "Next bit", "after"), lines.map { it.text })
        assertEquals(900L, lines[1].startMillis)
    }

    @Test
    fun timedWordsSurviveTheJsonCodec() {
        val words = listOf(WhisperTimedSegment(80, 400, "Grüß"), WhisperTimedSegment(400, 720, "\"Gott\""))
        assertEquals(words, TimedWordsCodec.decode(TimedWordsCodec.encode(words)))
        assertEquals(null, TimedWordsCodec.decode(null))
    }

    @Test
    fun gapsAreTokenFreeStretchesOfAtLeastTwoSeconds() {
        val tokens = listOf(
            ParakeetTimedToken(1, 0, 80),
            ParakeetTimedToken(2, 3_000, 3_160),
            ParakeetTimedToken(3, 4_000, 4_080),
        )
        assertEquals(
            listOf(80L to 3_000L, 4_080L to 10_000L),
            ParakeetGapRepair.gaps(tokens, durationMillis = 10_000),
        )
    }

    @Test
    fun gapRepairSplicesTokensRecoveredInsideAnAudibleGap() {
        val audio = FloatArray(10 * ParakeetAssets.SAMPLE_RATE) { index -> if (index % 2 == 0) 0.1f else -0.1f }
        val initial = listOf(ParakeetTimedToken(1, 0, 80), ParakeetTimedToken(9, 9_000, 9_080))
        var request: Pair<Int, Int>? = null
        val result = ParakeetGapRepair.repair(audio, initial) { start, end ->
            request = start to end
            listOf(
                ParakeetTimedToken(5, 60, 140), // inside the gap's edge margin: dropped
                ParakeetTimedToken(6, 4_000, 4_080),
                ParakeetTimedToken(7, 6_000, 6_160),
            )
        }
        // Gap 80-9000 ms padded by 3 s on each side, clipped to the 10 s of audio.
        assertEquals(0 to 10 * ParakeetAssets.SAMPLE_RATE, request)
        assertEquals(listOf(1, 6, 7, 9), result.tokens.map { it.id })
        assertEquals(1, result.gapsChecked)
        assertEquals(1, result.gapsRepaired)
    }

    @Test
    fun silentGapsAreNotRedecoded() {
        val audio = FloatArray(10 * ParakeetAssets.SAMPLE_RATE)
        val result = ParakeetGapRepair.repair(audio, emptyList()) { _, _ -> error("not called") }
        assertTrue(result.tokens.isEmpty())
        assertEquals(0, result.gapsChecked)
    }

    @Test
    fun tokenizerGroupsMetaspacePiecesIntoTimedWords() {
        val tokenizer = ParakeetTokenizer.fromPieces(
            mapOf(1 to "▁Gr", 2 to "<0xC3>", 3 to "<0xBC>", 4 to "ß", 5 to "▁Gott", 6 to "."),
        )
        val words = tokenizer.words(
            listOf(
                ParakeetTimedToken(1, 0, 80),
                ParakeetTimedToken(2, 80, 160),
                ParakeetTimedToken(3, 80, 160),
                ParakeetTimedToken(4, 160, 240),
                ParakeetTimedToken(5, 400, 560),
                ParakeetTimedToken(6, 560, 640),
            ),
        )
        assertEquals(
            listOf(WhisperTimedSegment(0, 240, "Grüß"), WhisperTimedSegment(400, 640, "Gott.")),
            words,
        )
    }
}
