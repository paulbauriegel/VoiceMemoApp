package com.example.voicememo.ui

import org.junit.Assert.assertEquals
import org.junit.Test

class SpeakerLineJoiningTest {
    private fun line(startSeconds: Int, endSeconds: Int, speaker: SpeakerUi?, text: String = "and then") =
        TranscriptSegmentUi(
            id = "segment-$startSeconds",
            startMillis = startSeconds * 1_000L,
            endMillis = endSeconds * 1_000L,
            startLabel = "",
            endLabel = "",
            text = text,
            speaker = speaker,
        )

    @Test
    fun joinsConsecutiveLinesOfOneSpeakerAndSplitsWhereTheSpeakerChanges() {
        val lines = listOf(
            line(0, 2, SpeakerUi.Me),
            line(2, 4, SpeakerUi.Me),
            line(4, 6, SpeakerUi.Other),
            line(6, 8, SpeakerUi.Other),
            line(8, 9, SpeakerUi.Me),
        )

        assertEquals(listOf(0..1, 2..3, 4..4), joinSpeakerLines(lines))
    }

    @Test
    fun unknownAndMixedLinesStayOnTheirOwn() {
        val lines = listOf(
            line(0, 2, null),
            line(2, 4, null),
            line(4, 6, SpeakerUi.Mixed),
            line(6, 8, SpeakerUi.Mixed),
        )

        assertEquals(listOf(0..0, 1..1, 2..2, 3..3), joinSpeakerLines(lines))
    }

    @Test
    fun closesAJoinedLineAtASentenceEndOnceItIsTenSecondsLong() {
        val lines = listOf(
            line(0, 4, SpeakerUi.Me, "A short sentence."),
            line(4, 11, SpeakerUi.Me, "It goes on until here.”"),
            line(11, 13, SpeakerUi.Me, "Next thought"),
        )

        assertEquals(listOf(0..1, 2..2), joinSpeakerLines(lines))
    }

    @Test
    fun neverJoinsPastTwentySeconds() {
        val lines = listOf(
            line(0, 9, SpeakerUi.Other),
            line(9, 18, SpeakerUi.Other),
            line(18, 21, SpeakerUi.Other),
            line(21, 24, SpeakerUi.Other),
        )

        assertEquals(listOf(0..1, 2..3), joinSpeakerLines(lines))
    }

    @Test
    fun aLongPauseKeepsLinesApart() {
        val lines = listOf(line(0, 2, SpeakerUi.Me), line(5, 7, SpeakerUi.Me))

        assertEquals(listOf(0..0, 1..1), joinSpeakerLines(lines))
    }

    @Test
    fun coversEveryLineOnceAndHandlesAnEmptyTranscript() {
        assertEquals(emptyList<IntRange>(), joinSpeakerLines(emptyList()))
        val lines = List(30) { line(it * 3, it * 3 + 3, if (it % 7 < 4) SpeakerUi.Me else SpeakerUi.Other) }

        assertEquals(lines.indices.toList(), joinSpeakerLines(lines).flatten())
    }
}
