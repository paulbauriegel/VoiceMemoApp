package com.example.voicememo.sharing

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class ShareTranscriptCsvTest {
    private fun memo(rows: List<ShareTranscriptRow>) = ShareableMemo(
        id = "memo",
        title = "Title",
        audioPath = "/audio.wav",
        effectiveTranscript = "Plain text",
        transcriptSegments = rows.map { it.text },
        transcriptRows = rows,
    )

    @Test
    fun timedTranscriptBecomesATableWithSpeakerAndTimes() {
        val csv = memo(
            listOf(
                ShareTranscriptRow(0, 4_200, "Me", "Guten Morgen, wie geht's?"),
                ShareTranscriptRow(4_200, 3_725_000, "Other", "Er sagte \"gut\"."),
                ShareTranscriptRow(3_725_000, 3_730_999, null, " Zeile\nzwei "),
                ShareTranscriptRow(3_731_000, 3_732_000, "Both", "   "),
            ),
        ).transcriptCsvForShare()

        assertEquals(
            "﻿start,end,speaker,text\r\n" +
                "00:00:00,00:00:04,Me,\"Guten Morgen, wie geht's?\"\r\n" +
                "00:00:04,01:02:05,Other,\"Er sagte \"\"gut\"\".\"\r\n" +
                "01:02:05,01:02:10,,\"Zeile\nzwei\"\r\n",
            csv,
        )
    }

    @Test
    fun transcriptWithoutTimedLinesStaysPlainText() {
        val memo = memo(emptyList())
        assertNull(memo.transcriptCsvForShare())
        assertEquals("Plain text", memo.transcriptForShare())
    }
}
