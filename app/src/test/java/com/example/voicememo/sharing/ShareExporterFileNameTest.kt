package com.example.voicememo.sharing

import org.junit.Assert.assertEquals
import org.junit.Test

class ShareExporterFileNameTest {
    @Test
    fun keepsUnicodeButRemovesPathSeparators() {
        assertEquals("Gespräch _ 23_07", safeShareFileName("  Gespräch / 23:07  "))
    }

    @Test
    fun separatesTranscriptSegmentsWithBlankLines() {
        val memo = ShareableMemo(
            id = "memo-id",
            title = "Meeting",
            audioPath = "/recordings/meeting.wav",
            effectiveTranscript = "First segment Second segment",
            transcriptSegments = listOf(" First segment ", "Second segment"),
        )

        assertEquals("First segment\n\nSecond segment", memo.transcriptForShare())
    }

    @Test
    fun keepsPlainTranscriptWhenSegmentsAreUnavailable() {
        val memo = ShareableMemo(
            id = "memo-id",
            title = "Meeting",
            audioPath = "/recordings/meeting.wav",
            effectiveTranscript = "Edited transcript",
        )

        assertEquals("Edited transcript", memo.transcriptForShare())
    }
}
