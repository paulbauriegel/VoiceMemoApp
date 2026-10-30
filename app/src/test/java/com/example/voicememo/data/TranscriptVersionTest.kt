package com.example.voicememo.data

import org.junit.Assert.*
import org.junit.Test

class TranscriptVersionTest {
    @Test
    fun correctionsRoundTripPreservesEmptyTextAndUnicode() {
        val edits = listOf(
            TranscriptSegmentEdit("memo", "job:0:0", "", 100),
            TranscriptSegmentEdit("memo", "job:0:1", "Grüße 世界\nNext = line", 200),
        )
        assertEquals(edits, TranscriptEditSnapshot.decode("memo", TranscriptEditSnapshot.encode(edits)))
        assertTrue(TranscriptEditSnapshot.decode("memo", "").isEmpty())
    }

    @Test
    fun historyRoundTripKeepsAlternativesSelectionAndIntentionalEmptyEdit() {
        val versions = listOf(
            version("first", false, "Original result", null),
            version("second", true, "New result", ""),
        )
        val restored = TranscriptVersionArchive.decode("remapped", TranscriptVersionArchive.encode(versions))
        assertEquals(listOf("Original result", "New result"), restored.map { it.transcript.generatedText })
        assertEquals(listOf(false, true), restored.map { it.selected })
        assertEquals("", restored.last().transcript.effectiveText)
        assertTrue(restored.all { it.memoId == "remapped" && it.transcript.memoId == "remapped" })
        assertEquals(2, restored.map { it.id }.distinct().size)
        assertTrue(restored.all { it.jobId == null })
    }

    @Test(expected = IllegalArgumentException::class)
    fun historyRejectsMultipleActiveVersions() {
        TranscriptVersionArchive.decode("memo", TranscriptVersionArchive.encode(listOf(
            version("first", true, "First", null), version("second", true, "Second", null),
        )))
    }

    private fun version(id: String, selected: Boolean, generated: String, edited: String?) =
        TranscriptVersion(id, "memo", "job-$id", "whisper", selected,
            TranscriptRecord("memo", generated, edited, "en", 100, 200.takeIf { edited != null }?.toLong(), 1))
}
