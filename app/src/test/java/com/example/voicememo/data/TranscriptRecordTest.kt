package com.example.voicememo.data

import org.junit.Assert.assertEquals
import org.junit.Test

class TranscriptRecordTest {
    @Test
    fun editedTextTakesPrecedenceOverGeneratedText() {
        val transcript = record(generated = "generated", edited = "corrected")

        assertEquals("corrected", transcript.effectiveText)
    }

    @Test
    fun intentionalEmptyEditDoesNotFallBackToGeneratedText() {
        val transcript = record(generated = "generated", edited = "")

        assertEquals("", transcript.effectiveText)
    }

    @Test
    fun nullEditUsesLatestGeneratedText() {
        val transcript = record(generated = "new generation", edited = null)

        assertEquals("new generation", transcript.effectiveText)
    }

    private fun record(generated: String?, edited: String?): TranscriptRecord =
        TranscriptRecord(
            memoId = "memo",
            generatedText = generated,
            editedText = edited,
            detectedLanguage = "en",
            generatedAtMillis = 1L,
            editedAtMillis = edited?.let { 2L },
            generatedRevision = 2L,
        )
}
