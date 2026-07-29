package com.example.voicememo.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class GeneratedMemoMetadataTest {
    @Test
    fun generatedTitleReplacesPlaceholderForNewRecording() {
        val updated = memo(titleEditedByUser = false).withGeneratedMetadata(
            generatedTitle = "Weekly planning",
            generatedSummary = "The team reviews priorities and assigns this week's work.",
            metadataModelId = "granite",
            generatedAtMillis = 200L,
        )

        assertEquals("Weekly planning", updated.title)
        assertEquals(
            "The team reviews priorities and assigns this week's work.",
            updated.generatedSummary,
        )
        assertEquals("granite", updated.metadataModelId)
        assertEquals(200L, updated.metadataGeneratedAtMillis)
        assertFalse(updated.titleEditedByUser)
    }

    @Test
    fun generatedMetadataNeverOverwritesUserTitle() {
        val updated = memo(titleEditedByUser = true).withGeneratedMetadata(
            generatedTitle = "Generated title",
            generatedSummary = "A generated one-line summary.",
            metadataModelId = "granite",
            generatedAtMillis = 200L,
        )

        assertEquals("My own title", updated.title)
        assertEquals("A generated one-line summary.", updated.generatedSummary)
        assertTrue(updated.titleEditedByUser)
    }

    private fun memo(titleEditedByUser: Boolean) = MemoRecord(
        id = "memo",
        title = "My own title",
        createdAtMillis = 100L,
        updatedAtMillis = 100L,
        durationMillis = 1_000L,
        audioPath = "/memo/audio.wav",
        waveformPath = null,
        waveformState = WaveformState.PENDING,
        languageOverride = null,
        titleEditedByUser = titleEditedByUser,
    )
}
