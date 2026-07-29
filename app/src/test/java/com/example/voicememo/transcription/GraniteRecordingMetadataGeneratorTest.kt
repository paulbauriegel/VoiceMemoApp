package com.example.voicememo.transcription

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class GraniteRecordingMetadataGeneratorTest {
    @Test
    fun parsesStrictTwoLineResponse() {
        val metadata = parseRecordingMetadata(
            """
            TITLE: Release Planning
            SUMMARY: The team agrees on the release scope and assigns the remaining work.
            """.trimIndent(),
        )

        assertEquals("Release Planning", metadata?.title)
        assertEquals(
            "The team agrees on the release scope and assigns the remaining work.",
            metadata?.summary,
        )
    }

    @Test
    fun parsesQuotedJsonStyleResponseAndKeepsOneLineValues() {
        val metadata = parseRecordingMetadata(
            """
            {
              "title": "Garden renovation.",
              "summary": "Plans the garden layout, materials, and contractor follow-up."
            }
            """.trimIndent(),
        )

        assertEquals("Garden renovation", metadata?.title)
        assertEquals(
            "Plans the garden layout, materials, and contractor follow-up.",
            metadata?.summary,
        )
    }

    @Test
    fun rejectsIncompleteResponse() {
        assertEquals(null, parseRecordingMetadata("TITLE: Only a title"))
    }

    @Test
    fun excerptSamplesBeginningMiddleAndEndWithinBudget() {
        val transcript = buildString {
            repeat(80) { append("begin$it ") }
            repeat(80) { append("middle$it ") }
            repeat(80) { append("end$it ") }
        }

        val excerpt = buildMetadataTranscriptExcerpt(transcript, maximumCharacters = 600)

        assertTrue(excerpt.length <= 600)
        assertTrue(excerpt.contains("begin"))
        assertTrue(excerpt.contains("middle"))
        assertTrue(excerpt.contains("end"))
        assertEquals(2, Regex("\\[…]").findAll(excerpt).count())
    }

    @Test
    fun promptRequestsSameLanguageAndBoundedOutput() {
        val prompt = metadataPrompt("Wir besprechen den Projektplan.")

        assertNotNull(prompt)
        assertTrue(prompt.contains("same language"))
        assertTrue(prompt.contains("at most 24 words"))
        assertTrue(prompt.endsWith("Wir besprechen den Projektplan."))
    }
}
