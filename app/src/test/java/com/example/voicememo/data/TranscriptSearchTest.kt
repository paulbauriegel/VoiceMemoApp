package com.example.voicememo.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class TranscriptSearchTest {
    private val segments = listOf(
        "Let's start the budget meeting.",
        "Grüße aus dem Café, wir planen das Projekt.",
        "The recording ends here.",
    )

    @Test
    fun exactPrefixAndTypoMatchesFindTheirSegments() {
        assertEquals(listOf(0), TranscriptSearch.find(segments, "budget").map { it.segmentIndex })
        assertEquals(listOf(0), TranscriptSearch.find(segments, "budg").map { it.segmentIndex })
        assertEquals(listOf(2), TranscriptSearch.find(segments, "recroding").map { it.segmentIndex })
    }

    @Test
    fun rangesPointAtTheOriginalAccentedText() {
        val matches = TranscriptSearch.find(segments, "grusse cafe")
        assertEquals(2, matches.size)
        assertEquals("Grüße", segments[1].substring(matches[0].range))
        assertEquals("Café", segments[1].substring(matches[1].range))
    }

    @Test
    fun decomposedAccentsStayInsideTheWord() {
        val decomposed = listOf("Café au lait")
        val match = TranscriptSearch.find(decomposed, "cafe").single()
        assertEquals("Café", decomposed[0].substring(match.range))
    }

    @Test
    fun everyTermMustMatchWithinOneSegment() {
        assertTrue(TranscriptSearch.find(segments, "budget projekt").isEmpty())
        assertEquals(2, TranscriptSearch.find(segments, "projekt planen").size)
    }

    @Test
    fun blankQueryFindsNothing() {
        assertTrue(TranscriptSearch.find(segments, "  ").isEmpty())
    }
}
