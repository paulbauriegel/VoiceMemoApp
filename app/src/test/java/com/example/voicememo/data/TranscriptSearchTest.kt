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

    @Test
    fun mergeKeepsFuzzyWordRangesFirstThenSemanticSegmentsByRank() {
        val fuzzy = TranscriptSearch.find(segments, "budget")
        val merged = TranscriptSearch.merge(
            segmentTexts = segments,
            fuzzy = fuzzy,
            semanticSegmentIndices = listOf(2, 1),
        )
        assertEquals(fuzzy, merged.take(fuzzy.size))
        assertEquals(TranscriptMatchKind.Fuzzy, merged[0].kind)
        assertEquals("budget", segments[0].substring(merged[0].range))
        assertEquals(listOf(2, 1), merged.drop(fuzzy.size).map { it.segmentIndex })
        merged.drop(fuzzy.size).forEach { match ->
            assertEquals(TranscriptMatchKind.Semantic, match.kind)
            assertEquals(segments[match.segmentIndex], segments[match.segmentIndex].substring(match.range))
        }
    }

    @Test
    fun mergeDoesNotRepeatSegmentsAlreadyMatchedByText() {
        val fuzzy = TranscriptSearch.find(segments, "budget")
        val merged = TranscriptSearch.merge(segments, fuzzy, semanticSegmentIndices = listOf(0, 2, 2))
        assertEquals(fuzzy.size + 1, merged.size)
        assertEquals(2, merged.last().segmentIndex)
    }

    @Test
    fun semanticOnlyResultsHighlightTrimmedWholeSegments() {
        val texts = listOf("  Wir treffen uns morgen.  ", "   ", "Ok")
        val merged = TranscriptSearch.merge(texts, emptyList(), semanticSegmentIndices = listOf(1, 0, 7))
        val match = merged.single()
        assertEquals(0, match.segmentIndex)
        assertEquals("Wir treffen uns morgen.", texts[0].substring(match.range))
    }

    @Test
    fun mergeWithoutSemanticHitsIsTheFuzzyResult() {
        val fuzzy = TranscriptSearch.find(segments, "recording")
        assertEquals(fuzzy, TranscriptSearch.merge(segments, fuzzy, emptyList()))
    }
}
