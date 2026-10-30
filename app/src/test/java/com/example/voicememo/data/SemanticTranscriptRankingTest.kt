package com.example.voicememo.data

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.sqrt

class SemanticTranscriptRankingTest {
    private val noHub = FloatArray(3)

    @Test
    fun ranksByCosineAndDropsWeakSegments() {
        val query = unit(1f, 0f, 0f)
        val hits = SemanticTranscriptRanking.rank(
            query = query,
            segments = listOf(
                "far" to unit(0f, 1f, 0f),
                "close" to unit(1f, 0.1f, 0f),
                "closest" to unit(1f, 0f, 0f),
            ),
            hubCentroid = noHub,
            minScore = 0.5f,
        )
        assertEquals(listOf("closest", "close"), hits.map(SemanticHit::segmentId))
        assertTrue(hits[0].score > hits[1].score)
    }

    @Test
    fun keepsOnlyTopK() {
        val query = unit(1f, 0f, 0f)
        val segments = (0 until 8).map { "s$it" to unit(1f, it * 0.1f, 0f) }
        val hits = SemanticTranscriptRanking.rank(query, segments, noHub, minScore = 0f, maxHits = 3)
        assertEquals(listOf("s0", "s1", "s2"), hits.map(SemanticHit::segmentId))
    }

    @Test
    fun equalScoresKeepTranscriptOrder() {
        val vector = unit(1f, 1f, 0f)
        val hits = SemanticTranscriptRanking.rank(
            query = unit(1f, 0f, 0f),
            segments = listOf("first" to vector, "second" to vector.copyOf()),
            hubCentroid = noHub,
            minScore = 0f,
        )
        assertEquals(listOf("first", "second"), hits.map(SemanticHit::segmentId))
    }

    @Test
    fun hubCorrectionDemotesGenericSegments() {
        // "generic" is close to every query, including the unrelated anchors;
        // "topical" is a little less close to the query but not to the anchors.
        val hub = unit(0f, 1f, 0f)
        val query = unit(0.6f, 0.8f, 0f)
        val generic = unit(0f, 1f, 0f)
        val topical = unit(1f, 0f, 0f)
        val raw = SemanticTranscriptRanking.rank(
            query,
            listOf("generic" to generic, "topical" to topical),
            hubCentroid = noHub,
            minScore = -1f,
        )
        assertEquals("generic", raw.first().segmentId)
        val corrected = SemanticTranscriptRanking.rank(
            query,
            listOf("generic" to generic, "topical" to topical),
            hubCentroid = hub,
            minScore = SemanticTranscriptRanking.MIN_SCORE,
        )
        assertEquals(listOf("topical"), corrected.map(SemanticHit::segmentId))
    }

    @Test
    fun centroidIsTheMeanVector() {
        val centroid = SemanticTranscriptRanking.centroid(
            listOf(floatArrayOf(1f, 0f), floatArrayOf(0f, 1f)),
        )
        assertArrayEquals(floatArrayOf(0.5f, 0.5f), centroid, 1e-6f)
    }

    @Test
    fun vectorBytesRoundTripLittleEndian() {
        val vector = floatArrayOf(1f, -0.25f, 3.5e-4f, Float.MIN_VALUE)
        val bytes = SemanticTranscriptRanking.toBytes(vector)
        assertEquals(16, bytes.size)
        // 1.0f is 0x3f800000; little-endian puts the low byte first.
        assertArrayEquals(byteArrayOf(0, 0, 0x80.toByte(), 0x3f), bytes.copyOfRange(0, 4))
        assertArrayEquals(vector, SemanticTranscriptRanking.fromBytes(bytes), 0f)
    }

    private fun unit(vararg values: Float): FloatArray {
        val norm = sqrt(values.sumOf { (it * it).toDouble() }).toFloat()
        return FloatArray(values.size) { values[it] / norm }
    }
}
