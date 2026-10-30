package com.example.voicememo.data

import java.nio.ByteBuffer
import java.nio.ByteOrder

/** A segment that is semantically close to the query; higher [score] is closer. */
data class SemanticHit(
    val segmentId: String,
    val score: Float,
)

/**
 * Cosine top-k over unit-length segment embeddings.
 *
 * Raw e5 cosines sit in a narrow 0.7–0.9 band and short, generic lines
 * ("What?", "Really.") score high for every query. Each score is therefore
 * hub-corrected: the segment's similarity to [hubCentroid], the mean of a few
 * unrelated multilingual anchor queries, is subtracted. On the five benchmark
 * transcripts a corrected score of [MIN_SCORE] kept 10 of 16 related queries
 * and none of 9 unrelated ones (de/en/es query wording).
 */
object SemanticTranscriptRanking {
    const val MIN_SCORE = 0.03f
    const val MAX_HITS = 5

    /** Queries shorter than this are left to fuzzy matching. */
    const val MIN_QUERY_LENGTH = 3

    /** Unrelated topics in the supported languages; embedded with the query prefix. */
    val HUB_ANCHOR_QUERIES = listOf(
        "weather", "money", "family", "work", "food", "travel", "health", "music",
        "Wetter", "Arbeit", "Familie", "Essen", "tiempo", "trabajo", "météo", "travail",
    )

    fun rank(
        query: FloatArray,
        segments: List<Pair<String, FloatArray>>,
        hubCentroid: FloatArray,
        minScore: Float = MIN_SCORE,
        maxHits: Int = MAX_HITS,
    ): List<SemanticHit> =
        segments
            .map { (id, vector) -> SemanticHit(id, dot(query, vector) - dot(hubCentroid, vector)) }
            .filter { it.score >= minScore }
            // sortedByDescending is stable, so ties keep transcript order.
            .sortedByDescending(SemanticHit::score)
            .take(maxHits)

    fun centroid(vectors: List<FloatArray>): FloatArray {
        require(vectors.isNotEmpty()) { "A centroid needs at least one vector." }
        val result = FloatArray(vectors.first().size)
        vectors.forEach { vector ->
            for (index in result.indices) result[index] += vector[index] / vectors.size
        }
        return result
    }

    fun dot(left: FloatArray, right: FloatArray): Float {
        require(left.size == right.size) { "Embedding sizes differ: ${left.size} vs ${right.size}." }
        var sum = 0f
        for (index in left.indices) sum += left[index] * right[index]
        return sum
    }

    fun toBytes(vector: FloatArray): ByteArray {
        val buffer = ByteBuffer.allocate(vector.size * Float.SIZE_BYTES)
            .order(ByteOrder.LITTLE_ENDIAN)
        buffer.asFloatBuffer().put(vector)
        return buffer.array()
    }

    fun fromBytes(bytes: ByteArray): FloatArray {
        val floats = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(floats.remaining()).also(floats::get)
    }
}
