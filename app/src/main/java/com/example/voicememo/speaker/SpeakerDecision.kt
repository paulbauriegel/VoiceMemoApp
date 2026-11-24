package com.example.voicememo.speaker

import kotlin.math.sqrt

/**
 * Second half of speaker labelling: decides which window-local speakers are the
 * enrolled person, adapting to the recording first.
 *
 * A voice sounds different from one recording to the next, so a fixed
 * similarity threshold against the profile is unreliable. Instead the
 * recording's voiceprints are split into "clearly like the profile" and
 * "clearly unlike it", each side gets its own centroid(s) from this recording,
 * and every voiceprint goes to the nearer side.
 */
internal object SpeakerDecision {
    /** Says whether a voiceprint belongs to the enrolled person. */
    class Decider(
        val threshold: Float,
        val me: FloatArray?,
        val others: List<FloatArray>,
        private val profile: List<FloatArray>,
    ) {
        fun isMe(voiceprint: FloatArray): Boolean = when {
            // Too little material to adapt: plain similarity to the profile.
            me == null && others.isEmpty() -> profileScore(voiceprint, profile) >= SAME_VOICE_SCORE
            me == null -> false
            others.isEmpty() -> true
            else -> dot(voiceprint, me) >= others.maxOf { dot(voiceprint, it) }
        }
    }

    fun adapt(voiceprints: List<FloatArray>, profile: List<FloatArray>): Decider {
        require(profile.isNotEmpty())
        if (voiceprints.size < MIN_VOICEPRINTS_TO_ADAPT) {
            return Decider(SAME_VOICE_SCORE, null, emptyList(), profile)
        }
        val scores = FloatArray(voiceprints.size) { profileScore(voiceprints[it], profile) }
        val threshold = otsu(scores)
        val highMean = scores.filter { it >= threshold }.average()
        val lowMean = scores.filter { it < threshold }.average()
        val high = voiceprints.filterIndexed { index, _ -> scores[index] >= (threshold + highMean) / 2 }
        val low = voiceprints.filterIndexed { index, _ -> scores[index] < (threshold + lowMean) / 2 }
        if (high.isEmpty() || low.isEmpty()) {
            return Decider(SAME_VOICE_SCORE, null, emptyList(), profile)
        }
        val meCentroid = centroid(high)
        val minimumCluster = maxOf(MIN_OTHER_CLUSTER, (OTHER_CLUSTER_SHARE * low.size).toInt().let {
            if (it < OTHER_CLUSTER_SHARE * low.size) it + 1 else it
        })
        var otherCentroids = cluster(low).filter { it.size >= minimumCluster }.map(::centroid)
        if (otherCentroids.isEmpty()) otherCentroids = listOf(centroid(low))
        // A recording with a single voice still gets split in two. A side whose
        // centroid is as close to the profile as the enrolled person is that person.
        val genuineOthers = otherCentroids.filter { profileScore(it, profile) < SAME_VOICE_SCORE }
        val me = meCentroid.takeIf { profileScore(it, profile) >= SAME_VOICE_SCORE }
        return Decider(threshold, me, if (me == null) otherCentroids else genuineOthers, profile)
    }

    fun profileScore(voiceprint: FloatArray, profile: List<FloatArray>): Float =
        profile.maxOf { dot(voiceprint, it) }

    fun centroid(vectors: List<FloatArray>): FloatArray {
        val sum = FloatArray(vectors.first().size)
        for (vector in vectors) for (index in sum.indices) sum[index] += vector[index]
        for (index in sum.indices) sum[index] /= vectors.size
        return CamPlusPlusEmbedder.normalise(sum)
    }

    fun dot(first: FloatArray, second: FloatArray): Float {
        var sum = 0f
        for (index in first.indices) sum += first[index] * second[index]
        return sum
    }

    /** Threshold that best separates the scores into two groups (Otsu's method). */
    internal fun otsu(scores: FloatArray): Float {
        val sorted = scores.sortedArray()
        val low = percentile(sorted, 2.0)
        val high = percentile(sorted, 98.0)
        var best = -1.0
        var threshold = percentile(sorted, 50.0)
        for (step in 0 until OTSU_STEPS) {
            val candidate = low + (high - low) * step / (OTSU_STEPS - 1)
            var belowCount = 0
            var belowSum = 0.0
            var aboveSum = 0.0
            for (score in scores) {
                if (score < candidate) {
                    belowCount++
                    belowSum += score
                } else {
                    aboveSum += score
                }
            }
            val aboveCount = scores.size - belowCount
            if (belowCount < 3 || aboveCount < 3) continue
            val difference = belowSum / belowCount - aboveSum / aboveCount
            val separation = belowCount.toDouble() * aboveCount * difference * difference
            if (separation > best) {
                best = separation
                threshold = candidate
            }
        }
        return threshold.toFloat()
    }

    private fun percentile(sorted: FloatArray, percent: Double): Double {
        val position = percent / 100.0 * (sorted.size - 1)
        val lower = position.toInt()
        val upper = minOf(lower + 1, sorted.size - 1)
        return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower)
    }

    /**
     * Average-linkage clustering on cosine distance, cut at [CLUSTER_DISTANCE]:
     * groups the "unlike the profile" voiceprints into separate other voices.
     */
    internal fun cluster(vectors: List<FloatArray>): List<List<FloatArray>> {
        // Cubic in the number of items; an even subsample keeps long meetings fast.
        val items = if (vectors.size <= MAX_CLUSTER_ITEMS) {
            vectors
        } else {
            List(MAX_CLUSTER_ITEMS) { vectors[(it.toLong() * (vectors.size - 1) / (MAX_CLUSTER_ITEMS - 1)).toInt()] }
        }
        val count = items.size
        if (count <= 2) return listOf(items)
        val norms = FloatArray(count) { sqrt(dot(items[it], items[it])) }
        val distance = Array(count) { row ->
            FloatArray(count) { column ->
                if (row == column) 0f else 1f - dot(items[row], items[column]) / (norms[row] * norms[column])
            }
        }
        val members = Array<MutableList<Int>?>(count) { mutableListOf(it) }
        while (true) {
            var bestRow = -1
            var bestColumn = -1
            var best = Float.MAX_VALUE
            for (row in 0 until count) {
                if (members[row] == null) continue
                for (column in row + 1 until count) {
                    if (members[column] == null) continue
                    if (distance[row][column] < best) {
                        best = distance[row][column]
                        bestRow = row
                        bestColumn = column
                    }
                }
            }
            if (bestRow < 0 || best > CLUSTER_DISTANCE) break
            val first = members[bestRow]!!
            val second = members[bestColumn]!!
            for (other in 0 until count) {
                if (members[other] == null || other == bestRow || other == bestColumn) continue
                val merged = (first.size * distance[bestRow][other] + second.size * distance[bestColumn][other]) /
                    (first.size + second.size)
                distance[bestRow][other] = merged
                distance[other][bestRow] = merged
            }
            first += second
            members[bestColumn] = null
        }
        return members.filterNotNull().map { group -> group.map(items::get) }
    }

    /**
     * A centroid scoring at least this against the profile is the enrolled
     * person. With the bundled voice model, 14 solo recordings score 0.69 or
     * more against the profile from the other 13, and 13 other voices from
     * four recordings 0.35 or less; provisional.
     */
    const val SAME_VOICE_SCORE = 0.5f
    private const val MIN_VOICEPRINTS_TO_ADAPT = 8
    private const val OTSU_STEPS = 100
    private const val CLUSTER_DISTANCE = 0.65f
    private const val MIN_OTHER_CLUSTER = 5
    private const val OTHER_CLUSTER_SHARE = 0.05
    private const val MAX_CLUSTER_ITEMS = 400
}
