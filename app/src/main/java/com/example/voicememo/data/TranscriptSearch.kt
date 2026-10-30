package com.example.voicememo.data

/**
 * A highlighted occurrence inside one transcript segment. [range] indexes the
 * original, unnormalized segment text so the UI can style it directly.
 */
data class TranscriptMatch(
    val segmentIndex: Int,
    val range: IntRange,
    val kind: TranscriptMatchKind = TranscriptMatchKind.Fuzzy,
)

enum class TranscriptMatchKind {
    /** A word that matches the query text. */
    Fuzzy,

    /** A whole segment whose meaning is close to the query. */
    Semantic,
}

/**
 * Find-in-transcript using the same fuzzy rules as library search.
 *
 * A segment matches when every query term matches at least one of its words;
 * every word that matches any term is then reported as an occurrence.
 */
object TranscriptSearch {
    // Combining marks are included so decomposed accents stay inside their word.
    private val wordPattern = Regex("[\\p{L}\\p{M}\\p{N}_]+")

    fun find(segmentTexts: List<String>, query: String): List<TranscriptMatch> {
        val terms = MemoSearchEngine.queryTerms(query)
        if (terms.isEmpty()) return emptyList()
        return segmentTexts.flatMapIndexed { segmentIndex, text ->
            val words = wordPattern.findAll(text).map { word ->
                word.range to MemoSearchEngine.normalize(word.value)
            }.toList()
            val matchedTerms = HashSet<String>()
            val ranges = words.mapNotNull { (range, normalized) ->
                val matching = terms.filter { term ->
                    MemoSearchEngine.matchTerm(term, normalized) != null
                }
                if (matching.isEmpty()) {
                    null
                } else {
                    matchedTerms += matching
                    range
                }
            }
            if (matchedTerms.size == terms.toSet().size) {
                ranges.map { TranscriptMatch(segmentIndex, it) }
            } else {
                emptyList()
            }
        }
    }
    /**
     * Combines fuzzy word matches with semantically ranked segments.
     *
     * Fuzzy matches come first in transcript order and keep their word
     * ranges. Semantic segments follow in rank order, highlighted whole, and
     * are skipped where a fuzzy match already marks that segment.
     *
     * @param semanticSegmentIndices segment indices, best first.
     */
    fun merge(
        segmentTexts: List<String>,
        fuzzy: List<TranscriptMatch>,
        semanticSegmentIndices: List<Int>,
    ): List<TranscriptMatch> {
        val fuzzySegments = fuzzy.mapTo(HashSet(), TranscriptMatch::segmentIndex)
        val semantic = semanticSegmentIndices
            .distinct()
            .filter { it in segmentTexts.indices && it !in fuzzySegments }
            .mapNotNull { index ->
                wholeSegmentRange(segmentTexts[index])?.let { range ->
                    TranscriptMatch(index, range, TranscriptMatchKind.Semantic)
                }
            }
        return fuzzy + semantic
    }

    /** The segment without surrounding whitespace, or null when it is blank. */
    private fun wholeSegmentRange(text: String): IntRange? {
        val first = text.indexOfFirst { !it.isWhitespace() }
        if (first < 0) return null
        val last = text.indexOfLast { !it.isWhitespace() }
        return first..last
    }
}
