package com.example.voicememo.data

/**
 * A highlighted occurrence inside one transcript segment. [range] indexes the
 * original, unnormalized segment text so the UI can style it directly.
 */
data class TranscriptMatch(
    val segmentIndex: Int,
    val range: IntRange,
)

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
}
