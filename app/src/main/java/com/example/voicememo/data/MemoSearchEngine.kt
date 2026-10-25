package com.example.voicememo.data

import java.text.Normalizer
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.abs

/**
 * Ranked, offline search over memo titles and effective transcripts.
 *
 * FTS remains the durable index, while this matcher adds behavior SQLite FTS4
 * does not provide: infix matching, typo tolerance, and relevance ranking.
 */
/** Ordered from strongest to weakest. */
internal enum class TermMatch { EXACT, PREFIX, SUBSTRING, FUZZY }

internal object MemoSearchEngine {
    private val tokenPattern = Regex("[\\p{L}\\p{N}_]+")
    private val combiningMarks = Regex("\\p{M}+")
    private val documentCache = ConcurrentHashMap<String, SearchDocument>()

    fun search(memos: List<MemoSummary>, query: String): List<MemoSummary> {
        if (query.isBlank()) return memos
        val parsed = parse(query) ?: return emptyList()
        val activeIds = memos.mapTo(hashSetOf(), MemoSummary::id)
        documentCache.keys.retainAll(activeIds)
        return memos
            .mapNotNull { memo ->
                val document = cachedDocument(memo)
                score(
                    title = document.title,
                    transcript = document.transcript,
                    query = parsed,
                )?.let { score -> RankedMemo(memo, score) }
            }
            .sortedWith(
                compareByDescending<RankedMemo> { it.score }
                    .thenByDescending { it.memo.createdAtMillis },
            )
            .map(RankedMemo::memo)
    }

    internal fun score(title: String, transcript: String, query: String): Int? =
        parse(query)?.let { parsed ->
            score(
                title = SearchField.from(title),
                transcript = SearchField.from(transcript),
                query = parsed,
            )
        }

    internal fun normalize(value: String): String =
        Normalizer.normalize(value, Normalizer.Form.NFKD)
            .lowercase(Locale.ROOT)
            .replace(combiningMarks, "")
            .replace("ß", "ss")
            .replace("æ", "ae")
            .replace("œ", "oe")
            .replace("ø", "o")
            .replace("ł", "l")
            .replace("đ", "d")
            .replace("ð", "d")
            .replace("þ", "th")

    /** Normalized query terms, using the same limits as library search. */
    internal fun queryTerms(query: String): List<String> =
        tokenPattern
            .findAll(normalize(query.take(MAX_QUERY_CHARACTERS)))
            .map { it.value }
            .take(MAX_QUERY_TERMS)
            .toList()

    /**
     * Matches one normalized query term against one normalized token with the
     * same rules library search uses: exact, prefix, infix, then conservative
     * typo tolerance.
     */
    internal fun matchTerm(term: String, token: String): TermMatch? = when {
        token == term -> TermMatch.EXACT
        token.startsWith(term) -> TermMatch.PREFIX
        (term.length >= MIN_SUBSTRING_LENGTH || containsCjkScript(term)) &&
            token.contains(term) -> TermMatch.SUBSTRING
        term.length >= MIN_FUZZY_LENGTH && (
            isAdjacentTransposition(term, token) ||
                (term.length > MIN_FUZZY_LENGTH && isSingleInsertionOrDeletion(term, token))
            ) -> TermMatch.FUZZY
        else -> null
    }

    private fun parse(query: String): ParsedQuery? {
        val terms = queryTerms(query)
        if (terms.isEmpty()) return null
        return ParsedQuery(
            phrase = terms.joinToString(separator = " "),
            terms = terms,
        )
    }

    private fun score(
        title: SearchField,
        transcript: SearchField,
        query: ParsedQuery,
    ): Int? {
        var score = 0
        var titleTermMatches = 0

        for (term in query.terms) {
            val titleScore = title.scoreTerm(
                term = term,
                exactScore = 120,
                prefixScore = 92,
                substringScore = 70,
                fuzzyScore = 58,
            )
            val transcriptScore = transcript.scoreTerm(
                term = term,
                exactScore = 68,
                prefixScore = 50,
                substringScore = 38,
                fuzzyScore = 28,
            )
            val bestScore = maxOf(titleScore, transcriptScore)
            if (bestScore == NO_MATCH) return null
            if (titleScore >= transcriptScore && titleScore != NO_MATCH) {
                titleTermMatches += 1
            }
            score += bestScore
        }

        score += when {
            title.normalized == query.phrase -> 700
            title.normalized.startsWith(query.phrase) -> 420
            title.normalized.contains(query.phrase) -> 300
            else -> 0
        }
        if (transcript.normalized.contains(query.phrase)) {
            score += 150
        }
        if (titleTermMatches == query.terms.size) {
            score += 90
        }
        return score
    }

    private data class ParsedQuery(
        val phrase: String,
        val terms: List<String>,
    )

    private data class RankedMemo(
        val memo: MemoSummary,
        val score: Int,
    )

    private data class SearchDocument(
        val sourceTitle: String,
        val sourceTranscript: String,
        val title: SearchField,
        val transcript: SearchField,
    )

    private data class SearchField(
        val normalized: String,
        val tokens: List<String>,
        val tokenSet: Set<String>,
    ) {
        fun scoreTerm(
            term: String,
            exactScore: Int,
            prefixScore: Int,
            substringScore: Int,
            fuzzyScore: Int,
        ): Int {
            if (term in tokenSet) return exactScore
            var best: TermMatch? = null
            for (token in tokens) {
                val match = matchTerm(term, token) ?: continue
                if (best == null || match < best) best = match
                if (best == TermMatch.PREFIX) break
            }
            return when (best) {
                TermMatch.EXACT -> exactScore
                TermMatch.PREFIX -> prefixScore
                TermMatch.SUBSTRING -> substringScore
                TermMatch.FUZZY -> fuzzyScore
                null -> NO_MATCH
            }
        }

        companion object {
            fun from(value: String): SearchField {
                val normalized = normalize(value)
                val allTokens = tokenPattern
                    .findAll(normalized)
                    .map { it.value }
                    .toList()
                val tokens = allTokens.distinct()
                return SearchField(
                    normalized = allTokens.joinToString(separator = " "),
                    tokens = tokens,
                    tokenSet = tokens.toHashSet(),
                )
            }
        }
    }

    private fun isAdjacentTransposition(left: String, right: String): Boolean {
        if (left.length != right.length || left.length < 2) return false
        var mismatch = 0
        while (mismatch < left.length && left[mismatch] == right[mismatch]) {
            mismatch += 1
        }
        if (mismatch >= left.lastIndex) return false
        if (
            left[mismatch] != right[mismatch + 1] ||
            left[mismatch + 1] != right[mismatch]
        ) {
            return false
        }
        for (index in mismatch + 2 until left.length) {
            if (left[index] != right[index]) return false
        }
        return true
    }

    /**
     * Accepts one missing or extra character without treating arbitrary
     * substitutions as typos. This avoids broad matches such as audio/audit.
     */
    private fun isSingleInsertionOrDeletion(left: String, right: String): Boolean {
        if (abs(left.length - right.length) != 1) return false
        val shorter = if (left.length < right.length) left else right
        val longer = if (left.length < right.length) right else left
        var shortIndex = 0
        var longIndex = 0
        var skipped = false
        while (shortIndex < shorter.length && longIndex < longer.length) {
            if (shorter[shortIndex] == longer[longIndex]) {
                shortIndex += 1
                longIndex += 1
            } else {
                if (skipped) return false
                skipped = true
                longIndex += 1
            }
        }
        return true
    }

    private fun cachedDocument(memo: MemoSummary): SearchDocument {
        val cached = documentCache[memo.id]
        if (
            cached != null &&
            cached.sourceTitle == memo.title &&
            cached.sourceTranscript == memo.effectiveTranscript
        ) {
            return cached
        }
        return SearchDocument(
            sourceTitle = memo.title,
            sourceTranscript = memo.effectiveTranscript,
            title = SearchField.from(memo.title),
            transcript = SearchField.from(memo.effectiveTranscript),
        ).also { documentCache[memo.id] = it }
    }

    private fun containsCjkScript(value: String): Boolean {
        var index = 0
        while (index < value.length) {
            val codePoint = Character.codePointAt(value, index)
            if (
                codePoint in 0x3040..0x30FF ||
                codePoint in 0x31F0..0x31FF ||
                codePoint in 0x3400..0x4DBF ||
                codePoint in 0x4E00..0x9FFF ||
                codePoint in 0x1100..0x11FF ||
                codePoint in 0x3130..0x318F ||
                codePoint in 0xA960..0xA97F ||
                codePoint in 0xAC00..0xD7FF ||
                codePoint in 0xFF65..0xFF9F ||
                codePoint in 0x20000..0x2EBEF
            ) {
                return true
            }
            index += Character.charCount(codePoint)
        }
        return false
    }

    private const val MAX_QUERY_CHARACTERS = 128
    private const val MAX_QUERY_TERMS = 20
    private const val MIN_SUBSTRING_LENGTH = 3
    private const val MIN_FUZZY_LENGTH = 4
    private const val NO_MATCH = Int.MIN_VALUE
}
