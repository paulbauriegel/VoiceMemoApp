package com.example.voicememo.transcription

import kotlin.math.abs
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt
import org.json.JSONArray

/** One emitted Parakeet token; times are relative to the decoded audio's start. */
internal data class ParakeetTimedToken(val id: Int, val startMillis: Long, val endMillis: Long)

/**
 * Re-decodes audible stretches that a window decoded as blanks only.
 *
 * The Parakeet encoder occasionally collapses to blank output over clear speech, depending
 * chaotically on where the window starts. A token-free stretch of at least two seconds with
 * audible audio is re-decoded once with a short window around it (gap +/- 3 s, at most 30 s,
 * clipped to the audio given), and the tokens found strictly inside the gap are spliced in.
 * Measured on a 54-minute memo, this recovers as much as additional nudged
 * full-window retries at a quarter of the extra decodes.
 */
internal object ParakeetGapRepair {
    private const val GAP_MILLIS = 2_000L
    private const val LOUD_DBFS = -50.0
    private const val LOUD_FRACTION = 0.3
    private const val LEVEL_FRAME_SAMPLES = 1_600
    private const val TIGHT_PAD_MILLIS = 3_000L
    private const val EDGE_MILLIS = 50L
    private const val SAMPLES_PER_MILLI = ParakeetAssets.SAMPLE_RATE / 1_000

    data class Result(val tokens: List<ParakeetTimedToken>, val gapsChecked: Int, val gapsRepaired: Int)

    /**
     * [decode] decodes audio[start, end) and returns tokens with times relative to the
     * start of [audio] (not of the sub-window).
     */
    fun repair(
        audio: FloatArray,
        tokens: List<ParakeetTimedToken>,
        decode: (start: Int, end: Int) -> List<ParakeetTimedToken>,
    ): Result {
        var pieces = tokens
        var checked = 0
        var repaired = 0
        val durationMillis = audio.size.toLong() / SAMPLES_PER_MILLI
        for ((gapStart, gapEnd) in gaps(tokens, durationMillis)) {
            if (!audible(audio, gapStart, gapEnd)) continue
            checked++
            val start = max(0L, (gapStart - TIGHT_PAD_MILLIS) * SAMPLES_PER_MILLI).toInt()
            val end = min(
                audio.size.toLong(),
                min(start.toLong() + ParakeetAssets.MAX_SAMPLES, (gapEnd + TIGHT_PAD_MILLIS) * SAMPLES_PER_MILLI),
            ).toInt()
            if (end <= start) continue
            val recovered = decode(start, end).filter {
                it.startMillis >= gapStart + EDGE_MILLIS && it.endMillis <= gapEnd - EDGE_MILLIS
            }
            if (recovered.isNotEmpty()) {
                repaired++
                pieces = (pieces.filterNot { it.startMillis in (gapStart + 1) until gapEnd } + recovered)
                    .sortedBy { it.startMillis }
            }
        }
        return Result(pieces, checked, repaired)
    }

    /** Token-free stretches of at least [GAP_MILLIS] in [0, durationMillis]. */
    fun gaps(tokens: List<ParakeetTimedToken>, durationMillis: Long): List<Pair<Long, Long>> {
        val result = mutableListOf<Pair<Long, Long>>()
        var cursor = 0L
        for (token in tokens.sortedBy { it.startMillis }) {
            if (token.startMillis - cursor >= GAP_MILLIS) result += cursor to token.startMillis
            cursor = max(cursor, token.endMillis)
        }
        if (durationMillis - cursor >= GAP_MILLIS) result += cursor to durationMillis
        return result
    }

    fun audible(audio: FloatArray, startMillis: Long, endMillis: Long): Boolean {
        val from = (startMillis * SAMPLES_PER_MILLI).toInt().coerceIn(0, audio.size)
        val to = (endMillis * SAMPLES_PER_MILLI).toInt().coerceIn(from, audio.size)
        val frames = (to - from) / LEVEL_FRAME_SAMPLES
        if (frames == 0) return false
        var loud = 0
        for (frame in 0 until frames) {
            var sum = 0.0
            val base = from + frame * LEVEL_FRAME_SAMPLES
            for (index in base until base + LEVEL_FRAME_SAMPLES) sum += audio[index] * audio[index]
            if (20 * log10(sqrt(sum / LEVEL_FRAME_SAMPLES) + 1e-9) > LOUD_DBFS) loud++
        }
        return loud.toDouble() / frames >= LOUD_FRACTION
    }
}

/**
 * Joins the word streams of overlapping windows (30 s every 25 s) into one transcript.
 *
 * Port of the "aligned" policy the Parakeet benchmarks were scored with: inside the overlap,
 * find runs of at least two words that both windows emitted at nearly the same time, take
 * the run closest to the middle of the overlap, and switch from the older to the newer
 * window there. Without such an anchor, cut at the overlap midpoint on whole words.
 */
object ParakeetTranscriptMerger {
    private const val ANCHOR_MARGIN_MILLIS = 250L
    private const val ANCHOR_TOLERANCE_MILLIS = 1_000L
    private const val MINIMUM_RUN = 2
    private const val LINE_PAUSE_MILLIS = 1_500L
    private const val LINE_MAXIMUM_MILLIS = 20_000L
    private val SENTENCE_END = Regex("[.!?…。？！]['\"”’)]*$")

    /** A window's words with absolute times. */
    data class Window(val startMillis: Long, val endMillis: Long, val words: List<WhisperTimedSegment>)

    fun merge(windows: List<Window>): List<WhisperTimedSegment> {
        var merged = emptyList<WhisperTimedSegment>()
        var previousEnd = 0L
        for (window in windows.sortedBy { it.startMillis }) {
            merged = when {
                merged.isEmpty() -> window.words
                window.startMillis >= previousEnd -> merged + window.words
                window.startMillis == 0L -> window.words
                else -> joinOverlap(merged, window.words, window.startMillis, previousEnd)
            }
            previousEnd = window.endMillis
        }
        return merged
    }

    private fun joinOverlap(
        previous: List<WhisperTimedSegment>,
        current: List<WhisperTimedSegment>,
        windowStart: Long,
        previousEnd: Long,
    ): List<WhisperTimedSegment> {
        val target = (windowStart + previousEnd) / 2.0
        val previousNorm = previous.map { normalize(it.text) }
        val currentNorm = current.map { normalize(it.text) }
        fun inOverlap(word: WhisperTimedSegment) =
            word.startMillis >= windowStart + ANCHOR_MARGIN_MILLIS && word.startMillis < previousEnd
        var best: IntArray? = null
        var bestDistance = Double.MAX_VALUE
        var bestLength = 0
        for (i in previous.indices) {
            if (!inOverlap(previous[i]) || previousNorm[i].isEmpty()) continue
            for (j in current.indices) {
                if (!inOverlap(current[j]) || previousNorm[i] != currentNorm[j]) continue
                if (abs(previous[i].startMillis - current[j].startMillis) > ANCHOR_TOLERANCE_MILLIS) continue
                var length = 0
                while (i + length < previous.size && j + length < current.size) {
                    val a = previous[i + length]
                    val b = current[j + length]
                    if (
                        previousNorm[i + length] != currentNorm[j + length] ||
                        abs(a.startMillis - b.startMillis) > ANCHOR_TOLERANCE_MILLIS ||
                        max(a.startMillis, b.startMillis) >= previousEnd
                    ) {
                        break
                    }
                    length++
                }
                if (length < MINIMUM_RUN) continue
                val distance = abs((previous[i].startMillis + current[j].startMillis) / 2.0 - target)
                if (distance < bestDistance || (distance == bestDistance && length > bestLength)) {
                    best = intArrayOf(i, j)
                    bestDistance = distance
                    bestLength = length
                }
            }
        }
        best?.let { (i, j) -> return previous.subList(0, i) + current.subList(j, current.size) }
        return previous.filter { it.startMillis < target } + current.filter { it.startMillis >= target }
    }

    /** Lowercased letters, digits and apostrophes; the benchmark's word identity. */
    fun normalize(text: String): String =
        text.lowercase().filter { it.isLetterOrDigit() || it == '\'' }

    fun text(words: List<WhisperTimedSegment>): String = words.joinToString(" ") { it.text }

    /** Groups words into display lines at sentence ends, pauses, or after 20 s. */
    fun lines(words: List<WhisperTimedSegment>): List<WhisperTimedSegment> {
        val lines = mutableListOf<WhisperTimedSegment>()
        var current = mutableListOf<WhisperTimedSegment>()
        fun flush() {
            if (current.isEmpty()) return
            lines += WhisperTimedSegment(current.first().startMillis, current.last().endMillis, text(current))
            current = mutableListOf()
        }
        for (word in words) {
            val last = current.lastOrNull()
            if (
                last != null && (
                    SENTENCE_END.containsMatchIn(last.text) ||
                        word.startMillis - last.endMillis >= LINE_PAUSE_MILLIS ||
                        word.startMillis - current.first().startMillis >= LINE_MAXIMUM_MILLIS
                    )
            ) {
                flush()
            }
            current += word
        }
        flush()
        return lines
    }
}

/** Compact JSON for a window's timed words, `[[startMillis, endMillis, "text"], ...]`. */
object TimedWordsCodec {
    fun encode(words: List<WhisperTimedSegment>): String = JSONArray().also { array ->
        words.forEach { word ->
            array.put(JSONArray().put(word.startMillis).put(word.endMillis).put(word.text))
        }
    }.toString()

    fun decode(json: String?): List<WhisperTimedSegment>? {
        if (json == null) return null
        val array = runCatching { JSONArray(json) }.getOrNull() ?: return null
        return List(array.length()) { index ->
            val word = array.getJSONArray(index)
            WhisperTimedSegment(word.getLong(0), word.getLong(1), word.getString(2))
        }
    }
}
