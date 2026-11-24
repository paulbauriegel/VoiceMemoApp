package com.example.voicememo.ui

/** A joined line is closed at the end of a sentence once it is this long. */
internal const val JOINED_LINE_TARGET_MILLIS = 10_000L

/** A joined line never covers more than this. */
internal const val JOINED_LINE_MAX_MILLIS = 20_000L

/** A pause this long between two lines keeps them apart. */
internal const val JOINED_LINE_MAX_PAUSE_MILLIS = 2_000L

/**
 * Groups consecutive transcript lines of one speaker for display, as index
 * ranges into [segments] that cover every line once and in order. Lines whose
 * speaker is unknown or mixed stay on their own. A group grows until it ends a
 * sentence after [JOINED_LINE_TARGET_MILLIS], and never past
 * [JOINED_LINE_MAX_MILLIS]. The stored lines are not changed.
 */
internal fun joinSpeakerLines(segments: List<TranscriptSegmentUi>): List<IntRange> {
    val groups = mutableListOf<IntRange>()
    var first = 0
    for (index in 1..segments.size) {
        val start = segments[first]
        val last = segments[index - 1]
        val next = segments.getOrNull(index)
        val joins = next != null &&
            (start.speaker == SpeakerUi.Me || start.speaker == SpeakerUi.Other) &&
            next.speaker == start.speaker &&
            next.startMillis - last.endMillis <= JOINED_LINE_MAX_PAUSE_MILLIS &&
            next.endMillis - start.startMillis <= JOINED_LINE_MAX_MILLIS &&
            !(last.endMillis - start.startMillis >= JOINED_LINE_TARGET_MILLIS && last.text.endsSentence())
        if (!joins) {
            groups += first until index
            first = index
        }
    }
    return groups
}

private fun String.endsSentence(): Boolean =
    trimEnd().trimEnd('"', '”', '“', '\'', '’', ')').lastOrNull() in SENTENCE_END

private val SENTENCE_END = setOf('.', '!', '?', '…')
