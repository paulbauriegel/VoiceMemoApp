package com.example.voicememo.data

import com.example.voicememo.transcription.ParakeetTranscriptMerger
import com.example.voicememo.transcription.TimedWordsCodec
import com.example.voicememo.transcription.WhisperTimestampTokens
import com.example.voicememo.transcription.WhisperTokenizer

/** One timed transcript line. [id] is stable for the chunk job that produced it. */
data class TimedTranscriptSegment(
    val id: String,
    val startMillis: Long,
    val endMillis: Long,
    val text: String,
    val isEdited: Boolean = false,
)

/**
 * Derives timed segments from stored chunks. Shared by the detail screen and
 * the semantic index so both agree on segment ids and text.
 */
object TranscriptSegments {
    private const val CHUNK_OVERLAP_MILLIS = 2_000L

    fun generated(detail: MemoDetail, tokenizer: WhisperTokenizer): List<TimedTranscriptSegment> =
        generated(detail.chunks, tokenizer)

    fun generated(
        chunks: List<TranscriptChunk>,
        tokenizer: WhisperTokenizer,
    ): List<TimedTranscriptSegment> {
        if (chunks.isNotEmpty() && chunks.all { it.timedWordsJson != null }) {
            return timedWordSegments(chunks)
        }
        return chunks
            .filter { it.rawText.isNotBlank() }
            .sortedBy { it.chunkIndex }
            .flatMap { chunk -> chunk.toSegments(tokenizer) }
    }

    /**
     * Parakeet windows overlap by five seconds; their words are joined exactly as the
     * transcript text was ([ParakeetTranscriptMerger]) and grouped into lines. Chunks of a
     * finished job never change, so the line ids are stable.
     */
    private fun timedWordSegments(chunks: List<TranscriptChunk>): List<TimedTranscriptSegment> {
        val windows = chunks.sortedBy { it.chunkIndex }.map { chunk ->
            ParakeetTranscriptMerger.Window(
                startMillis = chunk.windowStartMillis,
                endMillis = chunk.windowEndMillis,
                words = TimedWordsCodec.decode(chunk.timedWordsJson).orEmpty().map { word ->
                    word.copy(
                        startMillis = chunk.windowStartMillis + word.startMillis,
                        endMillis = chunk.windowStartMillis + word.endMillis,
                    )
                },
            )
        }
        val jobId = chunks.first().jobId
        return ParakeetTranscriptMerger.lines(ParakeetTranscriptMerger.merge(windows))
            .mapIndexed { index, line ->
                TimedTranscriptSegment(
                    id = "$jobId:w$index",
                    startMillis = line.startMillis,
                    endMillis = line.endMillis,
                    text = line.text,
                )
            }
    }

    fun isPlainTextEdit(detail: MemoDetail): Boolean =
        detail.transcript?.editedText != null && detail.segmentEdits.isEmpty()

    /**
     * Generated segments with the user's corrections applied. Plain-text edits
     * from before segment editing can't be aligned, so they have no segments.
     */
    fun timed(detail: MemoDetail, tokenizer: WhisperTokenizer): List<TimedTranscriptSegment> {
        if (isPlainTextEdit(detail)) return emptyList()
        val edits = detail.segmentEdits.associateBy(TranscriptSegmentEdit::segmentId)
        return generated(detail, tokenizer).map { segment ->
            edits[segment.id]
                ?.let { segment.copy(text = it.text, isEdited = true) }
                ?: segment
        }
    }

    private fun TranscriptChunk.toSegments(
        tokenizer: WhisperTokenizer,
    ): List<TimedTranscriptSegment> {
        val timestampBegin = WhisperTimestampTokens.DEFAULT_TIMESTAMP_BEGIN_TOKEN_ID
        if (!WhisperTimestampTokens.containsTimestamp(tokenIds, timestampBegin)) {
            return listOf(coarseSegment())
        }
        val chunkDuration = (windowEndMillis - windowStartMillis).coerceAtLeast(0L)
        val parsed = WhisperTimestampTokens.parseSegments(
            tokens = tokenIds,
            timestampBeginTokenId = timestampBegin,
            tokenizer = tokenizer,
            audioDurationMillis = chunkDuration,
        )
        if (parsed.isEmpty()) return listOf(coarseSegment())

        val overlapBoundary = if (chunkIndex == 0) 0L else CHUNK_OVERLAP_MILLIS
        return parsed.mapIndexedNotNull { segmentIndex, segment ->
            if (segment.endMillis <= overlapBoundary) {
                null
            } else {
                val startMillis =
                    windowStartMillis + maxOf(segment.startMillis, overlapBoundary)
                val endMillis = windowStartMillis + segment.endMillis
                if (endMillis <= startMillis || segment.text.isBlank()) {
                    null
                } else {
                    TimedTranscriptSegment(
                        id = "$jobId:$chunkIndex:$segmentIndex",
                        startMillis = startMillis,
                        endMillis = endMillis,
                        text = segment.text,
                    )
                }
            }
        }.ifEmpty { listOf(coarseSegment()) }
    }

    private fun TranscriptChunk.coarseSegment(): TimedTranscriptSegment =
        TimedTranscriptSegment(
            id = "$jobId:$chunkIndex",
            startMillis = windowStartMillis,
            endMillis = windowEndMillis,
            text = WhisperTimestampTokens.withoutTimestampMarkers(rawText),
        )
}
