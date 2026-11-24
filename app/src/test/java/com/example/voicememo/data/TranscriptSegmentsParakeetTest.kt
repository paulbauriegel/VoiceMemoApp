package com.example.voicememo.data

import com.example.voicememo.transcription.TimedWordsCodec
import com.example.voicememo.transcription.WhisperTimedSegment
import com.example.voicememo.transcription.WhisperTokenizer
import org.junit.Assert.assertEquals
import org.junit.Test

class TranscriptSegmentsParakeetTest {
    private fun chunk(index: Int, startMillis: Long, endMillis: Long, vararg words: Pair<Long, String>) =
        TranscriptChunk(
            jobId = "job",
            chunkIndex = index,
            memoId = "memo",
            windowStartMillis = startMillis,
            windowEndMillis = endMillis,
            status = TranscriptChunkStatus.COMPLETED,
            rawText = words.joinToString(" ") { it.second },
            tokenIds = emptyList(),
            detectedLanguage = null,
            truncated = false,
            warning = null,
            completedAtMillis = 1L,
            timedWordsJson = TimedWordsCodec.encode(
                words.map { (start, text) -> WhisperTimedSegment(start, start + 300, text) },
            ),
        )

    @Test
    fun overlappingWindowsBecomeOneSetOfTimedLines() {
        val segments = TranscriptSegments.generated(
            listOf(
                // Relative times; the second window starts at 25 s.
                chunk(0, 0, 30_000, 1_000L to "Hallo", 1_400L to "zusammen.", 26_000L to "Wir", 26_500L to "fangen"),
                chunk(1, 25_000, 40_000, 1_000L to "Wir", 1_500L to "fangen", 2_000L to "an."),
            ),
            WhisperTokenizer.fromTokens(emptyMap()),
        )
        assertEquals(listOf("Hallo zusammen.", "Wir fangen an."), segments.map { it.text })
        assertEquals(listOf("job:w0", "job:w1"), segments.map { it.id })
        assertEquals(26_000L, segments[1].startMillis)
        assertEquals(27_300L, segments[1].endMillis)
    }
}
