package com.example.voicememo.data

import java.util.Calendar
import org.junit.Assert.assertEquals
import org.junit.Test

class MonthGroupingTest {
    @Test
    fun groupsNewestMemosByLocalCalendarMonth() {
        val juneOlder = summary(id = "june-old", time = localTime(2026, 6, 1))
        val july = summary(id = "july", time = localTime(2026, 7, 12))
        val juneNewer = summary(id = "june-new", time = localTime(2026, 6, 28))

        val groups = groupByMonth(listOf(juneOlder, july, juneNewer))

        assertEquals(listOf(7, 6), groups.map { it.month })
        assertEquals(listOf("june-new", "june-old"), groups[1].memos.map { it.id })
    }

    private fun localTime(year: Int, month: Int, day: Int): Long =
        Calendar.getInstance().apply {
            clear()
            set(year, month - 1, day, 12, 0, 0)
        }.timeInMillis

    private fun summary(id: String, time: Long): MemoSummary =
        MemoSummary(
            id = id,
            title = id,
            createdAtMillis = time,
            updatedAtMillis = time,
            durationMillis = 1,
            audioPath = "$id.wav",
            waveformPath = null,
            waveformState = WaveformState.PENDING,
            languageOverride = null,
            effectiveTranscript = "",
            transcriptSource = TranscriptSource.NONE,
            detectedLanguage = null,
            latestJob = null,
        )
}
