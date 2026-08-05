package com.example.voicememo

import android.os.PowerManager
import com.example.voicememo.ui.TranscriptionPhase
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class TranscriptionTimingPresentationTest {
    @Test
    fun disabledTimingNeverProducesALabel() {
        assertNull(
            transcriptionTimingLabel(
                enabled = false,
                phase = TranscriptionPhase.Complete,
                transcriptionElapsedMillis = 12_345L,
                segmentInferenceMillis = listOf(4_000L, 6_000L),
            ),
        )
    }

    @Test
    fun transcribingShowsCompletedSegmentAverage() {
        assertEquals(
            "Average per segment 5.0 s",
            transcriptionTimingLabel(
                enabled = true,
                phase = TranscriptionPhase.Transcribing,
                transcriptionElapsedMillis = null,
                segmentInferenceMillis = listOf(4_000L, 6_000L),
            ),
        )
    }

    @Test
    fun completedTranscriptShowsMeasuredWallTime() {
        assertEquals(
            "Transcription time 1m 5.5s",
            transcriptionTimingLabel(
                enabled = true,
                phase = TranscriptionPhase.Complete,
                transcriptionElapsedMillis = 65_500L,
                segmentInferenceMillis = listOf(40_000L, 50_000L),
            ),
        )
    }

    @Test
    fun noCompletedSegmentsHasNoRunningAverage() {
        assertNull(
            transcriptionTimingLabel(
                enabled = true,
                phase = TranscriptionPhase.Transcribing,
                transcriptionElapsedMillis = null,
                segmentInferenceMillis = emptyList(),
            ),
        )
    }

    @Test
    fun runningAndCompletedThermalLabelsUseThePeakStatus() {
        assertEquals(
            "Thermal status: Moderate",
            transcriptionThermalStatusLabel(
                enabled = true,
                phase = TranscriptionPhase.Transcribing,
                maximumThermalStatus = PowerManager.THERMAL_STATUS_MODERATE,
            ),
        )
        assertEquals(
            "Peak thermal status: Moderate",
            transcriptionThermalStatusLabel(
                enabled = true,
                phase = TranscriptionPhase.Complete,
                maximumThermalStatus = PowerManager.THERMAL_STATUS_MODERATE,
            ),
        )
    }

    @Test
    fun completedTranscriptShowsRealTimeFactor() {
        assertEquals(
            "RTF 0.500",
            transcriptionRealTimeFactorLabel(
                enabled = true,
                phase = TranscriptionPhase.Complete,
                transcriptionElapsedMillis = 30_000L,
                audioDurationMillis = 60_000L,
            ),
        )
        assertNull(
            transcriptionRealTimeFactorLabel(
                enabled = true,
                phase = TranscriptionPhase.Transcribing,
                transcriptionElapsedMillis = 30_000L,
                audioDurationMillis = 60_000L,
            ),
        )
    }
}
