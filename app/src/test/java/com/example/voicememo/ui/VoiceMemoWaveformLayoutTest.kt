package com.example.voicememo.ui

import org.junit.Assert.assertEquals
import org.junit.Test

class VoiceMemoWaveformLayoutTest {
    @Test
    fun barCountFitsBarsAndGapsInsideAvailableWidth() {
        assertEquals(
            54,
            maximumWaveformBarCount(
                width = 267f,
                minimumBarWidth = 2f,
                gap = 3f,
            ),
        )
    }

    @Test
    fun narrowOrUnmeasuredWaveformStillKeepsOneBar() {
        assertEquals(
            1,
            maximumWaveformBarCount(
                width = 0f,
                minimumBarWidth = 2f,
                gap = 3f,
            ),
        )
        assertEquals(
            1,
            maximumWaveformBarCount(
                width = 1f,
                minimumBarWidth = 2f,
                gap = 3f,
            ),
        )
    }
}
