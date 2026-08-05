package com.example.voicememo.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
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

    @Test
    fun quietWaveformUsesPercentileGainWithoutLettingSpikesFlattenIt() {
        val quiet = List(68) { 0.03f } + List(4) { 0.15f }

        val normalized = normalizeWaveformForDisplay(quiet)

        assertEquals(0.72f, normalized.first(), 0.0001f)
        assertEquals(1f, normalized.last(), 0.0001f)
    }

    @Test
    fun readableWaveformIsNotReducedAndSilenceStaysSilent() {
        val readable = listOf(0.2f, 0.45f, 0.8f, 1f)

        assertEquals(readable, normalizeWaveformForDisplay(readable))
        assertEquals(
            listOf(0f, 0f, 0f),
            normalizeWaveformForDisplay(listOf(0f, 0f, 0f)),
        )
        assertTrue(normalizeWaveformForDisplay(listOf(Float.NaN)).single() == 0f)
    }
}
