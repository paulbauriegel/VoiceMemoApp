package com.example.voicememo.speaker

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class SpeakerPowersetTest {
    @Test
    fun decodesEachClassToItsSpeakers() {
        val logProbabilities = FloatArray(7 * SpeakerPowerset.CLASSES) { -10f }
        for (frame in 0 until 7) logProbabilities[frame * SpeakerPowerset.CLASSES + frame] = -0.1f
        assertArrayEquals(
            byteArrayOf(0b000, 0b001, 0b010, 0b100, 0b011, 0b101, 0b110),
            SpeakerPowerset.activeSpeakers(logProbabilities),
        )
    }

    @Test
    fun reportsActivityAndOverlap() {
        val pair: Byte = 0b101
        assertTrue(SpeakerPowerset.isActive(pair, 0))
        assertFalse(SpeakerPowerset.isActive(pair, 1))
        assertTrue(SpeakerPowerset.isActive(pair, 2))
        assertTrue(SpeakerPowerset.isOverlap(pair))
        assertFalse(SpeakerPowerset.isOverlap(0b010))
        assertFalse(SpeakerPowerset.isOverlap(0))
    }
}
