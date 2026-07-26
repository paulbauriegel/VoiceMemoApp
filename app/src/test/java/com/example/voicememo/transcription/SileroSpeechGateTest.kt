package com.example.voicememo.transcription

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class SileroSpeechGateTest {
    @Test
    fun threeConsecutivePositiveFrames_keepWindow() {
        val gate = SileroSpeechGate()

        assertFalse(gate.observe(0.51f))
        assertFalse(gate.observe(0.72f))
        assertTrue(gate.observe(0.55f))
    }

    @Test
    fun lowProbabilityFrame_resetsSustainedSpeech() {
        val gate = SileroSpeechGate()

        assertFalse(gate.observe(0.9f))
        assertFalse(gate.observe(0.8f))
        assertFalse(gate.observe(0.49f))
        assertFalse(gate.observe(0.9f))
        assertFalse(gate.observe(0.9f))
        assertTrue(gate.observe(0.9f))
    }
}
