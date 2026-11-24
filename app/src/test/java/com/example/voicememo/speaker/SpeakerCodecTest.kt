package com.example.voicememo.speaker

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Test

class SpeakerCodecTest {
    @Test
    fun vectorsAndBitSetsRoundTrip() {
        val vector = floatArrayOf(0f, -1.5f, 3.25e-7f, Float.MAX_VALUE)
        assertArrayEquals(vector, SpeakerCodec.floats(SpeakerCodec.bytes(vector)), 0f)
        assertArrayEquals(byteArrayOf(0, 0, 0xC0.toByte(), 0xBF.toByte()), SpeakerCodec.bytes(floatArrayOf(-1.5f)))

        val bits = BooleanArray(589) { it % 3 == 0 || it == 588 }
        val encoded = SpeakerCodec.bytes(bits)
        assertEquals(74, encoded.size)
        assertArrayEquals(bits, SpeakerCodec.bits(encoded, bits.size))
    }

    @Test
    fun profileKeyDependsOnlyOnTheSetOfSamples() {
        assertEquals(SpeakerLabelIndex.profileKey(listOf("a", "b")), SpeakerLabelIndex.profileKey(listOf("b", "a")))
        assertNotEquals(SpeakerLabelIndex.profileKey(listOf("a")), SpeakerLabelIndex.profileKey(listOf("a", "b")))
    }
}
