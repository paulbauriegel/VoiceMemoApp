package com.example.voicememo.importing

import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class PcmStreamConverterTest {
    @Test
    fun downmixesInterleavedStereoPcm16() {
        val output = mutableListOf<Short>()
        val rates = mutableListOf<Int>()
        val converter = PcmStreamConverter { rate, samples, count ->
            rates += rate
            repeat(count) { output += samples[it] }
        }
        val source = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).apply {
            putShort(1_000)
            putShort((-500).toShort())
            putShort(Short.MAX_VALUE)
            putShort(Short.MAX_VALUE)
            flip()
        }

        converter.consume(
            source,
            PlatformPcmFormat(
                sampleRate = 48_000,
                channelCount = 2,
                encoding = PCM_ENCODING_16_BIT,
            ),
        )
        converter.finish()

        assertEquals(listOf(48_000), rates)
        assertArrayEquals(shortArrayOf(250, Short.MAX_VALUE), output.toShortArray())
    }

    @Test
    fun carriesSplitStereoFrameAcrossCodecBuffers() {
        val output = mutableListOf<Short>()
        val converter = PcmStreamConverter { _, samples, count ->
            repeat(count) { output += samples[it] }
        }
        val format = PlatformPcmFormat(44_100, 2, PCM_ENCODING_16_BIT)
        val frame = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).apply {
            putShort(10_000)
            putShort((-2_000).toShort())
        }.array()

        converter.consume(ByteBuffer.wrap(frame.copyOfRange(0, 1)), format)
        assertTrueEmpty(output)
        converter.consume(ByteBuffer.wrap(frame.copyOfRange(1, frame.size)), format)
        converter.finish()

        assertArrayEquals(shortArrayOf(4_000), output.toShortArray())
    }

    @Test
    fun rejectsTruncatedFinalFrame() {
        val converter = PcmStreamConverter { _, _, _ -> }
        val format = PlatformPcmFormat(16_000, 2, PCM_ENCODING_16_BIT)
        converter.consume(ByteBuffer.wrap(byteArrayOf(1, 2, 3)), format)

        val failure = assertThrows(IllegalArgumentException::class.java) {
            converter.finish()
        }

        assertEquals("The decoder ended in the middle of a PCM frame.", failure.message)
    }

    private fun assertTrueEmpty(values: List<Short>) {
        assertEquals(emptyList<Short>(), values)
    }
}
