package com.example.voicememo.transcription

import java.io.ByteArrayOutputStream
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class WavAudioReaderTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test
    fun readsPcm16AndDownmixesStereo() {
        val file = temporaryFolder.newFile("stereo.wav")
        val samples = shortArrayOf(16_384, -16_384, 8_192, 8_192)
        file.writeBytes(createPcm16Wav(sampleRate = 48_000, channels = 2, samples = samples))

        val audio = WavAudioReader.read(file)

        assertEquals(48_000, audio.sampleRate)
        assertArrayEquals(floatArrayOf(0.0f, 0.25f), audio.samples, 1e-6f)
    }

    @Test
    fun deterministicResampler_keepsDuration() {
        val input = PcmAudio(FloatArray(48_000) { if (it % 3 == 0) 0.2f else 0.0f }, 48_000)

        val first = PcmResampler.toWhisperRate(input)
        val second = PcmResampler.toWhisperRate(input)

        assertEquals(16_000, first.size)
        assertArrayEquals(first, second, 0.0f)
    }

    private fun createPcm16Wav(
        sampleRate: Int,
        channels: Int,
        samples: ShortArray,
    ): ByteArray {
        val output = ByteArrayOutputStream()
        val dataBytes = samples.size * 2
        output.writeAscii("RIFF")
        output.writeIntLe(36 + dataBytes)
        output.writeAscii("WAVE")
        output.writeAscii("fmt ")
        output.writeIntLe(16)
        output.writeShortLe(1)
        output.writeShortLe(channels)
        output.writeIntLe(sampleRate)
        output.writeIntLe(sampleRate * channels * 2)
        output.writeShortLe(channels * 2)
        output.writeShortLe(16)
        output.writeAscii("data")
        output.writeIntLe(dataBytes)
        samples.forEach { output.writeShortLe(it.toInt()) }
        return output.toByteArray()
    }

    private fun ByteArrayOutputStream.writeAscii(value: String) =
        write(value.toByteArray(Charsets.US_ASCII))

    private fun ByteArrayOutputStream.writeShortLe(value: Int) {
        write(value and 0xff)
        write((value ushr 8) and 0xff)
    }

    private fun ByteArrayOutputStream.writeIntLe(value: Int) {
        write(value and 0xff)
        write((value ushr 8) and 0xff)
        write((value ushr 16) and 0xff)
        write((value ushr 24) and 0xff)
    }
}
