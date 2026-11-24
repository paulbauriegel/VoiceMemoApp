package com.example.voicememo.speaker

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class KaldiFbankTest {
    /**
     * One second of a public benchmark sample and its features from
     * kaldi-native-fbank (the library the host speaker spike used).
     */
    private class Fixture(val samples: FloatArray, val frames: Int, val features: FloatArray)

    private fun fixture(): Fixture {
        val bytes = checkNotNull(javaClass.classLoader?.getResourceAsStream("speaker/fbank-parity.bin"))
            .use { it.readBytes() }
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val samples = FloatArray(buffer.int) { buffer.short / 32_768f }
        val frames = buffer.int
        val features = FloatArray(frames * KaldiFbank.MEL_BINS) { buffer.float }
        return Fixture(samples, frames, features)
    }

    @Test
    fun matchesKaldiNativeFbank() {
        val fixture = fixture()
        val fbank = KaldiFbank()
        assertEquals(fixture.frames, fbank.frameCount(fixture.samples.size))
        val actual = fbank.compute(fixture.samples, scale = 32_768f)
        assertEquals(fixture.features.size, actual.size)
        var worst = 0f
        for (index in actual.indices) worst = maxOf(worst, abs(actual[index] - fixture.features[index]))
        assertTrue("largest difference $worst", worst < 2e-3f)
    }

    @Test
    fun offsetAndLengthSelectTheSameFrames() {
        val fixture = fixture()
        val fbank = KaldiFbank()
        val whole = fbank.compute(fixture.samples, scale = 32_768f)
        val part = fbank.compute(fixture.samples, offset = 1_600, length = 4_000, scale = 32_768f)
        assertEquals(fbank.frameCount(4_000) * KaldiFbank.MEL_BINS, part.size)
        val skip = 10 * KaldiFbank.MEL_BINS
        for (index in part.indices) assertEquals(whole[skip + index], part[index], 0f)
    }

    @Test
    fun meanSubtractionCentresEveryBin() {
        val fixture = fixture()
        val features = KaldiFbank().compute(fixture.samples, scale = 32_768f, subtractMean = true)
        for (bin in 0 until KaldiFbank.MEL_BINS) {
            var sum = 0.0
            for (frame in 0 until fixture.frames) sum += features[frame * KaldiFbank.MEL_BINS + bin]
            assertEquals(0.0, sum / fixture.frames, 1e-4)
        }
    }

    @Test
    fun tooShortAudioGivesNoFrames() {
        assertEquals(0, KaldiFbank().frameCount(399))
        assertEquals(1, KaldiFbank().frameCount(400))
        assertEquals(0, KaldiFbank().compute(FloatArray(399)).size)
    }
}
