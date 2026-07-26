package com.example.voicememo.transcription

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin
import org.junit.Assert.assertEquals
import org.junit.Test

class MixedRadixFftTest {
    @Test
    fun powerSpectrum_matchesDirectDft() {
        val input = FloatArray(400) { index ->
            (
                0.7 * sin(2.0 * PI * 7 * index / 400) +
                    0.2 * cos(2.0 * PI * 31 * index / 400)
                ).toFloat()
        }
        val actual = FloatArray(201)
        MixedRadixFft(400).powerSpectrum(input, actual)

        for (bin in listOf(0, 1, 7, 31, 73, 200)) {
            var expectedReal = 0.0
            var expectedImaginary = 0.0
            for (sample in input.indices) {
                val angle = -2.0 * PI * bin * sample / input.size
                expectedReal += input[sample] * cos(angle)
                expectedImaginary += input[sample] * sin(angle)
            }
            val expectedPower =
                expectedReal * expectedReal + expectedImaginary * expectedImaginary
            assertEquals("bin $bin", expectedPower, actual[bin].toDouble(), 0.03)
        }
    }
}
