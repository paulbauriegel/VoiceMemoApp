package com.example.voicememo.transcription

import kotlin.math.PI
import kotlin.math.sin
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class WhisperFeatureExtractorTest {
    @Test
    fun silence_matchesWhisperLogNormalization() {
        val features = WhisperFeatureExtractor().extract(FloatArray(16_000))

        assertEquals(128 * 3_000, features.size)
        for (index in features.indices step 997) {
            assertEquals(-1.5f, features[index], 1e-6f)
        }
    }

    @Test
    fun oneKilohertzTone_isFiniteAndHasSpectralContrast() {
        val samples = FloatArray(16_000) { index ->
            (0.25 * sin(2.0 * PI * 1_000 * index / 16_000)).toFloat()
        }
        val features = WhisperFeatureExtractor().extract(samples)

        assertTrue(features.all(Float::isFinite))
        val activeFrame = 25
        val values = FloatArray(128) { mel -> features[mel * 3_000 + activeFrame] }
        assertTrue(values.max() - values.min() > 1.0f)
    }

    @Test
    fun slaneyFilterbank_hasExpectedShapeAndPositiveArea() {
        val filters = WhisperFeatureExtractor().melFilterbankCopy()

        assertEquals(128 * 201, filters.size)
        assertTrue(filters.all { it >= 0.0f && it.isFinite() })
        assertTrue(filters.sum() > 0.0f)
    }
}
