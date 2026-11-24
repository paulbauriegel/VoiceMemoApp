package com.example.voicememo.transcription

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * Transformers `ParakeetFeatureExtractor` (= NeMo FilterbankFeatures) for one window of up to
 * 30 s: pre-emphasis 0.97, centred STFT with constant (zero) padding, n_fft 512, a 400-sample
 * symmetric Hann window centred in the frame, hop 160, Slaney mel x128, ln(x + 2^-24),
 * per-mel standardisation over the valid frames (unbiased), invalid frames zeroed.
 * Output is time-major [3001, 128], padded with zeros, plus the attention mask.
 */
internal object ParakeetFrontend {
    const val FRAMES = 3_001
    private const val MELS = 128
    private const val FFT = 512
    private const val BINS = FFT / 2 + 1
    private const val WINDOW = 400
    private const val HOP = 160
    private const val PREEMPHASIS = 0.97f
    private val LOG_GUARD = 2.0.pow(-24).toFloat()
    private const val EPSILON = 1e-5f
    private val window = FloatArray(FFT).also { padded ->
        val left = (FFT - WINDOW) / 2
        for (index in 0 until WINDOW) padded[left + index] = (0.5 - 0.5 * cos(2.0 * PI * index / (WINDOW - 1))).toFloat()
    }
    private val filters = slaneyFilters()

    // Each triangular filter is non-zero over a few bins only; keep [first, last] per mel.
    private val firstBin = IntArray(MELS) { mel -> (0 until BINS).firstOrNull { filters[mel * BINS + it] != 0f } ?: 0 }
    private val lastBin = IntArray(MELS) { mel -> (BINS - 1 downTo 0).firstOrNull { filters[mel * BINS + it] != 0f } ?: -1 }
    private val fft = MixedRadixFft(FFT)
    private val frame = FloatArray(FFT)
    private val power = FloatArray(BINS)

    class Features(val values: FloatArray, val mask: IntArray, val validMelFrames: Int)

    @Synchronized
    fun extract(audio: FloatArray, start: Int, end: Int): Features {
        val length = end - start
        require(length in 1..30 * 16_000) { "Window must hold 1 sample to 30 s." }
        val padded = FloatArray(length + FFT)  // FFT/2 zeros on both sides
        padded[FFT / 2] = audio[start]
        for (index in 1 until length) padded[FFT / 2 + index] = audio[start + index] - PREEMPHASIS * audio[start + index - 1]
        val frames = 1 + length / HOP
        val valid = length / HOP
        // Time-major log-mel, normalised in place.
        val values = FloatArray(FRAMES * MELS)
        for (time in 0 until frames) {
            val base = time * HOP
            for (index in 0 until FFT) frame[index] = padded[base + index] * window[index]
            fft.powerSpectrum(frame, power)
            val row = time * MELS
            for (mel in 0 until MELS) {
                var energy = 0.0
                val offset = mel * BINS
                for (bin in firstBin[mel]..lastBin[mel]) energy += filters[offset + bin] * power[bin]
                values[row + mel] = ln(energy.toFloat() + LOG_GUARD)
            }
        }
        val mask = IntArray(FRAMES)
        for (time in 0 until valid) mask[time] = 1
        if (valid > 0) {
            for (mel in 0 until MELS) {
                var sum = 0.0
                for (time in 0 until valid) sum += values[time * MELS + mel]
                val mean = sum / valid
                var squared = 0.0
                for (time in 0 until valid) {
                    val difference = values[time * MELS + mel] - mean
                    squared += difference * difference
                }
                val deviation = sqrt(squared / max(1, valid - 1)) + EPSILON
                for (time in 0 until valid) {
                    values[time * MELS + mel] = ((values[time * MELS + mel] - mean) / deviation).toFloat()
                }
            }
        }
        // Frames at or beyond `valid` (the centred tail frame and the padding) are zero.
        values.fill(0f, valid * MELS, values.size)
        return Features(values, mask, valid)
    }

    private fun slaneyFilters(): FloatArray {
        val maximumMel = hzToMel(8_000.0)
        val edges = DoubleArray(MELS + 2) { melToHz(maximumMel * it / (MELS + 1)) }
        return FloatArray(MELS * BINS).also { result ->
            for (mel in 0 until MELS) {
                val normalization = 2.0 / (edges[mel + 2] - edges[mel])
                for (bin in 0 until BINS) {
                    val frequency = 8_000.0 * bin / (BINS - 1)
                    val down = (frequency - edges[mel]) / (edges[mel + 1] - edges[mel])
                    val up = (edges[mel + 2] - frequency) / (edges[mel + 2] - edges[mel + 1])
                    result[mel * BINS + bin] = (max(0.0, min(down, up)) * normalization).toFloat()
                }
            }
        }
    }

    private fun hzToMel(hz: Double) = if (hz < 1_000.0) hz * 3.0 / 200.0 else 15.0 + ln(hz / 1_000.0) / (ln(6.4) / 27.0)
    private fun melToHz(mel: Double) = if (mel < 15.0) mel * 200.0 / 3.0 else 1_000.0 * exp((ln(6.4) / 27.0) * (mel - 15.0))
}
