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
 * NeMo/Google LiteRT-compatible Parakeet frontend.
 *
 * Produces `[1, 128, 500]` natural-log Slaney mel features with pre-emphasis
 * and per-mel standardization.
 */
internal class ParakeetFeatureExtractor {
    private val fft = MixedRadixFft(FFT_SIZE)
    private val frame = FloatArray(FFT_SIZE)
    private val power = FloatArray(FFT_BINS)
    private val window = FloatArray(FFT_SIZE) { index ->
        (0.5 - 0.5 * cos(2.0 * PI * index / FFT_SIZE)).toFloat()
    }
    private val melFilters = createSlaneyMelFilterbank()

    @Synchronized
    fun extract(audio16Khz: FloatArray): FloatArray {
        require(audio16Khz.size <= ParakeetAssets.MAX_SAMPLES) {
            "Parakeet accepts at most five seconds of 16 kHz PCM."
        }
        if (audio16Khz.isEmpty()) return FloatArray(MEL_BINS * FRAMES)

        val emphasized = FloatArray(audio16Khz.size)
        emphasized[0] = audio16Khz[0]
        for (index in 1 until audio16Khz.size) {
            emphasized[index] = audio16Khz[index] - PREEMPHASIS * audio16Khz[index - 1]
        }

        // Match librosa/JLibrosa center=true reflect padding.
        val centered = FloatArray(emphasized.size + FFT_SIZE)
        emphasized.copyInto(centered, destinationOffset = REFLECT_PAD)
        for (index in 0 until REFLECT_PAD) {
            val left = min(index + 1, emphasized.lastIndex)
            val right = max(0, emphasized.lastIndex - 1 - index)
            centered[REFLECT_PAD - 1 - index] = emphasized[left]
            centered[REFLECT_PAD + emphasized.size + index] = emphasized[right]
        }

        val validFrames = (audio16Khz.size / HOP_LENGTH).coerceIn(1, FRAMES)
        val result = FloatArray(MEL_BINS * FRAMES)
        for (time in 0 until validFrames) {
            val frameStart = time * HOP_LENGTH
            for (sample in 0 until FFT_SIZE) {
                frame[sample] = centered[frameStart + sample] * window[sample]
            }
            fft.powerSpectrum(frame, power)
            for (mel in 0 until MEL_BINS) {
                var energy = 0.0
                val filterOffset = mel * FFT_BINS
                for (bin in 0 until FFT_BINS) {
                    energy += melFilters[filterOffset + bin] * power[bin]
                }
                result[mel * FRAMES + time] =
                    ln(energy.toFloat() + LOG_ZERO_GUARD)
            }
        }

        for (mel in 0 until MEL_BINS) {
            val offset = mel * FRAMES
            var sum = 0.0
            for (time in 0 until validFrames) sum += result[offset + time]
            val mean = (sum / validFrames).toFloat()
            var squared = 0.0
            for (time in 0 until validFrames) {
                val difference = result[offset + time] - mean
                squared += difference * difference
            }
            val standardDeviation =
                sqrt(squared / max(1, validFrames - 1)).toFloat() + EPSILON
            for (time in 0 until validFrames) {
                result[offset + time] =
                    (result[offset + time] - mean) / standardDeviation
            }
        }
        return result
    }

    private fun createSlaneyMelFilterbank(): FloatArray {
        val edgesHz = DoubleArray(MEL_BINS + 2)
        val maximumMel = hzToSlaneyMel(ParakeetAssets.SAMPLE_RATE / 2.0)
        for (index in edgesHz.indices) {
            edgesHz[index] = slaneyMelToHz(maximumMel * index / (MEL_BINS + 1))
        }
        return FloatArray(MEL_BINS * FFT_BINS).also { filters ->
            for (mel in 0 until MEL_BINS) {
                val lower = edgesHz[mel]
                val center = edgesHz[mel + 1]
                val upper = edgesHz[mel + 2]
                val normalization = 2.0 / (upper - lower)
                for (bin in 0 until FFT_BINS) {
                    val frequency =
                        (ParakeetAssets.SAMPLE_RATE / 2.0) * bin / (FFT_BINS - 1)
                    val lowerSlope = (frequency - lower) / (center - lower)
                    val upperSlope = (upper - frequency) / (upper - center)
                    filters[mel * FFT_BINS + bin] =
                        (max(0.0, min(lowerSlope, upperSlope)) * normalization).toFloat()
                }
            }
        }
    }

    private fun hzToSlaneyMel(hertz: Double): Double =
        if (hertz < MIN_LOG_HZ) {
            hertz / LINEAR_HZ_PER_MEL
        } else {
            MIN_LOG_MEL + ln(hertz / MIN_LOG_HZ) / LOG_STEP
        }

    private fun slaneyMelToHz(mel: Double): Double =
        if (mel < MIN_LOG_MEL) {
            mel * LINEAR_HZ_PER_MEL
        } else {
            MIN_LOG_HZ * exp(LOG_STEP * (mel - MIN_LOG_MEL))
        }

    private companion object {
        const val FFT_SIZE = 512
        const val FFT_BINS = FFT_SIZE / 2 + 1
        const val MEL_BINS = 128
        const val FRAMES = 500
        const val HOP_LENGTH = 160
        const val REFLECT_PAD = FFT_SIZE / 2
        const val PREEMPHASIS = 0.97f
        const val EPSILON = 1.0e-5f
        val LOG_ZERO_GUARD = 2.0.pow(-24).toFloat()

        const val MIN_LOG_HZ = 1_000.0
        const val MIN_LOG_MEL = 15.0
        const val LINEAR_HZ_PER_MEL = 200.0 / 3.0
        const val LOG_STEP = 0.06875177742094912
    }
}
