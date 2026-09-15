package com.example.voicememo.transcription

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min

/**
 * OpenAI/Hugging Face-compatible Whisper large-v3 frontend.
 *
 * Output layout is [mel][frame], matching the LiteRT encode tensor
 * `[1, 128, 3000]`.
 */
internal fun interface WhisperFeatureProvider {
    fun extract(pcm16Khz: FloatArray): FloatArray
}

class WhisperFeatureExtractor : WhisperFeatureProvider {
    private val fft = MixedRadixFft(FFT_SIZE)
    private val frame = FloatArray(FFT_SIZE)
    private val power = FloatArray(FFT_BINS)
    private val window = FloatArray(FFT_SIZE) { index ->
        // Periodic Hann: torch.hann_window(400, periodic=true).
        (0.5 - 0.5 * cos(2.0 * PI * index / FFT_SIZE)).toFloat()
    }
    private val melFilters = createSlaneyMelFilterbank()
    private val melRanges = Array(MEL_BINS) { mel ->
        var first = 0
        while (first < FFT_BINS && melFilters[mel * FFT_BINS + first] == 0.0f) first++
        var lastExclusive = FFT_BINS
        while (
            lastExclusive > first &&
            melFilters[mel * FFT_BINS + lastExclusive - 1] == 0.0f
        ) {
            lastExclusive--
        }
        first until lastExclusive
    }

    @Synchronized
    override fun extract(pcm16Khz: FloatArray): FloatArray {
        val fixedAudio = FloatArray(MAX_SAMPLES)
        pcm16Khz.copyInto(
            destination = fixedAudio,
            endIndex = min(pcm16Khz.size, MAX_SAMPLES),
        )

        // torch.stft(center=true, pad_mode="reflect") after right padding the
        // waveform to exactly 30 seconds.
        val centered = FloatArray(MAX_SAMPLES + FFT_SIZE)
        fixedAudio.copyInto(centered, destinationOffset = REFLECT_PAD)
        for (index in 0 until REFLECT_PAD) {
            centered[REFLECT_PAD - 1 - index] = fixedAudio[index + 1]
            centered[REFLECT_PAD + MAX_SAMPLES + index] =
                fixedAudio[MAX_SAMPLES - 2 - index]
        }

        val result = FloatArray(MEL_BINS * FRAMES)
        var maximum = Float.NEGATIVE_INFINITY
        for (time in 0 until FRAMES) {
            val frameStart = time * HOP_LENGTH
            for (sample in 0 until FFT_SIZE) {
                frame[sample] = centered[frameStart + sample] * window[sample]
            }
            fft.powerSpectrum(frame, power)

            for (mel in 0 until MEL_BINS) {
                var energy = 0.0
                val filterOffset = mel * FFT_BINS
                for (bin in melRanges[mel]) {
                    energy += melFilters[filterOffset + bin] * power[bin]
                }
                val value = log10(max(energy.toFloat(), LOG_FLOOR))
                result[mel * FRAMES + time] = value
                maximum = max(maximum, value)
            }
        }

        val dynamicFloor = maximum - 8.0f
        for (index in result.indices) {
            result[index] = (max(result[index], dynamicFloor) + 4.0f) / 4.0f
        }
        return result
    }

    /** Exposed to golden tests without allowing callers to mutate cached data. */
    internal fun melFilterbankCopy(): FloatArray = melFilters.copyOf()

    private fun createSlaneyMelFilterbank(): FloatArray {
        val edgesHz = DoubleArray(MEL_BINS + 2)
        val minimumMel = hzToSlaneyMel(0.0)
        val maximumMel = hzToSlaneyMel(SAMPLE_RATE / 2.0)
        for (index in edgesHz.indices) {
            val mel = minimumMel +
                (maximumMel - minimumMel) * index / (MEL_BINS + 1).toDouble()
            edgesHz[index] = slaneyMelToHz(mel)
        }

        val filters = FloatArray(MEL_BINS * FFT_BINS)
        for (mel in 0 until MEL_BINS) {
            val lower = edgesHz[mel]
            val center = edgesHz[mel + 1]
            val upper = edgesHz[mel + 2]
            val slaneyNormalization = 2.0 / (upper - lower)
            for (bin in 0 until FFT_BINS) {
                val frequency = (SAMPLE_RATE / 2.0) * bin / (FFT_BINS - 1)
                val lowerSlope = (frequency - lower) / (center - lower)
                val upperSlope = (upper - frequency) / (upper - center)
                filters[mel * FFT_BINS + bin] =
                    (max(0.0, min(lowerSlope, upperSlope)) * slaneyNormalization).toFloat()
            }
        }
        return filters
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

    companion object {
        const val SAMPLE_RATE = 16_000
        const val MAX_SAMPLES = SAMPLE_RATE * 30
        const val MEL_BINS = 128
        const val FRAMES = 3_000
        const val FFT_SIZE = 400
        const val HOP_LENGTH = 160
        private const val FFT_BINS = FFT_SIZE / 2 + 1
        private const val REFLECT_PAD = FFT_SIZE / 2
        private const val LOG_FLOOR = 1.0e-10f

        private const val MIN_LOG_HZ = 1_000.0
        private const val MIN_LOG_MEL = 15.0
        private const val LINEAR_HZ_PER_MEL = 200.0 / 3.0
        private const val LOG_STEP = 0.06875177742094912 // ln(6.4) / 27
    }
}
