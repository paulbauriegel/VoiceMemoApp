package com.example.voicememo.speaker

import com.example.voicememo.transcription.MixedRadixFft
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.pow

/**
 * Kaldi-compatible log mel filterbank features, as the CAM++ voice models
 * were trained on: 16 kHz, 25 ms Povey window, 10 ms shift, 512-point FFT,
 * 80 mel bins from 20 Hz to 8 kHz, no dither, snip_edges, pre-emphasis 0.97,
 * DC removal, power spectrum, natural log. Not thread-safe.
 */
internal class KaldiFbank {
    private val fft = MixedRadixFft(FFT_SIZE)
    private val window = FloatArray(FRAME_LENGTH) { index ->
        (0.5 - 0.5 * cos(2.0 * PI * index / (FRAME_LENGTH - 1))).pow(0.85).toFloat()
    }
    private val frame = FloatArray(FFT_SIZE)
    private val power = FloatArray(FFT_SIZE / 2 + 1)
    private val filterStart = IntArray(MEL_BINS)
    private val filterWeights: Array<FloatArray>

    init {
        val melLow = mel(LOW_FREQUENCY)
        val melHigh = mel(SAMPLE_RATE / 2.0)
        val melDelta = (melHigh - melLow) / (MEL_BINS + 1)
        val binWidth = SAMPLE_RATE.toDouble() / FFT_SIZE
        // Kaldi leaves the Nyquist bin out of the filterbank.
        val fftBins = FFT_SIZE / 2
        filterWeights = Array(MEL_BINS) { bin ->
            val left = melLow + bin * melDelta
            val centre = left + melDelta
            val right = centre + melDelta
            var first = -1
            val weights = ArrayList<Float>()
            for (index in 0 until fftBins) {
                val value = mel(binWidth * index)
                if (value > left && value < right) {
                    if (first < 0) first = index
                    val weight = if (value <= centre) {
                        (value - left) / (centre - left)
                    } else {
                        (right - value) / (right - centre)
                    }
                    weights += weight.toFloat()
                }
            }
            filterStart[bin] = max(first, 0)
            weights.toFloatArray()
        }
    }

    fun frameCount(sampleCount: Int): Int =
        if (sampleCount < FRAME_LENGTH) 0 else 1 + (sampleCount - FRAME_LENGTH) / FRAME_SHIFT

    /**
     * Features of [samples] (range -1..1, scaled by [scale] first), row-major
     * `[frameCount, MEL_BINS]`. With [subtractMean] each bin's mean over the
     * frames is removed (cepstral mean normalisation).
     */
    fun compute(
        samples: FloatArray,
        offset: Int = 0,
        length: Int = samples.size - offset,
        scale: Float = 1f,
        subtractMean: Boolean = false,
    ): FloatArray {
        require(offset >= 0 && length >= 0 && offset + length <= samples.size)
        val frames = frameCount(length)
        val features = FloatArray(frames * MEL_BINS)
        for (frameIndex in 0 until frames) {
            val start = offset + frameIndex * FRAME_SHIFT
            var sum = 0f
            for (index in 0 until FRAME_LENGTH) {
                val value = samples[start + index] * scale
                frame[index] = value
                sum += value
            }
            val mean = sum / FRAME_LENGTH
            for (index in 0 until FRAME_LENGTH) frame[index] -= mean
            for (index in FRAME_LENGTH - 1 downTo 1) {
                frame[index] -= PREEMPHASIS * frame[index - 1]
            }
            frame[0] -= PREEMPHASIS * frame[0]
            for (index in 0 until FRAME_LENGTH) frame[index] *= window[index]
            frame.fill(0f, FRAME_LENGTH, FFT_SIZE)
            fft.powerSpectrum(frame, power)
            val row = frameIndex * MEL_BINS
            for (bin in 0 until MEL_BINS) {
                val weights = filterWeights[bin]
                val first = filterStart[bin]
                var energy = 0f
                for (index in weights.indices) energy += weights[index] * power[first + index]
                features[row + bin] = ln(max(energy, FLOAT_EPSILON))
            }
        }
        if (subtractMean && frames > 0) {
            for (bin in 0 until MEL_BINS) {
                var sum = 0.0
                for (frameIndex in 0 until frames) sum += features[frameIndex * MEL_BINS + bin]
                val mean = (sum / frames).toFloat()
                for (frameIndex in 0 until frames) features[frameIndex * MEL_BINS + bin] -= mean
            }
        }
        return features
    }

    companion object {
        const val SAMPLE_RATE = 16_000
        const val MEL_BINS = 80
        const val FRAME_LENGTH = 400
        const val FRAME_SHIFT = 160
        private const val FFT_SIZE = 512
        private const val LOW_FREQUENCY = 20.0
        private const val PREEMPHASIS = 0.97f
        private const val FLOAT_EPSILON = 1.1920929e-7f

        private fun mel(frequency: Double): Double = 1127.0 * ln(1.0 + frequency / 700.0)
    }
}
