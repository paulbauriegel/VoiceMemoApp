package com.example.voicememo.transcription

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

/**
 * Allocation-free Cooley-Tukey FFT specialized for sizes factorizable by 2
 * and 5. Whisper's 400-point transform is 2^4 * 5^2.
 */
internal class MixedRadixFft(private val size: Int) {
    private val outputReal = FloatArray(size)
    private val outputImaginary = FloatArray(size)
    private val scratchReal = FloatArray(size)
    private val scratchImaginary = FloatArray(size)
    private val zeroImaginary = FloatArray(size)
    private val twiddles = mutableMapOf<Int, TwiddleTable>()

    init {
        require(size > 0 && hasOnlyTwoAndFiveFactors(size)) {
            "FFT size must factor into powers of 2 and 5."
        }
        prepareTwiddles(size)
    }

    fun powerSpectrum(realInput: FloatArray, destination: FloatArray) {
        require(realInput.size == size)
        require(destination.size >= size / 2 + 1)
        transformRecursive(
            sourceReal = realInput,
            sourceImaginary = zeroImaginary,
            sourceOffset = 0,
            sourceStride = 1,
            destinationReal = outputReal,
            destinationImaginary = outputImaginary,
            destinationOffset = 0,
            workReal = scratchReal,
            workImaginary = scratchImaginary,
            workOffset = 0,
            length = size,
        )
        for (bin in 0..size / 2) {
            val real = outputReal[bin]
            val imaginary = outputImaginary[bin]
            destination[bin] = real * real + imaginary * imaginary
        }
    }

    private fun transformRecursive(
        sourceReal: FloatArray,
        sourceImaginary: FloatArray,
        sourceOffset: Int,
        sourceStride: Int,
        destinationReal: FloatArray,
        destinationImaginary: FloatArray,
        destinationOffset: Int,
        workReal: FloatArray,
        workImaginary: FloatArray,
        workOffset: Int,
        length: Int,
    ) {
        if (length == 1) {
            destinationReal[destinationOffset] = sourceReal[sourceOffset]
            destinationImaginary[destinationOffset] = sourceImaginary[sourceOffset]
            return
        }

        val radix = if (length % 2 == 0) 2 else 5
        val subLength = length / radix
        for (radixIndex in 0 until radix) {
            transformRecursive(
                sourceReal = sourceReal,
                sourceImaginary = sourceImaginary,
                sourceOffset = sourceOffset + radixIndex * sourceStride,
                sourceStride = sourceStride * radix,
                destinationReal = workReal,
                destinationImaginary = workImaginary,
                destinationOffset = workOffset + radixIndex * subLength,
                workReal = destinationReal,
                workImaginary = destinationImaginary,
                workOffset = destinationOffset + radixIndex * subLength,
                length = subLength,
            )
        }

        val table = requireNotNull(twiddles[length])
        for (frequency in 0 until length) {
            val subFrequency = frequency % subLength
            var sumReal = 0.0f
            var sumImaginary = 0.0f
            for (radixIndex in 0 until radix) {
                val sourceIndex = workOffset + radixIndex * subLength + subFrequency
                val twiddleIndex = radixIndex * length + frequency
                val cosine = table.cosine[twiddleIndex]
                val sine = table.sine[twiddleIndex]
                val real = workReal[sourceIndex]
                val imaginary = workImaginary[sourceIndex]
                sumReal += real * cosine - imaginary * sine
                sumImaginary += real * sine + imaginary * cosine
            }
            destinationReal[destinationOffset + frequency] = sumReal
            destinationImaginary[destinationOffset + frequency] = sumImaginary
        }
    }

    private fun prepareTwiddles(length: Int) {
        if (length == 1 || twiddles.containsKey(length)) return
        val radix = if (length % 2 == 0) 2 else 5
        val cosine = FloatArray(radix * length)
        val sine = FloatArray(radix * length)
        for (radixIndex in 0 until radix) {
            for (frequency in 0 until length) {
                val angle = -2.0 * PI * radixIndex * frequency / length
                val index = radixIndex * length + frequency
                cosine[index] = cos(angle).toFloat()
                sine[index] = sin(angle).toFloat()
            }
        }
        twiddles[length] = TwiddleTable(cosine, sine)
        prepareTwiddles(length / radix)
    }

    private data class TwiddleTable(
        val cosine: FloatArray,
        val sine: FloatArray,
    )

    private fun hasOnlyTwoAndFiveFactors(value: Int): Boolean {
        var remainder = value
        while (remainder % 2 == 0) remainder /= 2
        while (remainder % 5 == 0) remainder /= 5
        return remainder == 1
    }
}
