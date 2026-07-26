package com.example.voicememo.recording

import java.io.BufferedOutputStream
import java.io.DataOutputStream
import java.io.DataInputStream
import java.io.BufferedInputStream
import java.io.File

/**
 * Builds deterministic 50 ms min/max buckets and stores a compact binary sidecar.
 */
class WaveformAccumulator(
    sampleRate: Int,
    private val bucketMillis: Int = 50,
) {
    private val samplesPerBucket = maxOf(1, sampleRate * bucketMillis / 1_000)
    private val completed = mutableListOf<WaveformBucket>()
    private var count = 0
    private var minimum = Short.MAX_VALUE
    private var maximum = Short.MIN_VALUE

    fun add(samples: ShortArray, length: Int = samples.size): List<WaveformBucket> {
        require(length in 0..samples.size)
        repeat(length) { index ->
            val value = samples[index]
            if (value < minimum) minimum = value
            if (value > maximum) maximum = value
            count++
            if (count == samplesPerBucket) flushBucket()
        }
        return snapshot()
    }

    fun snapshot(): List<WaveformBucket> {
        if (count == 0) return completed.toList()
        return completed + WaveformBucket(minimum, maximum)
    }

    fun finishTo(file: File): List<WaveformBucket> {
        if (count > 0) flushBucket()
        file.parentFile?.mkdirs()
        DataOutputStream(BufferedOutputStream(file.outputStream())).use { output ->
            output.write(MAGIC)
            output.writeInt(VERSION)
            output.writeInt(bucketMillis)
            output.writeInt(completed.size)
            completed.forEach { bucket ->
                output.writeShort(bucket.minimum.toInt())
                output.writeShort(bucket.maximum.toInt())
            }
        }
        return completed.toList()
    }

    private fun flushBucket() {
        completed += WaveformBucket(minimum, maximum)
        count = 0
        minimum = Short.MAX_VALUE
        maximum = Short.MIN_VALUE
    }

    private companion object {
        val MAGIC = byteArrayOf('V'.code.toByte(), 'W'.code.toByte(), 'F'.code.toByte(), '1'.code.toByte())
        const val VERSION = 1
    }
}

object WaveformSidecar {
    fun read(file: File): List<WaveformBucket> {
        if (!file.isFile) return emptyList()
        return runCatching {
            DataInputStream(BufferedInputStream(file.inputStream())).use { input ->
                val magic = ByteArray(4).also(input::readFully)
                require(magic.contentEquals(byteArrayOf('V'.code.toByte(), 'W'.code.toByte(), 'F'.code.toByte(), '1'.code.toByte())))
                require(input.readInt() == 1) { "Unsupported waveform version" }
                input.readInt() // bucket duration; retained in the on-disk contract
                val count = input.readInt()
                require(count in 0..MAX_BUCKETS) { "Invalid waveform bucket count" }
                List(count) {
                    WaveformBucket(
                        minimum = input.readShort(),
                        maximum = input.readShort(),
                    )
                }
            }
        }.getOrDefault(emptyList())
    }

    fun normalized(file: File, maximumBuckets: Int = 160): List<Float> {
        require(maximumBuckets > 0)
        val source = read(file)
        if (source.isEmpty()) return emptyList()
        val stride = maxOf(1, (source.size + maximumBuckets - 1) / maximumBuckets)
        return source.chunked(stride).map { group ->
            group.maxOf { bucket ->
                maxOf(
                    kotlin.math.abs(bucket.minimum.toInt()),
                    kotlin.math.abs(bucket.maximum.toInt()),
                )
            } / Short.MAX_VALUE.toFloat()
        }
    }

    private const val MAX_BUCKETS = 2_000_000
}
