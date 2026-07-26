package com.example.voicememo.recording

import java.io.Closeable
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Writes a seekable mono PCM16 WAV and always leaves a valid header after [finish].
 */
class WavFileWriter(
    file: File,
    val sampleRate: Int,
) : Closeable {
    private val output = RandomAccessFile(file, "rw")
    private val scratch = ByteArray(16 * 1024)
    private var bytesWritten = 0L
    private var finished = false

    init {
        require(sampleRate > 0)
        output.setLength(0)
        output.write(ByteArray(HEADER_SIZE))
    }

    val frameCount: Long
        get() = bytesWritten / BYTES_PER_SAMPLE

    val durationMillis: Long
        get() = frameCount * 1_000L / sampleRate

    @Synchronized
    fun write(samples: ShortArray, count: Int = samples.size) {
        check(!finished) { "WAV writer is already finalized" }
        require(count in 0..samples.size)

        var sampleOffset = 0
        while (sampleOffset < count) {
            val batch = minOf(count - sampleOffset, scratch.size / BYTES_PER_SAMPLE)
            var byteOffset = 0
            repeat(batch) {
                val value = samples[sampleOffset + it].toInt()
                scratch[byteOffset++] = (value and 0xff).toByte()
                scratch[byteOffset++] = ((value ushr 8) and 0xff).toByte()
            }
            output.write(scratch, 0, byteOffset)
            bytesWritten += byteOffset
            sampleOffset += batch
        }
    }

    @Synchronized
    fun finish() {
        if (finished) return
        finished = true

        val dataLength = bytesWritten.coerceAtMost(UInt.MAX_VALUE.toLong())
        val riffLength = (36L + dataLength).coerceAtMost(UInt.MAX_VALUE.toLong())
        val header = ByteBuffer.allocate(HEADER_SIZE).order(ByteOrder.LITTLE_ENDIAN).apply {
            put("RIFF".toByteArray(Charsets.US_ASCII))
            putInt(riffLength.toInt())
            put("WAVE".toByteArray(Charsets.US_ASCII))
            put("fmt ".toByteArray(Charsets.US_ASCII))
            putInt(16)
            putShort(1.toShort())
            putShort(CHANNELS.toShort())
            putInt(sampleRate)
            putInt(sampleRate * CHANNELS * BYTES_PER_SAMPLE)
            putShort((CHANNELS * BYTES_PER_SAMPLE).toShort())
            putShort(BITS_PER_SAMPLE.toShort())
            put("data".toByteArray(Charsets.US_ASCII))
            putInt(dataLength.toInt())
        }.array()

        output.seek(0)
        output.write(header)
        output.fd.sync()
        output.close()
    }

    override fun close() = finish()

    private companion object {
        const val HEADER_SIZE = 44
        const val CHANNELS = 1
        const val BITS_PER_SAMPLE = 16
        const val BYTES_PER_SAMPLE = 2
    }
}
