package com.example.voicememo.importing

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.roundToInt

internal data class PlatformPcmFormat(
    val sampleRate: Int,
    val channelCount: Int,
    val encoding: Int,
) {
    val bytesPerSample: Int
        get() = when (encoding) {
            PCM_ENCODING_8_BIT -> 1
            PCM_ENCODING_16_BIT -> 2
            PCM_ENCODING_FLOAT, PCM_ENCODING_32_BIT -> 4
            PCM_ENCODING_24_BIT_PACKED -> 3
            else -> throw IllegalArgumentException("Unsupported decoded PCM encoding: $encoding.")
        }

    val bytesPerFrame: Int
        get() = bytesPerSample * channelCount

    init {
        require(sampleRate > 0)
        require(channelCount in 1..32) { "Unsupported decoded channel count: $channelCount." }
        bytesPerSample
    }
}

/**
 * Converts platform decoder output into streaming, interleaved-downmixed mono
 * PCM16. Codec buffers are normally frame-aligned, but a small remainder is
 * retained defensively when a device splits one PCM frame across buffers.
 */
internal class PcmStreamConverter(
    private val sink: DecodedPcmSink,
) {
    private var activeFormat: PlatformPcmFormat? = null
    private var remainder = ByteArray(0)

    fun consume(
        source: ByteBuffer,
        format: PlatformPcmFormat,
    ) {
        val previousFormat = activeFormat
        if (previousFormat != null && previousFormat != format) {
            require(remainder.isEmpty()) {
                "The decoder changed PCM format in the middle of a frame."
            }
        }
        activeFormat = format

        val bytes = ByteArray(remainder.size + source.remaining())
        remainder.copyInto(bytes)
        source.get(bytes, remainder.size, source.remaining())

        val frameBytes = format.bytesPerFrame
        val completeBytes = bytes.size - (bytes.size % frameBytes)
        if (completeBytes == 0) {
            remainder = bytes
            return
        }

        val input = ByteBuffer.wrap(bytes, 0, completeBytes).order(ByteOrder.LITTLE_ENDIAN)
        val output = ShortArray(completeBytes / frameBytes)
        for (frame in output.indices) {
            output[frame] = when (format.encoding) {
                PCM_ENCODING_8_BIT -> {
                    var sum = 0L
                    repeat(format.channelCount) {
                        sum += ((input.get().toInt() and 0xff) - 128) shl 8
                    }
                    (sum / format.channelCount).coerceToShort()
                }

                PCM_ENCODING_16_BIT -> {
                    var sum = 0L
                    repeat(format.channelCount) { sum += input.short.toLong() }
                    (sum / format.channelCount).coerceToShort()
                }

                PCM_ENCODING_FLOAT -> {
                    var sum = 0.0
                    repeat(format.channelCount) {
                        val value = input.float
                        if (value.isFinite()) sum += value else sum += 0.0
                    }
                    floatToPcm16((sum / format.channelCount).toFloat())
                }

                PCM_ENCODING_24_BIT_PACKED -> {
                    var sum = 0L
                    repeat(format.channelCount) { sum += input.readSigned24Le() }
                    ((sum / format.channelCount) shr 8).coerceToShort()
                }

                PCM_ENCODING_32_BIT -> {
                    var sum = 0L
                    repeat(format.channelCount) { sum += input.int.toLong() }
                    ((sum / format.channelCount) shr 16).coerceToShort()
                }

                else -> error("Validated by PlatformPcmFormat")
            }
        }
        sink.write(format.sampleRate, output, output.size)
        remainder = if (completeBytes == bytes.size) {
            ByteArray(0)
        } else {
            bytes.copyOfRange(completeBytes, bytes.size)
        }
    }

    fun finish() {
        require(remainder.isEmpty()) { "The decoder ended in the middle of a PCM frame." }
    }

    private fun ByteBuffer.readSigned24Le(): Long {
        val raw = (get().toInt() and 0xff) or
            ((get().toInt() and 0xff) shl 8) or
            ((get().toInt() and 0xff) shl 16)
        return if ((raw and 0x80_0000) != 0) {
            (raw or -0x100_0000).toLong()
        } else {
            raw.toLong()
        }
    }

    private fun Long.coerceToShort(): Short =
        coerceIn(Short.MIN_VALUE.toLong(), Short.MAX_VALUE.toLong()).toShort()

    private fun floatToPcm16(value: Float): Short {
        val normalized = value.coerceIn(-1f, 1f)
        val scale = if (normalized < 0f) 32_768f else 32_767f
        return (normalized * scale).roundToInt()
            .coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt())
            .toShort()
    }
}

internal const val PCM_ENCODING_16_BIT = 2
internal const val PCM_ENCODING_8_BIT = 3
internal const val PCM_ENCODING_FLOAT = 4
internal const val PCM_ENCODING_24_BIT_PACKED = 21
internal const val PCM_ENCODING_32_BIT = 22
