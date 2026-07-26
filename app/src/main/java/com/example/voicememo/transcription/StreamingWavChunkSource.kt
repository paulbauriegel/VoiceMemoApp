package com.example.voicememo.transcription

import java.io.Closeable
import java.io.EOFException
import java.io.File
import java.io.RandomAccessFile
import kotlin.math.PI
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.sin

/**
 * Metadata for a WAV file opened by [StreamingWavChunkSource].
 *
 * [whisperSampleCount] is the exact number of mono 16 kHz samples produced by
 * the streaming resampler. The source file is opened read-only and is never
 * rewritten or replaced.
 */
data class StreamingWavInfo(
    val sourceSampleRate: Int,
    val sourceChannels: Int,
    val sourceFrameCount: Long,
    val whisperSampleCount: Int,
) {
    val durationMillis: Long
        get() = sourceFrameCount * 1_000L / sourceSampleRate
}

/**
 * One fixed-size Whisper payload. Only this array and small reusable decode
 * buffers are resident; earlier chunks can be released before the next one is
 * requested.
 */
data class StreamingWavChunk(
    val window: AudioChunkWindow,
    val samples: FloatArray,
)

data class StreamingWavProgress(
    val chunkIndex: Int,
    val chunkCount: Int,
    val chunkSamplesRead: Int,
    val chunkSampleCount: Int,
    val workSamplesRead: Long,
    val totalWorkSamples: Long,
) {
    val fraction: Float
        get() = if (totalWorkSamples == 0L) {
            1.0f
        } else {
            (workSamplesRead.toDouble() / totalWorkSamples)
                .coerceIn(0.0, 1.0)
                .toFloat()
        }
}

/**
 * Bounded-memory WAV to Whisper-window source.
 *
 * PCM16 and IEEE float32 WAV data are decoded in bulk, downmixed to mono, and
 * resampled into 28-second payloads with the same two-second overlap used by
 * [WhisperChunker]. Resampling uses a precomputed polyphase Blackman-windowed
 * sinc kernel, so there is no trigonometry in the per-sample hot loop.
 *
 * Instances are intentionally single-reader and not thread safe.
 */
class StreamingWavChunkSource private constructor(
    private val input: RandomAccessFile,
    private val format: WavFormat,
    private val dataOffset: Long,
    val info: StreamingWavInfo,
) : Closeable {
    val chunkCount: Int =
        windowCount(info.whisperSampleCount)

    private val kernelBank: SincKernelBank? =
        if (format.sampleRate == TARGET_SAMPLE_RATE) null else SincKernelBank(format.sampleRate)
    private var rawBuffer = ByteArray(0)
    private var monoBuffer = FloatArray(0)
    private var closed = false

    /**
     * Reads a single payload by index. This supports durable resume without
     * decoding earlier chunks again.
     */
    fun readChunk(
        index: Int,
        shouldCancel: () -> Boolean = { false },
        onProgress: (StreamingWavProgress) -> Unit = {},
    ): StreamingWavChunk {
        check(!closed) { "The WAV chunk source is closed." }
        val window = window(index)
        val samples = FloatArray(window.sampleCount)
        var outputOffset = 0
        while (outputOffset < samples.size) {
            if (shouldCancel()) throw WhisperCancelledException()
            val batchSize = minOf(OUTPUT_BATCH_SAMPLES, samples.size - outputOffset)
            val globalOutputStart = window.startSample.toLong() + outputOffset
            if (format.sampleRate == TARGET_SAMPLE_RATE) {
                readMonoFrames(
                    firstFrame = globalOutputStart,
                    frameCount = batchSize,
                )
                monoBuffer.copyInto(
                    destination = samples,
                    destinationOffset = outputOffset,
                    startIndex = 0,
                    endIndex = batchSize,
                )
            } else {
                resampleBatch(
                    globalOutputStart = globalOutputStart,
                    output = samples,
                    outputOffset = outputOffset,
                    outputCount = batchSize,
                )
            }
            outputOffset += batchSize
            val completedBefore = index.toLong() * WhisperChunker.PAYLOAD_SAMPLES
            onProgress(
                StreamingWavProgress(
                    chunkIndex = index,
                    chunkCount = chunkCount,
                    chunkSamplesRead = outputOffset,
                    chunkSampleCount = samples.size,
                    workSamplesRead = completedBefore + outputOffset,
                    totalWorkSamples = totalWorkSamples(),
                ),
            )
        }
        return StreamingWavChunk(window, samples)
    }

    /**
     * Streams chunks to [onChunk]. Callers should finish using a chunk before
     * returning from the callback rather than retaining every [FloatArray].
     */
    fun forEachChunk(
        startChunkIndex: Int = 0,
        shouldCancel: () -> Boolean = { false },
        onProgress: (StreamingWavProgress) -> Unit = {},
        onChunk: (StreamingWavChunk) -> Unit,
    ) {
        require(startChunkIndex in 0..chunkCount) { "Invalid start chunk index." }
        for (index in startChunkIndex until chunkCount) {
            if (shouldCancel()) throw WhisperCancelledException()
            onChunk(readChunk(index, shouldCancel, onProgress))
        }
    }

    fun window(index: Int): AudioChunkWindow {
        require(index in 0 until chunkCount) { "Invalid chunk index $index of $chunkCount." }
        val start = index.toLong() * WhisperChunker.STRIDE_SAMPLES
        val end = minOf(
            start + WhisperChunker.PAYLOAD_SAMPLES,
            info.whisperSampleCount.toLong(),
        )
        return AudioChunkWindow(index, start.toInt(), end.toInt())
    }

    override fun close() {
        if (!closed) {
            closed = true
            input.close()
            rawBuffer = ByteArray(0)
            monoBuffer = FloatArray(0)
        }
    }

    private fun resampleBatch(
        globalOutputStart: Long,
        output: FloatArray,
        outputOffset: Int,
        outputCount: Int,
    ) {
        val bank = checkNotNull(kernelBank)
        val firstCenter = sourceCenter(globalOutputStart)
        val lastCenter = sourceCenter(globalOutputStart + outputCount - 1L)
        val firstFrame = maxOf(0L, firstCenter - bank.radius)
        val frameAfterLast = minOf(
            info.sourceFrameCount,
            lastCenter + bank.radius + 1L,
        )
        val framesToRead = (frameAfterLast - firstFrame).toInt()
        readMonoFrames(firstFrame, framesToRead)

        val initialNumerator = globalOutputStart * format.sampleRate
        var center = initialNumerator / TARGET_SAMPLE_RATE
        var phase = ((initialNumerator % TARGET_SAMPLE_RATE) / bank.rateGcd).toInt()
        repeat(outputCount) { localOutputIndex ->
            var weighted = 0.0
            var includedWeight = 0.0
            val kernelOffset = phase * bank.tapCount
            for (tap in 0 until bank.tapCount) {
                val sourceFrame = center + tap - bank.radius
                if (sourceFrame < 0L || sourceFrame >= info.sourceFrameCount) continue
                val sourceBufferIndex = (sourceFrame - firstFrame).toInt()
                val weight = bank.weights[kernelOffset + tap]
                weighted += monoBuffer[sourceBufferIndex] * weight
                includedWeight += weight
            }
            output[outputOffset + localOutputIndex] =
                if (kotlin.math.abs(includedWeight) < MIN_WEIGHT) {
                    0.0f
                } else {
                    (weighted / includedWeight).toFloat().coerceIn(-1.0f, 1.0f)
                }

            center += bank.wholeFrameStep
            phase += bank.phaseStep
            if (phase >= bank.phaseCount) {
                phase -= bank.phaseCount
                center++
            }
        }
    }

    /**
     * Reads interleaved source frames in one bulk operation and downmixes into
     * [monoBuffer]. There are no single-byte file reads in the audio hot path.
     */
    private fun readMonoFrames(firstFrame: Long, frameCount: Int) {
        require(firstFrame >= 0L && frameCount >= 0)
        require(firstFrame + frameCount <= info.sourceFrameCount)
        if (frameCount == 0) return
        val byteCount = Math.multiplyExact(frameCount, format.blockAlign)
        ensureRawCapacity(byteCount)
        ensureMonoCapacity(frameCount)
        input.seek(dataOffset + firstFrame * format.blockAlign)
        input.readFully(rawBuffer, 0, byteCount)

        var byteOffset = 0
        for (frame in 0 until frameCount) {
            var sum = 0.0f
            repeat(format.channels) {
                val sample = when (format.audioFormat) {
                    PCM_FORMAT -> {
                        val bits =
                            (rawBuffer[byteOffset].toInt() and 0xff) or
                                (rawBuffer[byteOffset + 1].toInt() shl 8)
                        byteOffset += 2
                        bits.toShort() / 32768.0f
                    }

                    IEEE_FLOAT_FORMAT -> {
                        val bits =
                            (rawBuffer[byteOffset].toInt() and 0xff) or
                                ((rawBuffer[byteOffset + 1].toInt() and 0xff) shl 8) or
                                ((rawBuffer[byteOffset + 2].toInt() and 0xff) shl 16) or
                                (rawBuffer[byteOffset + 3].toInt() shl 24)
                        byteOffset += 4
                        Float.fromBits(bits).takeIf(Float::isFinite) ?: 0.0f
                    }

                    else -> error("WAV format was validated when opened.")
                }
                sum += sample
            }
            monoBuffer[frame] = (sum / format.channels).coerceIn(-1.0f, 1.0f)
        }
    }

    private fun ensureRawCapacity(required: Int) {
        if (rawBuffer.size < required) rawBuffer = ByteArray(required)
    }

    private fun ensureMonoCapacity(required: Int) {
        if (monoBuffer.size < required) monoBuffer = FloatArray(required)
    }

    private fun sourceCenter(outputSample: Long): Long =
        outputSample * format.sampleRate / TARGET_SAMPLE_RATE

    private fun totalWorkSamples(): Long {
        if (chunkCount == 0) return 0L
        return (chunkCount - 1L) * WhisperChunker.PAYLOAD_SAMPLES +
            window(chunkCount - 1).sampleCount
    }

    companion object {
        private const val PCM_FORMAT = 1
        private const val IEEE_FLOAT_FORMAT = 3
        private const val TARGET_SAMPLE_RATE = WhisperFeatureExtractor.SAMPLE_RATE
        private const val OUTPUT_BATCH_SAMPLES = 4_096
        private const val MIN_WEIGHT = 1.0e-12

        fun open(file: File): StreamingWavChunkSource {
            val input = RandomAccessFile(file, "r")
            try {
                val parsed = parseHeader(input, file.name)
                val whisperSampleCountLong =
                    parsed.sourceFrameCount * TARGET_SAMPLE_RATE / parsed.format.sampleRate
                require(whisperSampleCountLong <= Int.MAX_VALUE) {
                    "WAV duration exceeds the supported transcription window range."
                }
                val info = StreamingWavInfo(
                    sourceSampleRate = parsed.format.sampleRate,
                    sourceChannels = parsed.format.channels,
                    sourceFrameCount = parsed.sourceFrameCount,
                    whisperSampleCount = whisperSampleCountLong.toInt(),
                )
                return StreamingWavChunkSource(
                    input = input,
                    format = parsed.format,
                    dataOffset = parsed.dataOffset,
                    info = info,
                )
            } catch (throwable: Throwable) {
                input.close()
                throw throwable
            }
        }

        private fun windowCount(sampleCount: Int): Int {
            if (sampleCount == 0) return 0
            if (sampleCount <= WhisperChunker.PAYLOAD_SAMPLES) return 1
            val remainingAfterFirst = sampleCount - WhisperChunker.PAYLOAD_SAMPLES.toLong()
            return 1 +
                (
                    (remainingAfterFirst + WhisperChunker.STRIDE_SAMPLES - 1L) /
                        WhisperChunker.STRIDE_SAMPLES
                    ).toInt()
        }

        private fun parseHeader(input: RandomAccessFile, fileName: String): ParsedWav {
            require(input.length() >= 44L) { "WAV file is too short: $fileName" }
            require(input.readAscii(4) == "RIFF") { "Not a RIFF file: $fileName" }
            input.readUInt32Le()
            require(input.readAscii(4) == "WAVE") { "Not a WAVE file: $fileName" }

            var format: WavFormat? = null
            var dataOffset = -1L
            var dataSize = -1L
            while (input.filePointer + 8L <= input.length()) {
                val chunkId = input.readAscii(4)
                val chunkSize = input.readUInt32Le()
                val payloadOffset = input.filePointer
                val payloadEnd = payloadOffset + chunkSize
                require(payloadEnd >= payloadOffset && payloadEnd <= input.length()) {
                    "Truncated WAV chunk '$chunkId' in $fileName"
                }
                when (chunkId) {
                    "fmt " -> {
                        require(chunkSize >= 16L) { "Invalid WAV format chunk." }
                        val audioFormat = input.readUInt16Le()
                        val channels = input.readUInt16Le()
                        val sampleRateLong = input.readUInt32Le()
                        require(sampleRateLong <= Int.MAX_VALUE) {
                            "Unsupported WAV sample rate."
                        }
                        input.readUInt32Le()
                        val blockAlign = input.readUInt16Le()
                        val bitsPerSample = input.readUInt16Le()
                        format = WavFormat(
                            audioFormat = audioFormat,
                            channels = channels,
                            sampleRate = sampleRateLong.toInt(),
                            blockAlign = blockAlign,
                            bitsPerSample = bitsPerSample,
                        )
                    }

                    "data" -> {
                        if (dataOffset < 0L) {
                            dataOffset = payloadOffset
                            dataSize = chunkSize
                        }
                    }
                }
                val paddedEnd = payloadEnd + (chunkSize and 1L)
                require(paddedEnd >= payloadEnd && paddedEnd <= input.length()) {
                    "Truncated WAV chunk padding in $fileName"
                }
                input.seek(paddedEnd)
            }

            val wavFormat = requireNotNull(format) { "WAV has no format chunk." }
            require(dataOffset >= 0L) { "WAV has no data chunk." }
            require(wavFormat.channels in 1..8) { "Unsupported WAV channel count." }
            require(wavFormat.sampleRate in 8_000..384_000) {
                "Unsupported WAV sample rate."
            }
            require(
                (wavFormat.audioFormat == PCM_FORMAT && wavFormat.bitsPerSample == 16) ||
                    (
                        wavFormat.audioFormat == IEEE_FLOAT_FORMAT &&
                            wavFormat.bitsPerSample == 32
                        ),
            ) {
                "Only PCM16 and float32 WAV files are supported."
            }
            val bytesPerSample = wavFormat.bitsPerSample / 8
            val expectedBlockAlign = bytesPerSample * wavFormat.channels
            require(wavFormat.blockAlign == expectedBlockAlign) {
                "Invalid WAV block alignment."
            }
            require(dataSize % expectedBlockAlign == 0L) {
                "WAV data ends mid-frame."
            }
            return ParsedWav(
                format = wavFormat,
                dataOffset = dataOffset,
                sourceFrameCount = dataSize / expectedBlockAlign,
            )
        }
    }

    private data class ParsedWav(
        val format: WavFormat,
        val dataOffset: Long,
        val sourceFrameCount: Long,
    )

    private data class WavFormat(
        val audioFormat: Int,
        val channels: Int,
        val sampleRate: Int,
        val blockAlign: Int,
        val bitsPerSample: Int,
    )

    private class SincKernelBank(sourceRate: Int) {
        val radius = maxOf(
            BASE_RADIUS,
            ceil(BASE_RADIUS.toDouble() * sourceRate / REFERENCE_DOWNSAMPLE_RATE).toInt(),
        )
        val tapCount = radius * 2 + 1
        val rateGcd = greatestCommonDivisor(sourceRate, TARGET_SAMPLE_RATE)
        val phaseCount = TARGET_SAMPLE_RATE / rateGcd
        private val sourceStep = sourceRate / rateGcd
        val wholeFrameStep = sourceStep / phaseCount
        val phaseStep = sourceStep % phaseCount
        val weights = FloatArray(phaseCount * tapCount)

        init {
            val rateRatio = TARGET_SAMPLE_RATE.toDouble() / sourceRate
            val cutoff =
                0.5 * minOf(1.0, rateRatio) * ANTI_ALIAS_CUTOFF_RATIO
            for (phase in 0 until phaseCount) {
                val fraction = phase.toDouble() / phaseCount
                var sum = 0.0
                val rowOffset = phase * tapCount
                for (tap in 0 until tapCount) {
                    val sourceOffset = tap - radius
                    val distance = fraction - sourceOffset
                    val normalized = distance / radius
                    val weight = if (kotlin.math.abs(normalized) >= 1.0) {
                        0.0
                    } else {
                        val window =
                            0.42 +
                                0.5 * cos(PI * normalized) +
                                0.08 * cos(2.0 * PI * normalized)
                        val argument = 2.0 * PI * cutoff * distance
                        val sinc = if (kotlin.math.abs(distance) < 1.0e-12) {
                            2.0 * cutoff
                        } else {
                            sin(argument) / (PI * distance)
                        }
                        window * sinc
                    }
                    weights[rowOffset + tap] = weight.toFloat()
                    sum += weight
                }
                require(kotlin.math.abs(sum) >= MIN_WEIGHT) {
                    "Could not construct a WAV resampling kernel."
                }
                for (tap in 0 until tapCount) {
                    weights[rowOffset + tap] =
                        (weights[rowOffset + tap] / sum).toFloat()
                }
            }
        }

        companion object {
            // A 64-tap base provides a narrow transition around the 8 kHz
            // output Nyquist while every fractional phase is precomputed.
            private const val BASE_RADIUS = 32
            private const val REFERENCE_DOWNSAMPLE_RATE = 48_000.0
            private const val ANTI_ALIAS_CUTOFF_RATIO = 0.97

            private fun greatestCommonDivisor(left: Int, right: Int): Int {
                var a = left
                var b = right
                while (b != 0) {
                    val remainder = a % b
                    a = b
                    b = remainder
                }
                return a
            }
        }
    }
}

private fun RandomAccessFile.readAscii(length: Int): String {
    val bytes = ByteArray(length)
    readFully(bytes)
    return bytes.toString(Charsets.US_ASCII)
}

private fun RandomAccessFile.readUInt16Le(): Int {
    val low = read()
    val high = read()
    if (low < 0 || high < 0) throw EOFException()
    return low or (high shl 8)
}

private fun RandomAccessFile.readInt32Le(): Int {
    val b0 = read()
    val b1 = read()
    val b2 = read()
    val b3 = read()
    if (b0 < 0 || b1 < 0 || b2 < 0 || b3 < 0) throw EOFException()
    return b0 or (b1 shl 8) or (b2 shl 16) or (b3 shl 24)
}

private fun RandomAccessFile.readUInt32Le(): Long =
    readInt32Le().toLong() and 0xffff_ffffL
