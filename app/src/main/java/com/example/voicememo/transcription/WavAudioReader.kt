package com.example.voicememo.transcription

import java.io.EOFException
import java.io.File
import java.io.RandomAccessFile
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.sin

data class PcmAudio(
    val samples: FloatArray,
    val sampleRate: Int,
)

object WavAudioReader {
    private const val PCM_FORMAT = 1
    private const val IEEE_FLOAT_FORMAT = 3

    fun read(file: File): PcmAudio = RandomAccessFile(file, "r").use { input ->
        require(input.length() >= 44L) { "WAV file is too short: ${file.name}" }
        require(input.readAscii(4) == "RIFF") { "Not a RIFF file: ${file.name}" }
        input.readUInt32Le()
        require(input.readAscii(4) == "WAVE") { "Not a WAVE file: ${file.name}" }

        var format: WavFormat? = null
        var dataOffset = -1L
        var dataSize = -1L

        while (input.filePointer + 8L <= input.length()) {
            val chunkId = input.readAscii(4)
            val chunkSize = input.readUInt32Le()
            val payloadOffset = input.filePointer
            val payloadEnd = payloadOffset + chunkSize
            require(payloadEnd <= input.length()) {
                "Truncated WAV chunk '$chunkId' in ${file.name}"
            }
            when (chunkId) {
                "fmt " -> {
                    require(chunkSize >= 16L) { "Invalid WAV format chunk." }
                    val audioFormat = input.readUInt16Le()
                    val channels = input.readUInt16Le()
                    val sampleRate = input.readUInt32Le().toInt()
                    input.readUInt32Le() // byte rate
                    val blockAlign = input.readUInt16Le()
                    val bitsPerSample = input.readUInt16Le()
                    format = WavFormat(
                        audioFormat = audioFormat,
                        channels = channels,
                        sampleRate = sampleRate,
                        blockAlign = blockAlign,
                        bitsPerSample = bitsPerSample,
                    )
                }

                "data" -> {
                    dataOffset = payloadOffset
                    dataSize = chunkSize
                }
            }
            input.seek(payloadEnd + (chunkSize and 1L))
        }

        val wavFormat = requireNotNull(format) { "WAV has no format chunk." }
        require(dataOffset >= 0L) { "WAV has no data chunk." }
        require(wavFormat.channels in 1..8) { "Unsupported WAV channel count." }
        require(wavFormat.sampleRate in 8_000..384_000) { "Unsupported WAV sample rate." }
        require(
            (wavFormat.audioFormat == PCM_FORMAT && wavFormat.bitsPerSample == 16) ||
                (wavFormat.audioFormat == IEEE_FLOAT_FORMAT && wavFormat.bitsPerSample == 32),
        ) {
            "Only PCM16 and float32 WAV files are supported."
        }

        val bytesPerSample = wavFormat.bitsPerSample / 8
        val expectedBlockAlign = bytesPerSample * wavFormat.channels
        require(wavFormat.blockAlign == expectedBlockAlign) { "Invalid WAV block alignment." }
        require(dataSize % expectedBlockAlign == 0L) { "WAV data ends mid-frame." }

        val frameCountLong = dataSize / expectedBlockAlign
        require(frameCountLong <= Int.MAX_VALUE) { "WAV is too large to decode in one pass." }
        val frameCount = frameCountLong.toInt()
        val mono = FloatArray(frameCount)
        input.seek(dataOffset)
        for (frame in 0 until frameCount) {
            var sum = 0.0f
            repeat(wavFormat.channels) {
                sum += when (wavFormat.audioFormat) {
                    PCM_FORMAT -> input.readInt16Le() / 32768.0f
                    IEEE_FLOAT_FORMAT -> Float.fromBits(input.readInt32Le())
                    else -> error("Validated above")
                }
            }
            mono[frame] = (sum / wavFormat.channels).coerceIn(-1.0f, 1.0f)
        }
        PcmAudio(mono, wavFormat.sampleRate)
    }

    private data class WavFormat(
        val audioFormat: Int,
        val channels: Int,
        val sampleRate: Int,
        val blockAlign: Int,
        val bitsPerSample: Int,
    )

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

    private fun RandomAccessFile.readInt16Le(): Short = readUInt16Le().toShort()

    private fun RandomAccessFile.readInt32Le(): Int {
        val b0 = read()
        val b1 = read()
        val b2 = read()
        val b3 = read()
        if (b0 < 0 || b1 < 0 || b2 < 0 || b3 < 0) throw EOFException()
        return b0 or (b1 shl 8) or (b2 shl 16) or (b3 shl 24)
    }

    private fun RandomAccessFile.readUInt32Le(): Long = readInt32Le().toLong() and 0xffff_ffffL
}

/**
 * Deterministic windowed-sinc resampling. Recording normally supplies 16 kHz
 * directly; this path protects inference quality when a device only accepts
 * 48 kHz (or another PCM rate).
 */
object PcmResampler {
    const val WHISPER_SAMPLE_RATE = 16_000
    private const val HALF_TAPS = 24

    fun toWhisperRate(audio: PcmAudio): FloatArray {
        if (audio.sampleRate == WHISPER_SAMPLE_RATE) return audio.samples.copyOf()
        require(audio.sampleRate > 0)
        if (audio.samples.isEmpty()) return FloatArray(0)

        val outputLength =
            floor(audio.samples.size.toDouble() * WHISPER_SAMPLE_RATE / audio.sampleRate).toInt()
        val result = FloatArray(outputLength)
        val rateRatio = WHISPER_SAMPLE_RATE.toDouble() / audio.sampleRate
        val cutoff = 0.5 * minOf(1.0, rateRatio) * 0.94

        for (outputIndex in result.indices) {
            val sourcePosition = outputIndex.toDouble() * audio.sampleRate / WHISPER_SAMPLE_RATE
            val center = floor(sourcePosition).toInt()
            var weighted = 0.0
            var weightSum = 0.0
            for (sourceIndex in center - HALF_TAPS + 1..center + HALF_TAPS) {
                if (sourceIndex !in audio.samples.indices) continue
                val distance = sourcePosition - sourceIndex
                val normalized = distance / HALF_TAPS
                if (kotlin.math.abs(normalized) >= 1.0) continue
                val window = 0.5 + 0.5 * cos(PI * normalized)
                val argument = 2.0 * PI * cutoff * distance
                val sinc = if (kotlin.math.abs(argument) < 1e-12) {
                    2.0 * cutoff
                } else {
                    sin(argument) / (PI * distance)
                }
                val weight = window * sinc
                weighted += audio.samples[sourceIndex] * weight
                weightSum += weight
            }
            result[outputIndex] =
                if (kotlin.math.abs(weightSum) < 1e-12) 0.0f
                else (weighted / weightSum).toFloat().coerceIn(-1.0f, 1.0f)
        }
        return result
    }
}
