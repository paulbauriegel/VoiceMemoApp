package com.example.voicememo.transcription

import java.io.ByteArrayOutputStream
import java.io.File
import java.io.RandomAccessFile
import java.security.MessageDigest
import kotlin.math.PI
import kotlin.math.sin
import kotlin.math.sqrt
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class StreamingWavChunkSourceTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test
    fun pcm16Stereo_streamsExistingWindowsAndPreservesOverlap() {
        val frameCount = 30 * 16_000
        val samples = ShortArray(frameCount * 2) { index ->
            val frame = index / 2
            if (index % 2 == 0) {
                (frame % 20_000 - 10_000).toShort()
            } else {
                (10_000 - frame % 20_000).toShort()
            }
        }
        val file = writePcm16Wav("long-stereo.wav", 16_000, 2, samples)
        val originalHash = sha256(file)

        StreamingWavChunkSource.open(file).use { source ->
            assertEquals(2, source.chunkCount)
            assertEquals(frameCount, source.info.whisperSampleCount)

            val first = source.readChunk(0)
            val second = source.readChunk(1)

            assertEquals(0, first.window.startSample)
            assertEquals(28 * 16_000, first.window.endSample)
            assertEquals(26 * 16_000, second.window.startSample)
            assertEquals(30 * 16_000, second.window.endSample)
            assertArrayEquals(
                first.samples.copyOfRange(26 * 16_000, 28 * 16_000),
                second.samples.copyOfRange(0, 2 * 16_000),
                0.0f,
            )
            assertTrue(first.samples.all { it == 0.0f })
        }

        assertEquals(originalHash, sha256(file))
    }

    @Test
    fun float32Stereo_downmixesAndResamplesWithoutLoadingRecording() {
        val frames = 48_000
        val interleaved = FloatArray(frames * 2) { index ->
            if (index % 2 == 0) 0.75f else -0.25f
        }
        val file = writeFloatWav("float.wav", 48_000, 2, interleaved)

        StreamingWavChunkSource.open(file).use { source ->
            val chunk = source.readChunk(0)

            assertEquals(16_000, chunk.samples.size)
            // Ignore the finite-filter boundary; a constant signal downmixes to 0.25.
            chunk.samples.copyOfRange(100, chunk.samples.size - 100).forEach {
                assertEquals(0.25f, it, 1.0e-4f)
            }
        }
    }

    @Test
    fun windowedSinc_preservesSpeechBandAndSuppressesAliasing() {
        val lowTone = writeMonoTone("low.wav", sourceRate = 48_000, frequency = 1_000.0)
        val speechBandEdge =
            writeMonoTone("speech-edge.wav", sourceRate = 48_000, frequency = 7_000.0)
        val nearStopBand =
            writeMonoTone("near-stop.wav", sourceRate = 48_000, frequency = 9_000.0)
        val highTone = writeMonoTone("high.wav", sourceRate = 48_000, frequency = 12_000.0)

        val lowRms = resampledInteriorRms(lowTone)
        val speechEdgeRms = resampledInteriorRms(speechBandEdge)
        val nearStopRms = resampledInteriorRms(nearStopBand)
        val highRms = resampledInteriorRms(highTone)

        assertTrue("Speech-band tone was attenuated: $lowRms", lowRms > 0.60)
        assertTrue("Speech-band edge was attenuated: $speechEdgeRms", speechEdgeRms > 0.55)
        assertTrue("Near-stop-band tone aliased into output: $nearStopRms", nearStopRms < 0.08)
        assertTrue("Out-of-band tone aliased into output: $highRms", highRms < 0.08)
    }

    @Test
    fun windowedSinc_scalesForHighRateImports() {
        val highTone = writeMonoTone(
            name = "high-384k.wav",
            sourceRate = 384_000,
            frequency = 12_000.0,
        )

        val highRms = resampledInteriorRms(highTone)

        assertTrue("High-rate tone aliased into output: $highRms", highRms < 0.08)
    }

    @Test
    fun resampledAdjacentWindowsHaveBitExactOverlap() {
        val frameCount = 30 * 44_100
        val samples = ShortArray(frameCount) { frame ->
            ((frame * 97L % 40_001L) - 20_000L).toShort()
        }
        val file = writePcm16Wav("overlap-44100.wav", 44_100, 1, samples)

        StreamingWavChunkSource.open(file).use { source ->
            val first = source.readChunk(0)
            val second = source.readChunk(1)

            assertArrayEquals(
                first.samples.copyOfRange(26 * 16_000, 28 * 16_000),
                second.samples.copyOfRange(0, 2 * 16_000),
                0.0f,
            )
        }
    }

    @Test
    fun repeatedReadsAreBitDeterministic() {
        val file = writeMonoTone("tone-44100.wav", sourceRate = 44_100, frequency = 997.0)

        StreamingWavChunkSource.open(file).use { source ->
            val first = source.readChunk(0).samples
            val second = source.readChunk(0).samples

            assertArrayEquals(first, second, 0.0f)
        }
    }

    @Test
    fun progressIsMonotonicAcrossOverlappingChunks() {
        val frames = 60 * 16_000
        val file = createSparsePcm16Wav("sixty-seconds.wav", 16_000, frames.toLong())
        val fractions = mutableListOf<Float>()
        var emittedChunks = 0

        StreamingWavChunkSource.open(file).use { source ->
            source.forEachChunk(
                onProgress = { fractions += it.fraction },
                onChunk = { emittedChunks++ },
            )
        }

        assertEquals(3, emittedChunks)
        assertTrue(fractions.isNotEmpty())
        assertTrue(fractions.zipWithNext().all { (left, right) -> right >= left })
        assertEquals(1.0f, fractions.last(), 0.0f)
    }

    @Test
    fun eighteenMinuteMemo_isExposedAsBoundedSeekableBatches() {
        val durationSeconds = 18 * 60 + 32
        val sourceRate = 44_100
        val file = createSparsePcm16Wav(
            name = "eighteen-minutes.wav",
            sampleRate = sourceRate,
            frameCount = durationSeconds.toLong() * sourceRate,
        )
        val originalLength = file.length()

        StreamingWavChunkSource.open(file).use { source ->
            assertEquals(43, source.chunkCount)
            assertEquals(durationSeconds * 16_000, source.info.whisperSampleCount)

            val first = source.readChunk(0)
            val last = source.readChunk(source.chunkCount - 1)

            assertEquals(28 * 16_000, first.samples.size)
            assertEquals(20 * 16_000, last.samples.size)
            assertEquals((18 * 60 + 12) * 16_000, last.window.startSample)
            assertEquals(durationSeconds * 16_000, last.window.endSample)
        }

        assertEquals(originalLength, file.length())
    }

    @Test
    fun cancellationStopsDuringPreparation() {
        val file = createSparsePcm16Wav(
            name = "cancel.wav",
            sampleRate = 48_000,
            frameCount = 60L * 48_000,
        )
        var progressCallbacks = 0

        StreamingWavChunkSource.open(file).use { source ->
            try {
                source.readChunk(
                    index = 0,
                    shouldCancel = { progressCallbacks >= 2 },
                    onProgress = { progressCallbacks++ },
                )
                fail("Expected preparation cancellation.")
            } catch (_: WhisperCancelledException) {
                assertEquals(2, progressCallbacks)
            }
        }
    }

    private fun resampledInteriorRms(file: File): Double =
        StreamingWavChunkSource.open(file).use { source ->
            val samples = source.readChunk(0).samples
            val interior = samples.copyOfRange(200, samples.size - 200)
            sqrt(interior.sumOf { it.toDouble() * it } / interior.size)
        }

    private fun writeMonoTone(name: String, sourceRate: Int, frequency: Double): File {
        val samples = FloatArray(sourceRate) { frame ->
            sin(2.0 * PI * frequency * frame / sourceRate).toFloat()
        }
        return writeFloatWav(name, sourceRate, 1, samples)
    }

    private fun writePcm16Wav(
        name: String,
        sampleRate: Int,
        channels: Int,
        samples: ShortArray,
    ): File {
        val output = ByteArrayOutputStream()
        val dataBytes = samples.size * 2
        output.writeWavHeader(
            audioFormat = 1,
            sampleRate = sampleRate,
            channels = channels,
            bitsPerSample = 16,
            dataBytes = dataBytes,
        )
        samples.forEach { output.writeShortLe(it.toInt()) }
        return temporaryFolder.newFile(name).apply { writeBytes(output.toByteArray()) }
    }

    private fun writeFloatWav(
        name: String,
        sampleRate: Int,
        channels: Int,
        samples: FloatArray,
    ): File {
        val output = ByteArrayOutputStream()
        val dataBytes = samples.size * 4
        output.writeWavHeader(
            audioFormat = 3,
            sampleRate = sampleRate,
            channels = channels,
            bitsPerSample = 32,
            dataBytes = dataBytes,
        )
        samples.forEach { output.writeIntLe(it.toBits()) }
        return temporaryFolder.newFile(name).apply { writeBytes(output.toByteArray()) }
    }

    private fun createSparsePcm16Wav(
        name: String,
        sampleRate: Int,
        frameCount: Long,
    ): File {
        val file = temporaryFolder.newFile(name)
        val dataBytes = frameCount * 2L
        require(dataBytes <= Int.MAX_VALUE)
        val header = ByteArrayOutputStream().apply {
            writeWavHeader(
                audioFormat = 1,
                sampleRate = sampleRate,
                channels = 1,
                bitsPerSample = 16,
                dataBytes = dataBytes.toInt(),
            )
        }.toByteArray()
        RandomAccessFile(file, "rw").use { output ->
            output.write(header)
            output.setLength(header.size + dataBytes)
        }
        return file
    }

    private fun ByteArrayOutputStream.writeWavHeader(
        audioFormat: Int,
        sampleRate: Int,
        channels: Int,
        bitsPerSample: Int,
        dataBytes: Int,
    ) {
        val bytesPerSample = bitsPerSample / 8
        writeAscii("RIFF")
        writeIntLe(36 + dataBytes)
        writeAscii("WAVE")
        writeAscii("fmt ")
        writeIntLe(16)
        writeShortLe(audioFormat)
        writeShortLe(channels)
        writeIntLe(sampleRate)
        writeIntLe(sampleRate * channels * bytesPerSample)
        writeShortLe(channels * bytesPerSample)
        writeShortLe(bitsPerSample)
        writeAscii("data")
        writeIntLe(dataBytes)
    }

    private fun ByteArrayOutputStream.writeAscii(value: String) =
        write(value.toByteArray(Charsets.US_ASCII))

    private fun ByteArrayOutputStream.writeShortLe(value: Int) {
        write(value and 0xff)
        write((value ushr 8) and 0xff)
    }

    private fun ByteArrayOutputStream.writeIntLe(value: Int) {
        write(value and 0xff)
        write((value ushr 8) and 0xff)
        write((value ushr 16) and 0xff)
        write((value ushr 24) and 0xff)
    }

    private fun sha256(file: File): String =
        MessageDigest.getInstance("SHA-256")
            .digest(file.readBytes())
            .joinToString("") { "%02x".format(it) }
}
