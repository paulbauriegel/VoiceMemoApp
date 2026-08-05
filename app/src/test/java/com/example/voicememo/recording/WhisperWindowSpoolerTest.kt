package com.example.voicememo.recording

import com.example.voicememo.work.WhisperPreTranscriptionStorage
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class WhisperWindowSpoolerTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test
    fun emitsExactOverlappingWindowsAndFinishedMarker() {
        val memoDirectory = temporaryFolder.newFolder("memo")
        val samples = ShortArray(54 * SAMPLE_RATE) { it.toShort() }

        WhisperWindowSpooler(memoDirectory, SAMPLE_RATE).use { spooler ->
            spooler.add(samples, 173)
            spooler.add(samples.copyOfRange(173, samples.size), samples.size - 173)
        }

        val directory = WhisperPreTranscriptionStorage.directory(memoDirectory)
        assertArrayEquals(
            samples.copyOfRange(0, 28 * SAMPLE_RATE),
            readPcm16(WhisperPreTranscriptionStorage.windowFile(directory, 0)),
        )
        assertArrayEquals(
            samples.copyOfRange(26 * SAMPLE_RATE, 54 * SAMPLE_RATE),
            readPcm16(WhisperPreTranscriptionStorage.windowFile(directory, 1)),
        )
        assertTrue(WhisperPreTranscriptionStorage.finishedMarker(directory).isFile)
    }

    private fun readPcm16(file: File): ShortArray {
        RandomAccessFile(file, "r").use { input ->
            input.seek(44)
            val bytes = ByteArray((input.length() - 44).toInt())
            input.readFully(bytes)
            val shorts = ByteBuffer.wrap(bytes)
                .order(ByteOrder.LITTLE_ENDIAN)
                .asShortBuffer()
            return ShortArray(shorts.remaining()).also(shorts::get)
        }
    }

    private companion object {
        const val SAMPLE_RATE = 10
    }
}
