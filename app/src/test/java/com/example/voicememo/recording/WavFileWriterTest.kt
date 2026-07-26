package com.example.voicememo.recording

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.io.path.createTempFile
import org.junit.Assert.assertEquals
import org.junit.Test

class WavFileWriterTest {
    @Test
    fun writesSeekablePcm16HeaderAndDuration() {
        val file = createTempFile(suffix = ".wav").toFile()
        val samples = shortArrayOf(Short.MIN_VALUE, -1, 0, 1, Short.MAX_VALUE)

        WavFileWriter(file, sampleRate = 1_000).use { writer ->
            writer.write(samples)
            assertEquals(5L, writer.durationMillis)
        }

        val bytes = file.readBytes()
        val header = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        assertEquals("RIFF", bytes.copyOfRange(0, 4).decodeToString())
        assertEquals("WAVE", bytes.copyOfRange(8, 12).decodeToString())
        assertEquals(1_000, header.getInt(24))
        assertEquals(1, header.getShort(22).toInt())
        assertEquals(16, header.getShort(34).toInt())
        assertEquals(samples.size * 2, header.getInt(40))
        assertEquals(44 + samples.size * 2, bytes.size)
    }
}
