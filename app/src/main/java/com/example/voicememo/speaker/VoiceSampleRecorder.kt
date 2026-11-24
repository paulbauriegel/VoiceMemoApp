package com.example.voicememo.speaker

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import androidx.core.content.ContextCompat
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Records a short voice sample (16 kHz mono PCM WAV) into app-private storage.
 * Uses the same microphone source as normal recordings so the voice sounds the
 * same to the voice model.
 */
internal class VoiceSampleRecorder(private val context: Context) {
    class Recording(val durationMillis: Long, val inputChannel: String?)

    fun hasPermission(): Boolean =
        ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED

    /**
     * Records into [target] until [shouldStop] or [MAX_MILLIS]. Returns null
     * when the microphone could not be opened; [target] is then removed.
     */
    @SuppressLint("MissingPermission")
    suspend fun record(
        target: File,
        shouldStop: () -> Boolean,
        onElapsed: (Long) -> Unit,
    ): Recording? = withContext(Dispatchers.IO) {
        if (!hasPermission()) return@withContext null
        val minimum = AudioRecord.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        if (minimum <= 0) return@withContext null
        val recorder = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            maxOf(minimum, SAMPLE_RATE),
        )
        if (recorder.state != AudioRecord.STATE_INITIALIZED) {
            recorder.release()
            return@withContext null
        }
        var written = 0L
        var inputChannel: String? = null
        try {
            RandomAccessFile(target, "rw").use { output ->
                output.setLength(0)
                output.write(ByteArray(HEADER_BYTES))
                val buffer = ShortArray(SAMPLE_RATE / 10)
                val bytes = ByteBuffer.allocate(buffer.size * 2).order(ByteOrder.LITTLE_ENDIAN)
                recorder.startRecording()
                inputChannel = recorder.routedDevice?.type?.let { "device-type-$it" }
                while (!shouldStop() && written < MAX_MILLIS * SAMPLE_RATE / 1_000) {
                    val count = recorder.read(buffer, 0, buffer.size)
                    if (count <= 0) break
                    bytes.clear()
                    bytes.asShortBuffer().put(buffer, 0, count)
                    output.write(bytes.array(), 0, count * 2)
                    written += count
                    onElapsed(written * 1_000 / SAMPLE_RATE)
                }
                output.seek(0)
                output.write(header(written.toInt() * 2))
            }
        } finally {
            runCatching { recorder.stop() }
            recorder.release()
        }
        if (written == 0L) {
            target.delete()
            null
        } else {
            Recording(written * 1_000 / SAMPLE_RATE, inputChannel)
        }
    }

    private fun header(dataBytes: Int): ByteArray =
        ByteBuffer.allocate(HEADER_BYTES).order(ByteOrder.LITTLE_ENDIAN).apply {
            put("RIFF".toByteArray()); putInt(36 + dataBytes); put("WAVE".toByteArray())
            put("fmt ".toByteArray()); putInt(16); putShort(1); putShort(1)
            putInt(SAMPLE_RATE); putInt(SAMPLE_RATE * 2); putShort(2); putShort(16)
            put("data".toByteArray()); putInt(dataBytes)
        }.array()

    companion object {
        const val MAX_MILLIS = 30_000L
        private const val SAMPLE_RATE = 16_000
        private const val HEADER_BYTES = 44
    }
}
