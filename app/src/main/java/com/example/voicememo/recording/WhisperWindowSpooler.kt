package com.example.voicememo.recording

import com.example.voicememo.transcription.WhisperChunker
import com.example.voicememo.work.WhisperPreTranscriptionStorage
import java.io.Closeable
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/**
 * Copies only completed Whisper windows out of the microphone stream.
 *
 * Disk writes happen on a dedicated thread so model preparation can never
 * delay [AudioRecord.read] or the primary recording WAV.
 */
internal class WhisperWindowSpooler(
    memoDirectory: File,
    private val sampleRate: Int,
) : Closeable {
    private val directory = WhisperPreTranscriptionStorage.directory(memoDirectory).apply {
        mkdirs()
    }
    private val payloadFrames = WhisperChunker.PAYLOAD_SECONDS * sampleRate
    private val strideFrames =
        (WhisperChunker.PAYLOAD_SECONDS - WhisperChunker.OVERLAP_SECONDS) * sampleRate
    private val ring = ShortArray(payloadFrames)
    private val writer = Executors.newSingleThreadExecutor { task ->
        Thread(task, "whisper-window-spooler").apply { priority = Thread.MIN_PRIORITY }
    }
    private var ringWriteIndex = 0
    private var totalFrames = 0L
    private var nextWindowEnd = payloadFrames.toLong()
    private var nextWindowIndex = 0
    private var closed = false

    fun add(samples: ShortArray, count: Int) {
        check(!closed) { "Whisper window spooler is closed." }
        require(count in 0..samples.size)
        var sourceOffset = 0
        while (sourceOffset < count) {
            val untilBoundary = (nextWindowEnd - totalFrames).toInt()
            val batch = minOf(count - sourceOffset, untilBoundary)
            copyIntoRing(samples, sourceOffset, batch)
            sourceOffset += batch
            totalFrames += batch
            if (totalFrames == nextWindowEnd) {
                enqueueWindow(nextWindowIndex++, snapshot())
                nextWindowEnd += strideFrames
            }
        }
    }

    override fun close() {
        if (closed) return
        closed = true
        writer.execute {
            WhisperPreTranscriptionStorage.finishedMarker(directory).apply {
                parentFile?.mkdirs()
                createNewFile()
            }
        }
        writer.shutdown()
        check(writer.awaitTermination(30, TimeUnit.SECONDS)) {
            "Timed out finalizing Whisper recording windows."
        }
    }

    private fun copyIntoRing(samples: ShortArray, sourceOffset: Int, count: Int) {
        var copied = 0
        while (copied < count) {
            val batch = minOf(count - copied, ring.size - ringWriteIndex)
            samples.copyInto(
                destination = ring,
                destinationOffset = ringWriteIndex,
                startIndex = sourceOffset + copied,
                endIndex = sourceOffset + copied + batch,
            )
            ringWriteIndex = (ringWriteIndex + batch) % ring.size
            copied += batch
        }
    }

    private fun snapshot(): ShortArray {
        val result = ShortArray(ring.size)
        val tail = ring.size - ringWriteIndex
        ring.copyInto(result, 0, ringWriteIndex, ring.size)
        ring.copyInto(result, tail, 0, ringWriteIndex)
        return result
    }

    private fun enqueueWindow(index: Int, samples: ShortArray) {
        writer.execute {
            val destination = WhisperPreTranscriptionStorage.windowFile(directory, index)
            val temporary = File(directory, "${destination.name}.tmp")
            WavFileWriter(temporary, sampleRate).use { wav ->
                wav.write(samples)
            }
            check(temporary.renameTo(destination)) {
                "Could not publish Whisper window ${destination.name}."
            }
        }
    }
}
