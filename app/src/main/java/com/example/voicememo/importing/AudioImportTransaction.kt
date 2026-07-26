package com.example.voicememo.importing

import com.example.voicememo.data.MemoRecord
import com.example.voicememo.data.NewMemo
import com.example.voicememo.data.WaveformState
import com.example.voicememo.recording.WavFileWriter
import com.example.voicememo.recording.WaveformAccumulator
import java.io.File
import java.util.UUID
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext

internal fun interface DecodedPcmSink {
    /**
     * Receives mono PCM16 samples. The sample rate must remain stable for one
     * import and [samples] is consumed before this method returns.
     */
    fun write(sampleRate: Int, samples: ShortArray, count: Int)
}

internal class AudioImportTransaction(
    private val memoRoot: File,
    private val createMemo: suspend (NewMemo) -> MemoRecord,
    private val storageIdFactory: () -> String = { UUID.randomUUID().toString() },
) {
    suspend fun import(
        title: String,
        createdAtMillis: Long,
        decode: suspend (DecodedPcmSink) -> Unit,
    ): MemoRecord {
        require(title.isNotBlank()) { "Imported memo title must not be blank." }
        val storageId = storageIdFactory()
        require(storageId.matches(SAFE_STORAGE_ID)) { "Invalid import storage identifier." }

        check(memoRoot.mkdirs() || memoRoot.isDirectory) {
            "The private memo directory could not be created."
        }
        val canonicalRoot = memoRoot.canonicalFile
        val stagingDirectory = File(canonicalRoot, ".importing-$storageId")
        val publishedDirectory = File(canonicalRoot, storageId)
        require(stagingDirectory.canonicalFile.parentFile == canonicalRoot)
        require(publishedDirectory.canonicalFile.parentFile == canonicalRoot)
        check(!stagingDirectory.exists() && !publishedDirectory.exists()) {
            "An import with this storage identifier already exists."
        }
        check(stagingDirectory.mkdir()) { "The import staging directory could not be created." }

        val stagedAudio = File(stagingDirectory, AUDIO_FILE_NAME)
        val stagedWaveform = File(stagingDirectory, WAVEFORM_FILE_NAME)
        var writer: WavFileWriter? = null
        var waveform: WaveformAccumulator? = null
        var sampleRate: Int? = null
        var published = false
        var committed = false

        try {
            val sink = DecodedPcmSink { rate, samples, count ->
                require(rate in MIN_SAMPLE_RATE..MAX_SAMPLE_RATE) {
                    "Unsupported decoded sample rate: $rate Hz."
                }
                require(count in 0..samples.size)
                val establishedRate = sampleRate
                if (establishedRate == null) {
                    sampleRate = rate
                    writer = WavFileWriter(stagedAudio, rate)
                    waveform = WaveformAccumulator(rate)
                } else {
                    require(establishedRate == rate) {
                        "The decoded audio sample rate changed during import."
                    }
                }
                if (count > 0) {
                    val projectedFrames = requireNotNull(writer).frameCount + count
                    require(projectedFrames <= MAX_WAV_PCM_FRAMES) {
                        "The decoded audio is too large for a standard WAV file."
                    }
                    writer?.write(samples, count)
                    waveform?.add(samples, count)
                }
            }

            decode(sink)
            currentCoroutineContext().ensureActive()

            val completedWriter = requireNotNull(writer) {
                "The selected file contains no decodable audio."
            }
            val durationMillis = completedWriter.durationMillis
            require(durationMillis > 0L) {
                "The selected file contains no decodable audio."
            }
            completedWriter.finish()
            writer = null
            requireNotNull(waveform).finishTo(stagedWaveform)
            waveform = null

            currentCoroutineContext().ensureActive()

            // Publishing the completed files and inserting their Room row form
            // one non-cancellable commit boundary. Failure before the database
            // insert removes the published directory again.
            val memo = withContext(NonCancellable) {
                check(stagingDirectory.renameTo(publishedDirectory)) {
                    "The imported audio could not be moved into the memo library."
                }
                published = true
                val publishedAudio = File(publishedDirectory, AUDIO_FILE_NAME)
                val publishedWaveform = File(publishedDirectory, WAVEFORM_FILE_NAME)
                try {
                    createMemo(
                        NewMemo(
                            title = title,
                            createdAtMillis = createdAtMillis,
                            durationMillis = durationMillis,
                            audioPath = publishedAudio.absolutePath,
                            waveformPath = publishedWaveform.absolutePath,
                            waveformState = WaveformState.READY,
                        ),
                    ).also { committed = true }
                } catch (failure: Throwable) {
                    cleanupDirectory(publishedDirectory, failure)
                    published = false
                    throw failure
                }
            }
            return memo
        } catch (failure: Throwable) {
            runCatching { writer?.finish() }
                .exceptionOrNull()
                ?.let(failure::addSuppressed)
            if (!committed) {
                cleanupDirectory(stagingDirectory, failure)
                if (published) cleanupDirectory(publishedDirectory, failure)
            }
            throw failure
        }
    }

    private fun cleanupDirectory(directory: File, originalFailure: Throwable) {
        if (!directory.exists()) return
        runCatching {
            check(directory.deleteRecursively()) {
                "Could not clean up incomplete import ${directory.name}."
            }
        }.exceptionOrNull()?.let(originalFailure::addSuppressed)
    }

    private companion object {
        val SAFE_STORAGE_ID = Regex("[A-Za-z0-9_-]+")
        const val AUDIO_FILE_NAME = "audio.wav"
        const val WAVEFORM_FILE_NAME = "waveform.vwf"
        const val MIN_SAMPLE_RATE = 8_000
        const val MAX_SAMPLE_RATE = 384_000
        // RIFF stores chunk sizes as unsigned 32-bit integers. Keep enough
        // space for the 36 bytes preceding the PCM payload.
        const val MAX_WAV_PCM_FRAMES = (4_294_967_295L - 36L) / 2L
    }
}
