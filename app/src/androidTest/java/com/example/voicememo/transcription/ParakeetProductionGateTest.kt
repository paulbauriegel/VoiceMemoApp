package com.example.voicememo.transcription

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Five-sample gate for the shipped Parakeet path: the same window source, engine and word
 * merge that [com.example.voicememo.work.TranscriptionWorker] uses, on staged WAVs in
 * files/gpu-audio. Writes cache/parakeet-gate/<run_id>.json; the host scores the text.
 */
@RunWith(AndroidJUnit4::class)
class ParakeetProductionGateTest {
    @Test
    fun transcribeStagedAudioLikeTheApp() {
        val arguments = InstrumentationRegistry.getArguments()
        val audioNames = arguments.getString(ARG_AUDIO_NAMES)
        assumeTrue("Parakeet gate audio not requested.", audioNames != null)
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val backend = if (arguments.getString(ARG_BACKEND) == "cpu") {
            WhisperBackendRequest.CPU
        } else {
            WhisperBackendRequest.GPU
        }
        val runId = arguments.getString(ARG_RUN_ID) ?: System.currentTimeMillis().toString()
        val audioDirectory = File(context.filesDir, AUDIO_DIRECTORY).canonicalFile

        val before = ParakeetBenchmarkSupport.memory()
        val loadStarted = System.nanoTime()
        val engine = LiteRtParakeetEngine.open(context, backend)
        val loadMillis = (System.nanoTime() - loadStarted) / 1_000_000
        val report = JSONObject()
            .put("run_id", runId)
            .put("backend", if (engine.usesGpu) "gpu" else "cpu")
            .put("fingerprint", ParakeetInferenceFingerprint.forConfiguration(backend))
            .put("model_load_ms", loadMillis)
            .put("memory_before", before)
            .put("memory_after_load", ParakeetBenchmarkSupport.memory())
        val samples = JSONArray()
        try {
            for (name in requireNotNull(audioNames).split(',').filter { it.isNotBlank() }) {
                val file = File(audioDirectory, name).canonicalFile
                require(file.parentFile == audioDirectory && file.isFile) { "Missing staged audio $name." }
                val started = System.nanoTime()
                val transcript = StreamingWavChunkSource.open(
                    file = file,
                    windowConfig = TranscriptionModel.PARAKEET_TDT_0_6B_V3.windowConfig(),
                ).use { source ->
                    LongFormTranscriber(
                        engine = engine,
                        lockAutoDetectedLanguage = false,
                        mergeTimedWords = true,
                    ).transcribeBatches(
                        windows = (0 until source.chunkCount).map(source::window),
                        loadAudio = { window -> source.readChunk(window.index).samples },
                        language = WhisperLanguage.Auto,
                    )
                }
                val wallMillis = (System.nanoTime() - started) / 1_000_000
                val audioSeconds = transcript.chunks.last().window.endSample.toDouble() /
                    ParakeetAssets.SAMPLE_RATE
                val words = ParakeetTranscriptMerger.merge(transcript.chunks.map { it.timedWordWindow() })
                samples.put(
                    JSONObject()
                        .put("audio", name)
                        .put("audio_seconds", audioSeconds)
                        .put("wall_ms", wallMillis)
                        .put("rtf", wallMillis / 1000.0 / audioSeconds)
                        .put("windows", transcript.chunks.size)
                        .put("text", transcript.text)
                        .put("words", JSONArray(TimedWordsCodec.encode(words)))
                        .put("memory", ParakeetBenchmarkSupport.memory()),
                )
            }
        } finally {
            engine.close()
        }
        report.put("samples", samples).put("memory_after", ParakeetBenchmarkSupport.memory())
        val directory = File(context.cacheDir, REPORT_DIRECTORY).apply { mkdirs() }
        File(directory, "$runId.json").writeText(report.toString(1))
    }

    private companion object {
        const val ARG_AUDIO_NAMES = "parakeet_audio_names"
        const val ARG_BACKEND = "parakeet_backend"
        const val ARG_RUN_ID = "run_id"
        const val AUDIO_DIRECTORY = "gpu-audio"
        const val REPORT_DIRECTORY = "parakeet-gate"
    }
}
