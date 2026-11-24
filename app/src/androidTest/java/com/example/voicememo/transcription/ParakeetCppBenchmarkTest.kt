package com.example.voicememo.transcription

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.voicememo.transcription.ParakeetBenchmarkSupport.FRAME_SECONDS
import com.example.voicememo.transcription.ParakeetBenchmarkSupport.SAMPLE_RATE
import com.example.voicememo.transcription.ParakeetBenchmarkSupport.TimedPiece
import java.io.File
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.math.max

/**
 * Diagnostic benchmark: Parakeet TDT 0.6B v3 through whisper.cpp's ggml backend,
 * 30 s windows at a 25 s stride, with optional gap repair (see [ParakeetBenchmarkSupport]).
 *
 * The device only decodes. Raw timed pieces per window go to
 * cache/parakeet-cpp/<run_id>.json; the host merges and scores them.
 */
@RunWith(AndroidJUnit4::class)
class ParakeetCppBenchmarkTest {
    @Test
    fun transcribeStagedAudio() {
        val arguments = InstrumentationRegistry.getArguments()
        val modelName = arguments.getString(ARG_MODEL_NAME)
        assumeTrue("Parakeet ggml model not requested.", modelName != null)
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val audioNames = requireNotNull(arguments.getString(ARG_AUDIO_NAMES)) {
            "Pass -e $ARG_AUDIO_NAMES name1.wav,name2.wav"
        }.split(',').filter { it.isNotBlank() }
        val useGpu = arguments.getString(ARG_USE_GPU).toBoolean()
        val threads = arguments.getString(ARG_THREADS)?.toInt() ?: DEFAULT_THREADS
        val gapRepair = arguments.getString(ARG_GAP_REPAIR)?.toBoolean() ?: true
        val repairMode = arguments.getString(ARG_REPAIR_MODE) ?: "tight"
        require(repairMode == "tight" || repairMode == "full") { "Repair mode must be tight or full." }
        val runId = arguments.getString(ARG_RUN_ID) ?: System.currentTimeMillis().toString()

        val modelFile = ParakeetBenchmarkSupport.stagedFile(context, MODEL_DIRECTORY, requireNotNull(modelName), ".bin")
        ParakeetCppBenchmarkNative.load()
        val before = ParakeetBenchmarkSupport.memory()
        val loadStarted = System.nanoTime()
        val handle = ParakeetCppBenchmarkNative.create(modelFile.absolutePath, threads, useGpu)
        val loadMillis = (System.nanoTime() - loadStarted) / 1_000_000
        val report = JSONObject()
            .put("run_id", runId)
            .put("engine", "ggml")
            .put("model", modelFile.name)
            .put("model_bytes", modelFile.length())
            .put("backend", ParakeetCppBenchmarkNative.getBackend(handle))
            .put("threads", threads)
            .put("gap_repair", gapRepair)
            .put("repair_mode", repairMode)
            .put("window_seconds", ParakeetBenchmarkSupport.WINDOW_SECONDS)
            .put("stride_seconds", ParakeetBenchmarkSupport.STRIDE_SECONDS)
            .put("model_load_ms", loadMillis)
            .put("memory_before", before)
        val samples = JSONArray()
        var peakPssKib = 0L
        val decoder = ParakeetBenchmarkSupport.WindowDecoder { audio, start, end, stats ->
            ParakeetCppBenchmarkNative.transcribe(handle, audio.slice(start, end))
            stats.decodes++
            stats.decodeMillis += ParakeetCppBenchmarkNative.getInferenceMillis(handle)
            val triplets = ParakeetCppBenchmarkNative.getTokenTriplets(handle)
            val pieces = ParakeetCppBenchmarkNative.getPieces(handle)
            val offset = start.toDouble() / SAMPLE_RATE
            pieces.indices.map { index ->
                val frame = triplets[index * 3 + 1]
                val duration = triplets[index * 3 + 2]
                TimedPiece(
                    tokenId = triplets[index * 3],
                    text = pieces[index],
                    startSeconds = offset + frame * FRAME_SECONDS,
                    endSeconds = offset + (frame + duration) * FRAME_SECONDS,
                )
            }
        }
        try {
            for (name in audioNames) {
                val audio = ParakeetBenchmarkSupport.readPcm16Wav(
                    ParakeetBenchmarkSupport.stagedFile(context, AUDIO_DIRECTORY, name, ".wav"),
                )
                samples.put(
                    ParakeetBenchmarkSupport.transcribeSample(name, audio, decoder, gapRepair, repairMode == "tight") {
                        peakPssKib = max(peakPssKib, ParakeetBenchmarkSupport.memory().getLong("total_pss_kib"))
                    },
                )
            }
        } finally {
            ParakeetCppBenchmarkNative.destroy(handle)
        }
        report.put("samples", samples).put("peak_total_pss_kib", peakPssKib).put("memory_after", ParakeetBenchmarkSupport.memory())
        val directory = File(context.cacheDir, REPORT_DIRECTORY).apply { mkdirs() }
        File(directory, "parakeet-$runId.json").writeText(report.toString(1))
    }

    private companion object {
        const val ARG_MODEL_NAME = "parakeet_model_name"
        const val ARG_AUDIO_NAMES = "parakeet_audio_names"
        const val ARG_USE_GPU = "parakeet_use_gpu"
        const val ARG_THREADS = "parakeet_threads"
        const val ARG_GAP_REPAIR = "parakeet_gap_repair"
        const val ARG_REPAIR_MODE = "parakeet_repair_mode"
        const val ARG_RUN_ID = "run_id"
        const val MODEL_DIRECTORY = "whispercpp-models"
        const val AUDIO_DIRECTORY = "gpu-audio"
        const val REPORT_DIRECTORY = "parakeet-cpp"
        const val DEFAULT_THREADS = 4
    }
}

internal object ParakeetCppBenchmarkNative {
    @Volatile
    private var loaded = false

    @Synchronized
    fun load() {
        if (loaded) return
        System.loadLibrary("voicememo_whispercpp_benchmark")
        loaded = true
    }

    @JvmStatic external fun create(modelPath: String, threads: Int, useGpu: Boolean): Long
    @JvmStatic external fun transcribe(handle: Long, audio: FloatArray)
    @JvmStatic external fun getTokenTriplets(handle: Long): IntArray
    @JvmStatic external fun getPieces(handle: Long): Array<String>
    @JvmStatic external fun getInferenceMillis(handle: Long): Long
    @JvmStatic external fun getBackend(handle: Long): String
    @JvmStatic external fun destroy(handle: Long)
}
