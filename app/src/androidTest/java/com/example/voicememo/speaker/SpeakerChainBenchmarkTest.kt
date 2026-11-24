package com.example.voicememo.speaker

import android.os.Build
import android.os.PowerManager
import android.os.Process
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.voicememo.transcription.ParakeetBenchmarkSupport
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Isolated speed and parity benchmark for the production speaker-label engines:
 * [PyannoteSegmenter] and [CamPlusPlusEmbedder] (with the Kotlin [KaldiFbank]),
 * loaded from the bundled assets exactly as the app will load them.
 *
 * The host runner (`scripts/run-speaker-chain-benchmark.py`) must build the
 * target with `-PisolatedBenchmarkPackage=true`. Audio and reports live only
 * under that separate package's private storage.
 */
@RunWith(AndroidJUnit4::class)
class SpeakerChainBenchmarkTest {
    @Test
    fun runBenchmark() {
        val arguments = InstrumentationRegistry.getArguments()
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        check(context.packageName == BENCHMARK_PACKAGE) {
            "Refusing to run against non-isolated package ${context.packageName}."
        }
        val runId = requireSafe(arguments.getString("runId"), "runId")
        val threads = requireSafe(arguments.getString("threads"), "threads").toInt()
        require(threads in 1..8) { "Invalid threads." }
        val audioFile = ParakeetBenchmarkSupport.stagedFile(
            context, DIRECTORY, arguments.getString("audioName").orEmpty(), ".wav",
        )
        // Optional staged alternative voice model; the bundled one otherwise.
        val voiceModel = arguments.getString("voiceModelName")?.let { name ->
            ParakeetBenchmarkSupport.stagedFile(context, DIRECTORY, name, ".onnx")
        }
        val reportDirectory = File(context.cacheDir, DIRECTORY).apply { mkdirs() }
        val reportFile = File(reportDirectory, "$runId.json")
        val temporaryReport = File(reportDirectory, "$runId.json.tmp")
        val segmentationFile = File(reportDirectory, "$runId.seg.f32")
        val embeddingFile = File(reportDirectory, "$runId.emb.f32")
        listOf(reportFile, temporaryReport, segmentationFile, embeddingFile).forEach(File::delete)
        val report = JSONObject()
            .put("schema_version", 1)
            .put("run_id", runId)
            .put("package_name", context.packageName)
            .put("threads", threads)
            .put("audio_file_name", audioFile.name)
            .put("segmentation_model_sha256", PyannoteSegmenter.MODEL_SHA256)
            .put("voice_model", voiceModel?.name ?: CamPlusPlusEmbedder.MODEL_ID)
            .put("device", JSONObject()
                .put("manufacturer", Build.MANUFACTURER)
                .put("model", Build.MODEL)
                .put("android_release", Build.VERSION.RELEASE))
            .put("status", "running")

        fun publish() {
            temporaryReport.writeText(report.toString(1))
            reportFile.delete()
            check(temporaryReport.renameTo(reportFile)) { "Could not publish the report." }
        }
        publish()
        try {
            report.put("result", benchmark(audioFile, voiceModel, threads, segmentationFile, embeddingFile))
            report.put("status", "completed")
        } catch (error: Throwable) {
            report.put("status", "failed")
                .put("error_type", error.javaClass.name)
                .put("error_message", error.message?.lineSequence()?.firstOrNull()?.take(500))
            throw error
        } finally {
            publish()
        }
    }

    private fun benchmark(
        audioFile: File,
        voiceModel: File?,
        threads: Int,
        segmentationFile: File,
        embeddingFile: File,
    ): JSONObject {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val audio = ParakeetBenchmarkSupport.readPcm16Wav(audioFile).let { it.slice(0, it.size) }
        val thermalBefore = thermalStatus()
        val cpuStarted = Process.getElapsedCpuTime()

        // 1. Segmentation: 10 s windows every 2.5 s.
        val segmentationStarts =
            (0 until maxOf(1, audio.size - PyannoteSegmenter.WINDOW_SAMPLES + SEGMENTATION_HOP) step SEGMENTATION_HOP)
        val segmentationMillis = JSONArray()
        var loadStarted = System.nanoTime()
        val segmentationLoadMillis: Double
        PyannoteSegmenter.open(context, threads).use { segmenter ->
            segmentationLoadMillis = (System.nanoTime() - loadStarted) / 1e6
            segmentationFile.outputStream().buffered().use { output ->
                for (start in segmentationStarts) {
                    val started = System.nanoTime()
                    val logProbabilities = segmenter.logProbabilities(audio, start)
                    segmentationMillis.put((System.nanoTime() - started) / 1e6)
                    output.write(littleEndian(logProbabilities))
                }
            }
        }

        // 2. Voiceprints: 1.5 s windows every 0.75 s, then a few longer stretches for scaling.
        val embeddingMillis = JSONArray()
        val longMillis = JSONObject()
        loadStarted = System.nanoTime()
        val embeddingLoadMillis: Double
        var windows = 0
        var dimensions = 0
        val graphOptimisation =
            InstrumentationRegistry.getArguments().getString("voiceGraphOptimisation") == "on"
        val sampleScale = InstrumentationRegistry.getArguments().getString("voiceSampleScale")?.toFloat()
            ?: CamPlusPlusEmbedder.SAMPLE_SCALE
        val framePadding = InstrumentationRegistry.getArguments().getString("voiceFramePadding")?.toInt()
            ?: CamPlusPlusEmbedder.FRAME_PADDING
        val opened = voiceModel?.let { CamPlusPlusEmbedder.open(it, threads, graphOptimisation, sampleScale, framePadding) }
            ?: CamPlusPlusEmbedder.open(context, threads)
        opened.use { embedder ->
            dimensions = embedder.dimensions
            embeddingLoadMillis = (System.nanoTime() - loadStarted) / 1e6
            embeddingFile.outputStream().buffered().use { output ->
                var start = 0
                while (start + EMBEDDING_WINDOW <= audio.size) {
                    val started = System.nanoTime()
                    val voiceprint = checkNotNull(embedder.embed(audio, start, EMBEDDING_WINDOW))
                    embeddingMillis.put((System.nanoTime() - started) / 1e6)
                    output.write(littleEndian(voiceprint))
                    windows++
                    start += EMBEDDING_HOP
                }
            }
            for (seconds in intArrayOf(3, 5, 10)) {
                val length = seconds * SAMPLE_RATE
                if (length > audio.size) continue
                embedder.embed(audio, 0, length)
                val started = System.nanoTime()
                repeat(3) { embedder.embed(audio, 0, length) }
                longMillis.put(seconds.toString(), (System.nanoTime() - started) / 3e6)
            }
        }
        return JSONObject()
            .put("audio_seconds", audio.size.toDouble() / SAMPLE_RATE)
            .put("segmentation_load_ms", segmentationLoadMillis)
            .put("segmentation_chunk_ms", segmentationMillis)
            .put("embedding_load_ms", embeddingLoadMillis)
            .put("embedding_window_ms", embeddingMillis)
            .put("embedding_windows", windows)
            .put("embedding_dimensions", dimensions)
            .put("embedding_long_ms", longMillis)
            .put("cpu_ms", Process.getElapsedCpuTime() - cpuStarted)
            .put("memory_after_run", ParakeetBenchmarkSupport.memory())
            .put("thermal_status_before", thermalBefore)
            .put("thermal_status_after", thermalStatus())
    }

    private fun thermalStatus(): Any {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            context.getSystemService(PowerManager::class.java)?.currentThermalStatus ?: JSONObject.NULL
        } else {
            JSONObject.NULL
        }
    }

    private companion object {
        private const val BENCHMARK_PACKAGE = "com.example.voicememo.benchmark"
        private const val DIRECTORY = "speaker-chain-benchmark"
        private const val SAMPLE_RATE = 16_000
        private const val SEGMENTATION_HOP = 40_000
        private const val EMBEDDING_WINDOW = 24_000
        private const val EMBEDDING_HOP = 12_000
        private val SAFE_VALUE = Regex("[A-Za-z0-9-]{1,128}")

        private fun requireSafe(value: String?, label: String): String =
            value?.takeIf(SAFE_VALUE::matches) ?: error("$label is missing or invalid.")

        private fun littleEndian(values: FloatArray): ByteArray {
            val bytes = ByteBuffer.allocate(values.size * Float.SIZE_BYTES).order(ByteOrder.LITTLE_ENDIAN)
            bytes.asFloatBuffer().put(values)
            return bytes.array()
        }
    }
}
