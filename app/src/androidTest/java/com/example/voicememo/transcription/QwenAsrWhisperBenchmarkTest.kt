package com.example.voicememo.transcription

import android.os.Debug
import android.util.Base64
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.voicememo.VoiceMemoApplication
import java.io.File
import java.util.Locale
import kotlinx.coroutines.runBlocking
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Read-only whole-memo benchmark for the opt-in Qwen3-ASR LiteRT model.
 *
 * Audio, memo metadata, transcription jobs, and persisted transcripts are
 * never changed. The large model is supplied through an app-private cache path
 * instead of being bundled into the production APK.
 */
@RunWith(AndroidJUnit4::class)
class QwenAsrWhisperBenchmarkTest {
    @Test
    fun compareQwenCpuAndGpuToWhisper() = runBlocking {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val arguments = InstrumentationRegistry.getArguments()
        val context = instrumentation.targetContext
        val application = context.applicationContext as VoiceMemoApplication
        val memoId = requireArgument(ARG_MEMO_ID)
        val modelFile = File(requireArgument(ARG_MODEL_PATH))
        val tokenizerFile = File(requireArgument(ARG_TOKENIZER_PATH))
        val requestedBackends = arguments.getString(ARG_BACKENDS)
            ?.lowercase(Locale.ROOT)
            ?.split(',')
            ?.map(String::trim)
            ?.filter(String::isNotEmpty)
            ?.toSet()
            ?: setOf("cpu", "gpu")
        require(requestedBackends.isNotEmpty() && requestedBackends.all { it in setOf("cpu", "gpu") }) {
            "Instrumentation argument '$ARG_BACKENDS' must be cpu, gpu, or cpu,gpu."
        }
        val runId = arguments.getString(ARG_RUN_ID)
            ?.takeIf(SAFE_RUN_ID::matches)
            ?: System.currentTimeMillis().toString()
        val memo = requireNotNull(application.container.memoRepository.getMemo(memoId)) {
            "Unknown memo $memoId."
        }
        require(!memo.pendingDeletion) { "Memo $memoId is pending deletion." }
        val audioFile = File(memo.audioPath)
        require(audioFile.isFile) { "Memo audio is missing." }
        require(modelFile.isFile) { "Qwen model is missing: $modelFile" }
        require(tokenizerFile.isFile) { "Qwen tokenizer is missing: $tokenizerFile" }

        val reportDirectory = File(context.cacheDir, REPORT_DIRECTORY).apply { mkdirs() }
        val reportFile = File(reportDirectory, "qwen-$runId.json")
        val referenceText = String(
            Base64.decode(
                requireArgument(ARG_WHISPER_REFERENCE_TEXT_BASE64),
                Base64.NO_WRAP,
            ),
            Charsets.UTF_8,
        )
        val report = JSONObject()
            .put("run_id", runId)
            .put("memo_id", memoId)
            .put("persisted_output", false)
            .put("model_revision", MODEL_REVISION)
            .put("model_sha256", MODEL_SHA256)
            .put("tokenizer_revision", TOKENIZER_REVISION)
            .put(
                "whisper_cpu_baseline",
                JSONObject()
                    .put("source", "supplied_from_prior_same-device_run")
                    .put("text", referenceText),
            )
            .put("status", "RUNNING")
        reportFile.writeText(report.toString(2))

        var cpu: JSONObject? = null
        if ("cpu" in requestedBackends) {
            cpu = runCandidate(
                audioFile = audioFile,
                modelFile = modelFile,
                tokenizerFile = tokenizerFile,
                backend = WhisperBackendRequest.CPU,
                referenceText = referenceText,
            )
            report.put("qwen_cpu", cpu)
            reportFile.writeText(report.toString(2))
        }

        var gpu: JSONObject? = null
        if ("gpu" in requestedBackends) {
            gpu = runCandidate(
                audioFile = audioFile,
                modelFile = modelFile,
                tokenizerFile = tokenizerFile,
                backend = WhisperBackendRequest.GPU,
                referenceText = referenceText,
            )
            report.put("qwen_gpu", gpu)
            reportFile.writeText(report.toString(2))
        }
        if (
            cpu?.optString("status") == "COMPLETED" &&
            gpu?.optString("status") == "COMPLETED"
        ) {
            report.put(
                "qwen_gpu_vs_cpu_wer",
                wordErrorRate(cpu.getString("text"), gpu.getString("text")),
            )
        }
        report.put("status", "COMPLETED")
        reportFile.writeText(report.toString(2))
    }

    private fun runCandidate(
        audioFile: File,
        modelFile: File,
        tokenizerFile: File,
        backend: WhisperBackendRequest,
        referenceText: String,
    ): JSONObject = try {
        benchmark(audioFile, modelFile, tokenizerFile, backend, referenceText)
    } catch (failure: Throwable) {
        JSONObject()
            .put("status", "FAILED")
            .put("backend", backend.reportValue)
            .put("error_type", failure::class.java.name)
            .put("error", failure.message ?: failure.toString())
    }

    private fun benchmark(
        audioFile: File,
        modelFile: File,
        tokenizerFile: File,
        backend: WhisperBackendRequest,
        referenceText: String,
    ): JSONObject {
        val totalStarted = System.nanoTime()
        val loadStarted = System.nanoTime()
        val engine = LiteRtQwenAsrBenchmarkEngine.open(modelFile, tokenizerFile, backend)
        val loadMillis = elapsedMillis(loadStarted)
        try {
            val inferenceStarted = System.nanoTime()
            val chunkTexts = mutableListOf<String>()
            StreamingWavChunkSource.open(
                file = audioFile,
                windowConfig = TranscriptionWindowConfig(
                    payloadSamples = QwenAsrFeatureExtractor.MAX_SAMPLES,
                    strideSamples = QwenAsrFeatureExtractor.STRIDE_SAMPLES,
                ),
            ).use { source ->
                for (chunkIndex in 0 until source.chunkCount) {
                    chunkTexts += engine.transcribeChunk(
                        audio = source.readChunk(chunkIndex).samples,
                        language = WhisperLanguage.Auto,
                    ).text.trim()
                }
            }
            val text = mergeOverlappingChunks(chunkTexts)
            return JSONObject()
                .put("status", "COMPLETED")
                .put("model_id", "qwen3-asr-0.6b")
                .put("backend", backend.reportValue)
                .put("model_load_ms", loadMillis)
                .put("inference_ms", elapsedMillis(inferenceStarted))
                .put("total_ms", elapsedMillis(totalStarted))
                .put("native_heap_bytes", Debug.getNativeHeapAllocatedSize())
                .put("chunk_count", chunkTexts.size)
                .put("overlap_merger", "fuzzy-word-suffix-prefix-v1")
                .put("chunks", JSONArray(chunkTexts))
                .put("text", text)
                .put("wer_vs_whisper_cpu", wordErrorRate(referenceText, text))
        } finally {
            engine.close()
        }
    }

    /**
     * Qwen's graph does not expose token timestamps. With 50%-overlapped
     * windows, align the previous chunk's latter half against the current
     * chunk's first half and retain the best fuzzy boundary.
     */
    private fun mergeOverlappingChunks(chunks: List<String>): String {
        var merged = emptyList<String>()
        for (chunk in chunks) {
            val current = chunk.split(Regex("\\s+")).filter(String::isNotBlank)
            if (current.isEmpty()) continue
            if (merged.isEmpty()) {
                merged = current
                continue
            }
            val expectedPreviousStart = merged.size / 2
            val expectedCurrentEnd = current.size / 2
            var bestScore = Double.POSITIVE_INFINITY
            var bestPreviousStart = merged.size
            for (previousStart in
                (expectedPreviousStart - ALIGNMENT_WINDOW).coerceAtLeast(0)..
                    (expectedPreviousStart + ALIGNMENT_WINDOW).coerceAtMost(merged.size)
            ) {
                for (currentEnd in
                    (expectedCurrentEnd - ALIGNMENT_WINDOW).coerceAtLeast(1)..
                        (expectedCurrentEnd + ALIGNMENT_WINDOW).coerceAtMost(current.size)
                ) {
                    val previousOverlap = canonical(merged.subList(previousStart, merged.size))
                    val currentOverlap = canonical(current.subList(0, currentEnd))
                    val length = maxOf(previousOverlap.length, currentOverlap.length, 1)
                    val editRatio =
                        levenshtein(previousOverlap, currentOverlap).toDouble() / length
                    val positionPenalty =
                        kotlin.math.abs(previousStart - expectedPreviousStart) * 0.01 +
                            kotlin.math.abs(currentEnd - expectedCurrentEnd) * 0.01
                    val score = editRatio + positionPenalty
                    if (score < bestScore) {
                        bestScore = score
                        bestPreviousStart = previousStart
                    }
                }
            }
            merged = if (bestScore <= MAX_ALIGNMENT_SCORE) {
                // Prefer the newer chunk inside the overlap. Its leading half
                // is farther from right-padding and usually has more context
                // for the boundary phrase than the previous chunk's tail.
                merged.take(bestPreviousStart) + current
            } else {
                merged + current
            }
        }
        return merged.joinToString(" ").trim()
    }

    private fun canonical(tokens: List<String>): String =
        tokens.joinToString(" ")
            .lowercase(Locale.ROOT)
            .replace(Regex("[^\\p{L}\\p{N}']+"), " ")
            .trim()

    private fun levenshtein(left: String, right: String): Int {
        var previous = IntArray(right.length + 1) { it }
        left.forEachIndexed { leftIndex, leftChar ->
            val current = IntArray(right.length + 1)
            current[0] = leftIndex + 1
            right.forEachIndexed { rightIndex, rightChar ->
                current[rightIndex + 1] = minOf(
                    current[rightIndex] + 1,
                    previous[rightIndex + 1] + 1,
                    previous[rightIndex] + if (leftChar == rightChar) 0 else 1,
                )
            }
            previous = current
        }
        return previous.last()
    }

    private fun wordErrorRate(reference: String, candidate: String): Double {
        val expected = words(reference)
        val actual = words(candidate)
        if (expected.isEmpty()) return if (actual.isEmpty()) 0.0 else 1.0
        var previous = IntArray(actual.size + 1) { it }
        expected.forEachIndexed { expectedIndex, expectedWord ->
            val current = IntArray(actual.size + 1)
            current[0] = expectedIndex + 1
            actual.forEachIndexed { actualIndex, actualWord ->
                current[actualIndex + 1] = minOf(
                    current[actualIndex] + 1,
                    previous[actualIndex + 1] + 1,
                    previous[actualIndex] + if (expectedWord == actualWord) 0 else 1,
                )
            }
            previous = current
        }
        return previous.last().toDouble() / expected.size
    }

    private fun words(text: String): List<String> =
        text.lowercase(Locale.ROOT)
            .replace(Regex("[^\\p{L}\\p{N}']+"), " ")
            .trim()
            .split(Regex("\\s+"))
            .filter(String::isNotEmpty)

    private fun requireArgument(name: String): String =
        requireNotNull(InstrumentationRegistry.getArguments().getString(name)) {
            "Instrumentation argument '$name' is required."
        }

    private fun elapsedMillis(started: Long): Long =
        (System.nanoTime() - started) / 1_000_000L

    private companion object {
        const val ARG_MEMO_ID = "memo_id"
        const val ARG_RUN_ID = "run_id"
        const val ARG_MODEL_PATH = "qwen_model_path"
        const val ARG_TOKENIZER_PATH = "qwen_tokenizer_path"
        const val ARG_BACKENDS = "qwen_backends"
        const val ARG_WHISPER_REFERENCE_TEXT_BASE64 = "whisper_reference_text_b64"
        const val REPORT_DIRECTORY = "qwen-benchmark"
        const val ALIGNMENT_WINDOW = 3
        const val MAX_ALIGNMENT_SCORE = 0.65
        const val MODEL_REVISION = "dd8ed8ba3c32bccfe0a76c0cc81e9b7cf4f2ad81"
        const val MODEL_SHA256 =
            "952aab6a43f1fc1a76537e72852ea98e12ed407087eb6835bf15a96d4ad4c385"
        const val TOKENIZER_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
        val SAFE_RUN_ID = Regex("[A-Za-z0-9._-]{1,80}")
    }
}
