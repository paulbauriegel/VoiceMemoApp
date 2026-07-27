package com.example.voicememo.transcription

import android.content.Context
import android.os.Debug
import android.util.Base64
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.voicememo.VoiceMemoApplication
import java.io.File
import java.util.Locale
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Read-only whole-memo benchmark for Parakeet CPU/GPU against the existing
 * Whisper CPU baseline. No transcript, job, memo, or recording is mutated.
 */
@RunWith(AndroidJUnit4::class)
class ParakeetWhisperBenchmarkTest {
    @Test
    fun compareParakeetCpuAndGpuToWhisper() = runBlocking {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val arguments = InstrumentationRegistry.getArguments()
        val context = instrumentation.targetContext
        val application = context.applicationContext as VoiceMemoApplication
        val memoId = requireNotNull(arguments.getString(ARG_MEMO_ID)) {
            "Instrumentation argument '$ARG_MEMO_ID' is required."
        }
        val runId = arguments.getString(ARG_RUN_ID)
            ?.takeIf { SAFE_RUN_ID.matches(it) }
            ?: System.currentTimeMillis().toString()
        val memo = requireNotNull(application.container.memoRepository.getMemo(memoId)) {
            "Unknown memo $memoId."
        }
        require(!memo.pendingDeletion) { "Memo $memoId is pending deletion." }
        val audioFile = File(memo.audioPath)
        require(audioFile.isFile) { "Memo audio is missing." }

        val reportDirectory = File(context.cacheDir, REPORT_DIRECTORY).apply { mkdirs() }
        val reportFile = File(reportDirectory, "parakeet-$runId.json")
        val report = JSONObject()
            .put("run_id", runId)
            .put("memo_id", memoId)
            .put("persisted_output", false)
            .put("status", "RUNNING")
        reportFile.writeText(report.toString(2))

        val suppliedReference = arguments.getString(ARG_WHISPER_REFERENCE_TEXT_BASE64)
            ?.let { encoded ->
                String(Base64.decode(encoded, Base64.NO_WRAP), Charsets.UTF_8)
            }
        val whisperReferenceText = if (suppliedReference == null) {
            val whisper = benchmark(
                context = context,
                audioFile = audioFile,
                model = TranscriptionModel.WHISPER_LARGE_V3_TURBO,
                backend = WhisperBackendRequest.CPU,
                openEngines = {
                    val first = LiteRtWhisperEngine.open(context, threadCount = 3)
                    try {
                        listOf(first, LiteRtWhisperEngine.open(context, threadCount = 3))
                    } catch (failure: Throwable) {
                        first.close()
                        throw failure
                    }
                },
            )
            report.put("whisper_cpu_baseline", whisper.toJson())
            whisper.transcript.text
        } else {
            report.put(
                "whisper_cpu_baseline",
                JSONObject()
                    .put("source", "supplied_from_prior_same-device_run")
                    .put("text", suppliedReference),
            )
            suppliedReference
        }
        reportFile.writeText(report.toString(2))

        val parakeetCpu = benchmark(
            context = context,
            audioFile = audioFile,
            model = TranscriptionModel.PARAKEET_TDT_0_6B_V3,
            backend = WhisperBackendRequest.CPU,
            referenceText = whisperReferenceText,
            openEngines = {
                listOf(LiteRtParakeetEngine.open(context, WhisperBackendRequest.CPU))
            },
        )
        report.put("parakeet_cpu", parakeetCpu.toJson())
        reportFile.writeText(report.toString(2))

        val parakeetGpu = benchmark(
            context = context,
            audioFile = audioFile,
            model = TranscriptionModel.PARAKEET_TDT_0_6B_V3,
            backend = WhisperBackendRequest.GPU,
            referenceText = whisperReferenceText,
            openEngines = {
                listOf(LiteRtParakeetEngine.open(context, WhisperBackendRequest.GPU))
            },
        )
        report
            .put("parakeet_gpu", parakeetGpu.toJson())
            .put(
                "parakeet_gpu_vs_cpu_wer",
                wordErrorRate(parakeetCpu.transcript.text, parakeetGpu.transcript.text),
            )
            .put("status", "COMPLETED")
        reportFile.writeText(report.toString(2))
    }

    private fun benchmark(
        context: Context,
        audioFile: File,
        model: TranscriptionModel,
        backend: WhisperBackendRequest,
        referenceText: String? = null,
        openEngines: () -> List<WhisperEngine>,
    ): BenchmarkResult {
        val totalStarted = System.nanoTime()
        val loadStarted = System.nanoTime()
        val engines = openEngines()
        val loadMillis = elapsedMillis(loadStarted)
        try {
            val inferenceStarted = System.nanoTime()
            val transcript = StreamingWavChunkSource.open(
                file = audioFile,
                windowConfig = model.windowConfig(),
            ).use { source ->
                LongFormTranscriber(
                    engines = engines,
                    lockAutoDetectedLanguage =
                        model == TranscriptionModel.WHISPER_LARGE_V3_TURBO,
                    trimOverlappingTokensAtMillis =
                        if (model == TranscriptionModel.PARAKEET_TDT_0_6B_V3) {
                            2_500L
                        } else {
                            null
                        },
                ).transcribeBatches(
                    windows = (0 until source.chunkCount).map(source::window),
                    loadAudio = { window -> source.readChunk(window.index).samples },
                    language = WhisperLanguage.Auto,
                )
            }
            return BenchmarkResult(
                model = model,
                backend = backend,
                transcript = transcript,
                modelLoadMillis = loadMillis,
                inferenceMillis = elapsedMillis(inferenceStarted),
                totalMillis = elapsedMillis(totalStarted),
                wordErrorRate = referenceText?.let {
                    wordErrorRate(it, transcript.text)
                },
                nativeHeapBytes = Debug.getNativeHeapAllocatedSize(),
            )
        } finally {
            engines.asReversed().forEach { runCatching(it::close) }
        }
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

    private fun elapsedMillis(started: Long): Long =
        (System.nanoTime() - started) / 1_000_000L

    private data class BenchmarkResult(
        val model: TranscriptionModel,
        val backend: WhisperBackendRequest,
        val transcript: LongFormTranscript,
        val modelLoadMillis: Long,
        val inferenceMillis: Long,
        val totalMillis: Long,
        val wordErrorRate: Double?,
        val nativeHeapBytes: Long,
    ) {
        fun toJson() = JSONObject()
            .put("model_id", model.id)
            .put("backend", backend.reportValue)
            .put("model_load_ms", modelLoadMillis)
            .put("inference_ms", inferenceMillis)
            .put("total_ms", totalMillis)
            .put("native_heap_bytes", nativeHeapBytes)
            .put("chunk_count", transcript.chunks.size)
            .put("detected_language", transcript.detectedLanguage)
            .put("text", transcript.text)
            .put("wer_vs_whisper_cpu", wordErrorRate ?: JSONObject.NULL)
    }

    private companion object {
        const val ARG_MEMO_ID = "memo_id"
        const val ARG_RUN_ID = "run_id"
        const val ARG_WHISPER_REFERENCE_TEXT_BASE64 = "whisper_reference_text_b64"
        const val REPORT_DIRECTORY = "parakeet-benchmark"
        val SAFE_RUN_ID = Regex("[A-Za-z0-9._-]{1,80}")
    }
}
