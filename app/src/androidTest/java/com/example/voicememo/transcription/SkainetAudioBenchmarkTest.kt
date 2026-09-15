package com.example.voicememo.transcription

import android.content.Context
import android.os.Build
import android.os.Debug
import android.os.PowerManager
import android.os.Process
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.voicememo.BuildConfig
import java.io.File
import java.lang.reflect.Method
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.abs
import kotlin.math.sqrt
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Isolated, non-persisting comparison of VoiceMemo and SKaiNET audio frontends.
 *
 * The host runner must build the target with `-PisolatedBenchmarkPackage=true`.
 * Inputs and reports live only under that separate package's private storage.
 */
@RunWith(AndroidJUnit4::class)
class SkainetAudioBenchmarkTest {
    @Test
    fun runBenchmark() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val arguments = InstrumentationRegistry.getArguments()
        val context = instrumentation.targetContext
        check(context.packageName == BENCHMARK_PACKAGE) {
            "Refusing to run against non-isolated package ${context.packageName}."
        }

        val runId = requireSafe(arguments.getString(ARG_RUN_ID), ARG_RUN_ID)
        val mode = requireSafe(arguments.getString(ARG_MODE), ARG_MODE)
        require(mode in SUPPORTED_MODES) { "Unsupported benchmark mode: $mode" }
        val audioName = requireSafeFile(arguments.getString(ARG_AUDIO_NAME))
        val audioFile = requireStagedAudio(context, audioName)
        val languageCode = requireSafe(arguments.getString(ARG_LANGUAGE), ARG_LANGUAGE)
        require(LANGUAGE_PATTERN.matches(languageCode)) { "Invalid language: $languageCode" }

        val reportDirectory = File(context.cacheDir, REPORT_DIRECTORY).apply { mkdirs() }
        val reportFile = File(reportDirectory, "$runId.json")
        val temporaryReport = File(reportDirectory, "$runId.json.tmp")
        reportFile.delete()
        temporaryReport.delete()

        val report = JSONObject()
            .put("schema_version", 1)
            .put("run_id", runId)
            .put("mode", mode)
            .put("package_name", context.packageName)
            .put("audio_file_name", audioName)
            .put("audio_sha256", sha256(audioFile))
            .put("requested_language", languageCode)
            .put("persisted_output", false)
            .put("worker_count", WORKER_COUNT)
            .put("threads_per_worker", THREADS_PER_WORKER)
            .put("skainet_audio_version", SKAINET_VERSION)
            .put("build", buildJson())
            .put("device", deviceJson())
            .put("status", STATUS_RUNNING)

        fun publish() {
            temporaryReport.writeText(report.toString(2))
            reportFile.delete()
            check(temporaryReport.renameTo(reportFile)) {
                "Could not publish SKaiNET audio benchmark report."
            }
        }

        publish()
        try {
            if (mode == MODE_PARITY) {
                report.put("parity", parity(audioFile))
            } else {
                report.put(
                    "result",
                    benchmark(
                        context = context,
                        audioFile = audioFile,
                        languageCode = languageCode,
                        mode = mode,
                    ),
                )
            }
            report.put("status", STATUS_COMPLETED)
        } catch (error: Throwable) {
            report
                .put("status", STATUS_FAILED)
                .put("error_type", error.javaClass.name)
                .put("error_message", sanitizedMessage(error))
            throw error
        } finally {
            publish()
        }
    }

    private fun benchmark(
        context: Context,
        audioFile: File,
        languageCode: String,
        mode: String,
    ): JSONObject {
        val inputMetrics = DurationMetrics()
        val featureMetrics = DurationMetrics()
        val frontendInitializationStarted = System.nanoTime()
        val providers = List(WORKER_COUNT) {
            TimedFeatureProvider(newFeatureProvider(mode), featureMetrics)
        }
        val frontendInitializationMillis = elapsedMillis(frontendInitializationStarted)

        val loadStarted = System.nanoTime()
        val loadCpuStarted = Process.getElapsedCpuTime()
        val engines = mutableListOf<LiteRtWhisperEngine>()
        try {
            providers.forEach { provider ->
                engines += LiteRtWhisperEngine.openForBenchmark(
                    context = context,
                    threadCount = THREADS_PER_WORKER,
                    featureExtractor = provider,
                )
            }
        } catch (error: Throwable) {
            engines.asReversed().forEach { runCatching(it::close) }
            throw error
        }
        val modelLoadMillis = elapsedMillis(loadStarted)
        val modelLoadCpuMillis = Process.getElapsedCpuTime() - loadCpuStarted

        val thermalBefore = thermalStatus(context)
        val pipelineStarted = System.nanoTime()
        val pipelineCpuStarted = Process.getElapsedCpuTime()
        val audio = try {
            openAudio(audioFile, mode, inputMetrics)
        } catch (error: Throwable) {
            engines.asReversed().forEach { runCatching(it::close) }
            throw error
        }
        val inferenceStarted = System.nanoTime()
        val inferenceCpuStarted = Process.getElapsedCpuTime()
        val transcript = try {
            LongFormTranscriber(engines).transcribeBatches(
                windows = audio.windows,
                loadAudio = audio.loadAudio,
                language = WhisperLanguage.Forced(languageCode),
            )
        } finally {
            engines.asReversed().forEach { runCatching(it::close) }
            audio.close()
        }
        val inferenceWallMillis = elapsedMillis(inferenceStarted)
        val inferenceCpuMillis = Process.getElapsedCpuTime() - inferenceCpuStarted
        val pipelineWallMillis = elapsedMillis(pipelineStarted)
        val pipelineCpuMillis = Process.getElapsedCpuTime() - pipelineCpuStarted
        val audioDurationMillis = audio.sampleCount * 1_000.0 / SAMPLE_RATE

        return JSONObject()
            .put("frontend", mode)
            .put("audio_sample_count", audio.sampleCount)
            .put("audio_duration_millis", audioDurationMillis)
            .put("chunk_count", audio.windows.size)
            .put("frontend_initialization_millis", frontendInitializationMillis)
            .put("model_load_wall_millis", modelLoadMillis)
            .put("model_load_cpu_millis", modelLoadCpuMillis)
            .put("input_preparation_wall_millis", inputMetrics.wallMillis())
            .put("input_preparation_cpu_millis", inputMetrics.cpuMillis())
            .put("feature_extraction_wall_millis", featureMetrics.wallMillis())
            .put("feature_extraction_cpu_millis", featureMetrics.cpuMillis())
            .put("feature_extraction_calls", featureMetrics.calls.get())
            .put("inference_wall_millis", inferenceWallMillis)
            .put("inference_cpu_millis", inferenceCpuMillis)
            .put("pipeline_wall_millis", pipelineWallMillis)
            .put("pipeline_cpu_millis", pipelineCpuMillis)
            .put("pipeline_rtf", pipelineWallMillis / audioDurationMillis)
            .put("feature_extraction_rtf", featureMetrics.wallMillis() / audioDurationMillis)
            .put("detected_language", transcript.detectedLanguage)
            .put("text", transcript.text)
            .put("tokens", JSONArray(transcript.tokens))
            .put("token_count", transcript.tokens.size)
            .put("warnings", JSONArray(transcript.warnings))
            .put("chunks", chunksJson(transcript))
            .put("thermal_status_before", thermalBefore)
            .put("thermal_status_after", thermalStatus(context))
            .put("memory_after", memoryJson())
    }

    private fun openAudio(
        audioFile: File,
        mode: String,
        metrics: DurationMetrics,
    ): BenchmarkAudio = when (mode) {
        MODE_VOICEMEMO -> {
            val source = metrics.measure { StreamingWavChunkSource.open(audioFile) }
            BenchmarkAudio(
                sampleCount = source.info.whisperSampleCount,
                windows = List(source.chunkCount, source::window),
                loadAudio = { window -> metrics.measure { source.readChunk(window.index).samples } },
                close = source::close,
            )
        }

        MODE_SKAINET -> {
            val pcm = metrics.measure {
                SkainetAudioBridge.decodeTo16k(audioFile.readBytes())
            }
            BenchmarkAudio(
                sampleCount = pcm.size,
                windows = WhisperChunker.windows(pcm.size),
                loadAudio = { window ->
                    metrics.measure { pcm.copyOfRange(window.startSample, window.endSample) }
                },
                close = {},
            )
        }

        else -> error("Unsupported timed mode: $mode")
    }

    private fun newFeatureProvider(mode: String): WhisperFeatureProvider = when (mode) {
        MODE_VOICEMEMO -> WhisperFeatureExtractor()
        MODE_SKAINET -> SkainetWhisperFeatureProvider()
        else -> error("Unsupported frontend: $mode")
    }

    private fun parity(audioFile: File): JSONObject {
        val skainetPcm = SkainetAudioBridge.decodeTo16k(audioFile.readBytes())
        val pcmComparison = FloatComparison()
        val featureComparison = FloatComparison()
        val windows = JSONArray()
        val currentFeatures = WhisperFeatureExtractor()
        val skainetFeatures = SkainetWhisperFeatureProvider()

        StreamingWavChunkSource.open(audioFile).use { currentAudio ->
            require(currentAudio.info.whisperSampleCount == skainetPcm.size) {
                "Frontend sample counts differ: ${currentAudio.info.whisperSampleCount} vs ${skainetPcm.size}."
            }
            repeat(currentAudio.chunkCount) { index ->
                val window = currentAudio.window(index)
                val currentSamples = currentAudio.readChunk(index).samples
                val candidateSamples = skainetPcm.copyOfRange(window.startSample, window.endSample)
                val windowPcm = FloatComparison().also {
                    it.add(currentSamples, candidateSamples)
                    pcmComparison.add(currentSamples, candidateSamples)
                }
                val referenceMel = currentFeatures.extract(currentSamples)
                val candidateMel = skainetFeatures.extract(candidateSamples)
                val windowFeatures = FloatComparison().also {
                    it.add(referenceMel, candidateMel)
                    featureComparison.add(referenceMel, candidateMel)
                }
                windows.put(
                    JSONObject()
                        .put("index", index)
                        .put("start_sample", window.startSample)
                        .put("end_sample", window.endSample)
                        .put("voicememo_pcm_sha256", floatSha256(currentSamples))
                        .put("skainet_pcm_sha256", floatSha256(candidateSamples))
                        .put("pcm", windowPcm.toJson())
                        .put("voicememo_mel_sha256", floatSha256(referenceMel))
                        .put("skainet_mel_sha256", floatSha256(candidateMel))
                        .put("mel", windowFeatures.toJson()),
                )
            }
        }

        return JSONObject()
            .put("audio_sample_count", skainetPcm.size)
            .put("chunk_count", windows.length())
            .put("pcm", pcmComparison.toJson())
            .put("mel", featureComparison.toJson())
            .put("windows", windows)
    }

    private fun chunksJson(transcript: LongFormTranscript): JSONArray =
        JSONArray().also { array ->
            transcript.chunks.forEach { chunk ->
                array.put(
                    JSONObject()
                        .put("index", chunk.window.index)
                        .put("start_sample", chunk.window.startSample)
                        .put("end_sample", chunk.window.endSample)
                        .put("inference_millis", chunk.result.inferenceMillis)
                        .put("token_count", chunk.result.tokens.size)
                        .put("truncated", chunk.result.truncated)
                        .put("skipped_by_vad", chunk.skippedByVad)
                        .put("retried_as_half_windows", chunk.retriedAsHalfWindows),
                )
            }
        }

    private class SkainetWhisperFeatureProvider : WhisperFeatureProvider {
        private val extractor = SkainetAudioBridge.newWhisperMelExtractor()

        override fun extract(pcm16Khz: FloatArray): FloatArray =
            extractor.extract(pcm16Khz)
    }

    /**
     * Cached bridge to the exact published 0.2.1 runtime. The dependency was
     * compiled with Kotlin 2.4 metadata, so keeping it off the Kotlin 2.2
     * compile classpath avoids an app-wide compiler migration.
     */
    private object SkainetAudioBridge {
        private val wavClass = Class.forName("sk.ainet.audio.wav.Wav")
        private val wavInstance = wavClass.getField("INSTANCE").get(null)
        private val wavDecode = wavClass.getMethod("decode", ByteArray::class.java)
        private val resamplerClass = Class.forName("sk.ainet.audio.core.Resampler")
        private val resamplerInstance = resamplerClass.getField("INSTANCE").get(null)
        private val resampleTo16k = resamplerClass.getMethod(
            "to16k",
            FloatArray::class.java,
            Int::class.javaPrimitiveType,
        )

        fun decodeTo16k(bytes: ByteArray): FloatArray {
            val decoded = wavDecode.invoke(wavInstance, bytes)
            val sampleRate = decoded.javaClass.getMethod("getSampleRate").invoke(decoded) as Int
            val pcm = decoded.javaClass.getMethod("getPcm").invoke(decoded) as FloatArray
            return resampleTo16k.invoke(resamplerInstance, pcm, sampleRate) as FloatArray
        }

        fun newWhisperMelExtractor(): ReflectiveMelExtractor {
            val filterbankClass = Class.forName("sk.ainet.audio.mel.MelFilterbanks")
            val filterbankInstance = filterbankClass.getField("INSTANCE").get(null)
            val filterbank = filterbankClass.getMethod(
                "slaney",
                Int::class.javaPrimitiveType,
                Int::class.javaPrimitiveType,
                Int::class.javaPrimitiveType,
                Float::class.javaPrimitiveType,
                Float::class.javaPrimitiveType,
            ).invoke(
                filterbankInstance,
                WhisperFeatureExtractor.MEL_BINS,
                WhisperFeatureExtractor.FFT_SIZE,
                WhisperFeatureExtractor.SAMPLE_RATE,
                0.0f,
                WhisperFeatureExtractor.SAMPLE_RATE / 2.0f,
            ) as FloatArray

            val normClass = Class.forName("sk.ainet.audio.mel.MelNorm")
            val whisperNorm = normClass.getField("WHISPER").get(null)
            val configClass = Class.forName("sk.ainet.audio.mel.MelConfig")
            val config = configClass.getConstructor(
                Int::class.javaPrimitiveType,
                Int::class.javaPrimitiveType,
                Int::class.javaPrimitiveType,
                Integer::class.java,
                Boolean::class.javaPrimitiveType,
                normClass,
            ).newInstance(
                WhisperFeatureExtractor.MEL_BINS,
                WhisperFeatureExtractor.FFT_SIZE,
                WhisperFeatureExtractor.HOP_LENGTH,
                WhisperFeatureExtractor.FRAMES,
                true,
                whisperNorm,
            )
            val extractorClass = Class.forName("sk.ainet.audio.mel.MelSpectrogram")
            val extractor = extractorClass.getConstructor(FloatArray::class.java, configClass)
                .newInstance(filterbank, config)
            return ReflectiveMelExtractor(
                extractor = extractor,
                extract = extractorClass.getMethod("extract", FloatArray::class.java),
            )
        }
    }

    private class ReflectiveMelExtractor(
        private val extractor: Any,
        private val extract: Method,
    ) {
        private var dataGetter: Method? = null

        fun extract(pcm16Khz: FloatArray): FloatArray {
            val frames = extract.invoke(extractor, pcm16Khz)
            val getter = dataGetter ?: frames.javaClass.getMethod("getData").also {
                dataGetter = it
            }
            return getter.invoke(frames) as FloatArray
        }
    }

    private class TimedFeatureProvider(
        private val delegate: WhisperFeatureProvider,
        private val metrics: DurationMetrics,
    ) : WhisperFeatureProvider {
        override fun extract(pcm16Khz: FloatArray): FloatArray =
            metrics.measure { delegate.extract(pcm16Khz) }
    }

    private class DurationMetrics {
        val calls = AtomicLong()
        private val wallNanos = AtomicLong()
        private val cpuNanos = AtomicLong()

        fun <T> measure(block: () -> T): T {
            val wallStarted = System.nanoTime()
            val cpuStarted = Debug.threadCpuTimeNanos()
            return try {
                block()
            } finally {
                cpuNanos.addAndGet(Debug.threadCpuTimeNanos() - cpuStarted)
                wallNanos.addAndGet(System.nanoTime() - wallStarted)
                calls.incrementAndGet()
            }
        }

        fun wallMillis(): Double = wallNanos.get() / 1_000_000.0

        fun cpuMillis(): Double = cpuNanos.get() / 1_000_000.0
    }

    private data class BenchmarkAudio(
        val sampleCount: Int,
        val windows: List<AudioChunkWindow>,
        val loadAudio: (AudioChunkWindow) -> FloatArray,
        val close: () -> Unit,
    )

    private class FloatComparison {
        private var count = 0L
        private var absoluteError = 0.0
        private var maximumAbsoluteError = 0.0
        private var dot = 0.0
        private var referenceNorm = 0.0
        private var candidateNorm = 0.0

        fun add(reference: FloatArray, candidate: FloatArray) {
            require(reference.size == candidate.size) {
                "Float arrays differ in size: ${reference.size} vs ${candidate.size}."
            }
            for (index in reference.indices) {
                val left = reference[index].toDouble()
                val right = candidate[index].toDouble()
                val difference = abs(left - right)
                absoluteError += difference
                maximumAbsoluteError = maxOf(maximumAbsoluteError, difference)
                dot += left * right
                referenceNorm += left * left
                candidateNorm += right * right
            }
            count += reference.size
        }

        fun toJson(): JSONObject = JSONObject()
            .put("element_count", count)
            .put("mean_absolute_error", if (count == 0L) 0.0 else absoluteError / count)
            .put("maximum_absolute_error", maximumAbsoluteError)
            .put(
                "cosine_similarity",
                dot / (sqrt(referenceNorm) * sqrt(candidateNorm) + 1.0e-12),
            )
    }

    companion object {
        private const val BENCHMARK_PACKAGE = "com.example.voicememo.benchmark"
        private const val SKAINET_VERSION = "0.2.1"
        private const val WORKER_COUNT = 2
        private const val THREADS_PER_WORKER = 3
        private const val SAMPLE_RATE = 16_000.0
        private const val REPORT_DIRECTORY = "skainet-audio-benchmark"
        private const val AUDIO_DIRECTORY = "skainet-audio-benchmark"
        private const val ARG_RUN_ID = "runId"
        private const val ARG_MODE = "mode"
        private const val ARG_AUDIO_NAME = "externalAudioName"
        private const val ARG_LANGUAGE = "forcedLanguage"
        private const val MODE_VOICEMEMO = "voicememo"
        private const val MODE_SKAINET = "skainet"
        private const val MODE_PARITY = "parity"
        private const val STATUS_RUNNING = "running"
        private const val STATUS_COMPLETED = "completed"
        private const val STATUS_FAILED = "failed"
        private val SUPPORTED_MODES = setOf(MODE_VOICEMEMO, MODE_SKAINET, MODE_PARITY)
        private val SAFE_VALUE = Regex("[A-Za-z0-9-]{1,128}")
        private val SAFE_FILE = Regex("[A-Za-z0-9][A-Za-z0-9._-]{0,127}\\.wav")
        private val LANGUAGE_PATTERN = Regex("[a-z]{2,3}")

        private fun requireSafe(value: String?, label: String): String =
            value?.takeIf(SAFE_VALUE::matches)
                ?: error("$label is missing or invalid.")

        private fun requireSafeFile(value: String?): String =
            value?.takeIf(SAFE_FILE::matches)
                ?: error("$ARG_AUDIO_NAME is missing or invalid.")

        private fun requireStagedAudio(context: Context, name: String): File {
            val parent = File(context.filesDir, AUDIO_DIRECTORY).canonicalFile
            val file = File(parent, name).canonicalFile
            require(file.parentFile == parent && file.isFile) { "Staged audio is missing." }
            return file
        }

        private fun elapsedMillis(startedNanos: Long): Long =
            (System.nanoTime() - startedNanos) / 1_000_000L

        private fun floatSha256(values: FloatArray): String {
            val digest = MessageDigest.getInstance("SHA-256")
            val buffer = ByteBuffer.allocate(4 * 4_096).order(ByteOrder.LITTLE_ENDIAN)
            values.forEach { value ->
                if (buffer.remaining() < Float.SIZE_BYTES) {
                    digest.update(buffer.array(), 0, buffer.position())
                    buffer.clear()
                }
                buffer.putFloat(value)
            }
            digest.update(buffer.array(), 0, buffer.position())
            return digest.digest().joinToString("") { "%02x".format(it) }
        }

        private fun sha256(file: File): String {
            val digest = MessageDigest.getInstance("SHA-256")
            file.inputStream().buffered().use { input ->
                val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                while (true) {
                    val count = input.read(buffer)
                    if (count < 0) break
                    digest.update(buffer, 0, count)
                }
            }
            return digest.digest().joinToString("") { "%02x".format(it) }
        }

        private fun thermalStatus(context: Context): Any =
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                context.getSystemService(PowerManager::class.java)?.currentThermalStatus
                    ?: JSONObject.NULL
            } else {
                JSONObject.NULL
            }

        private fun memoryJson(): JSONObject {
            val info = Debug.MemoryInfo()
            Debug.getMemoryInfo(info)
            val runtime = Runtime.getRuntime()
            return JSONObject()
                .put("total_pss_kib", info.totalPss)
                .put("private_dirty_kib", info.totalPrivateDirty)
                .put("native_heap_allocated_bytes", Debug.getNativeHeapAllocatedSize())
                .put("java_heap_used_bytes", runtime.totalMemory() - runtime.freeMemory())
                .put("java_heap_max_bytes", runtime.maxMemory())
        }

        private fun deviceJson(): JSONObject = JSONObject()
            .put("manufacturer", Build.MANUFACTURER)
            .put("model", Build.MODEL)
            .put("device", Build.DEVICE)
            .put("soc", if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) Build.SOC_MODEL else "")
            .put("android_release", Build.VERSION.RELEASE)
            .put("sdk", Build.VERSION.SDK_INT)
            .put("fingerprint", Build.FINGERPRINT)

        private fun buildJson(): JSONObject = JSONObject()
            .put("application_id", BuildConfig.APPLICATION_ID)
            .put("version_name", BuildConfig.VERSION_NAME)
            .put("version_code", BuildConfig.VERSION_CODE)
            .put("whisper_model_asset_path", BuildConfig.WHISPER_MODEL_ASSET_PATH)
            .put("whisper_model_size_bytes", BuildConfig.WHISPER_MODEL_SIZE_BYTES)
            .put("whisper_model_sha256", BuildConfig.WHISPER_MODEL_SHA256)

        private fun sanitizedMessage(error: Throwable): String =
            error.message
                ?.lineSequence()
                ?.firstOrNull()
                ?.trim()
                ?.take(500)
                ?.takeIf(String::isNotEmpty)
                ?: error.javaClass.simpleName
    }
}
