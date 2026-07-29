package com.example.voicememo.transcription

import com.example.voicememo.BuildConfig
import java.io.File
import java.util.concurrent.atomic.AtomicLong

/**
 * Diagnostic-only JNI adapter for pinned whisper.cpp benchmark models.
 *
 * This class lives in androidTest, and its native library is built only when
 * -PincludeWhisperCppBenchmark=true is supplied. Production transcription and
 * saved model selection cannot instantiate it.
 */
internal class WhisperCppBenchmarkEngine private constructor(
    private val handle: AtomicLong,
    private val backend: WhisperCppBenchmarkBackend,
) : WhisperEngine {
    override fun transcribeChunk(
        audio: FloatArray,
        language: WhisperLanguage,
        shouldCancel: () -> Boolean,
        onTokenProgress: (generated: Int, limit: Int) -> Unit,
    ): WhisperChunkResult = synchronized(handle) {
        if (shouldCancel()) throw WhisperCancelledException()
        require(audio.isNotEmpty() && audio.size <= MAX_SAMPLE_COUNT) {
            "whisper.cpp benchmark chunks must contain 1 to 30 seconds of 16 kHz audio."
        }
        val nativeHandle = requireOpenHandle()
        val languageCode = when (language) {
            WhisperLanguage.Auto -> "auto"
            is WhisperLanguage.Forced -> language.code
        }
        WhisperCppBenchmarkNative.transcribe(nativeHandle, audio, languageCode)
        if (shouldCancel()) throw WhisperCancelledException()
        val tokens = WhisperCppBenchmarkNative.getTokens(nativeHandle).toList()
        onTokenProgress(tokens.size, tokens.size)
        WhisperChunkResult(
            tokens = tokens,
            text = WhisperCppBenchmarkNative.getText(nativeHandle).trim(),
            detectedLanguage = WhisperCppBenchmarkNative.getLanguage(nativeHandle),
            inferenceMillis = WhisperCppBenchmarkNative.getInferenceMillis(nativeHandle),
            truncated = false,
            warnings = listOf(
                "benchmark-only:backend=${backend.argument}:${WhisperCppBenchmarkNative.runtimeInfo()}",
            ),
        )
    }

    override fun decodeTokens(tokens: List<Int>): String = synchronized(handle) {
        WhisperCppBenchmarkNative.decode(requireOpenHandle(), tokens.toIntArray())
    }

    override fun close() = synchronized(handle) {
        val nativeHandle = handle.getAndSet(0L)
        if (nativeHandle != 0L) {
            WhisperCppBenchmarkNative.destroy(nativeHandle)
        }
    }

    private fun requireOpenHandle(): Long =
        handle.get().takeIf { it != 0L }
            ?: error("The whisper.cpp benchmark engine is closed.")

    companion object {
        private const val MAX_SAMPLE_COUNT = 30 * 16_000
        fun open(
            modelFile: File,
            threadCount: Int,
            backend: WhisperCppBenchmarkBackend = WhisperCppBenchmarkBackend.CPU,
        ): WhisperCppBenchmarkEngine {
            check(isWhisperCppBenchmarkIncluded()) {
                "Rebuild with -PincludeWhisperCppBenchmark=true."
            }
            val model = WhisperCppBenchmarkModel.fromFileName(modelFile.name)
            require(modelFile.isFile && model != null) {
                "The staged whisper.cpp benchmark model is missing or unsupported."
            }
            require(modelFile.length() == model.sizeBytes) {
                "The staged whisper.cpp model size does not match the pinned artifact."
            }
            require(threadCount in 1..8) {
                "whisper.cpp benchmark threads must be between 1 and 8."
            }
            WhisperCppBenchmarkNative.load()
            return WhisperCppBenchmarkEngine(
                handle = AtomicLong(
                    WhisperCppBenchmarkNative.create(
                        modelPath = modelFile.absolutePath,
                        threads = threadCount,
                        useVulkan = backend == WhisperCppBenchmarkBackend.VULKAN,
                    ).also { check(it != 0L) { "whisper.cpp returned an empty handle." } },
                ),
                backend = backend,
            )
        }
    }
}

/**
 * Reads the target APK's value at runtime instead of letting the Kotlin compiler
 * inline a stale BuildConfig constant into the separately built test APK.
 */
internal fun isWhisperCppBenchmarkIncluded(): Boolean =
    BuildConfig::class.java
        .getField("WHISPER_CPP_BENCHMARK_INCLUDED")
        .getBoolean(null)

internal enum class WhisperCppBenchmarkModel(
    val argument: String,
    val fileName: String,
    val sizeBytes: Long,
    val sha256: String,
) {
    Q5_0(
        argument = "q5_0",
        fileName = "ggml-large-v3-turbo-q5_0.bin",
        sizeBytes = 574_041_195L,
        sha256 = "394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2",
    ),
    Q8_0(
        argument = "q8_0",
        fileName = "ggml-large-v3-turbo-q8_0.bin",
        sizeBytes = 874_188_075L,
        sha256 = "317eb69c11673c9de1e1f0d459b253999804ec71ac4c23c17ecf5fbe24e259a1",
    ),
    FP16(
        argument = "fp16",
        fileName = "ggml-large-v3-turbo.bin",
        sizeBytes = 1_624_555_275L,
        sha256 = "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69",
    ),
    ;

    companion object {
        fun fromFileName(value: String): WhisperCppBenchmarkModel? =
            entries.singleOrNull { it.fileName == value }
    }
}

internal enum class WhisperCppBenchmarkBackend(
    val argument: String,
) {
    CPU("cpu"),
    VULKAN("vulkan"),
    ;

    companion object {
        fun fromArgument(value: String): WhisperCppBenchmarkBackend =
            entries.singleOrNull { it.argument == value }
                ?: error("Unsupported whisper.cpp backend: $value")
    }
}

internal object WhisperCppBenchmarkNative {
    @Volatile
    private var loaded = false

    @Synchronized
    fun load() {
        if (loaded) return
        System.loadLibrary("voicememo_whispercpp_benchmark")
        loaded = true
    }

    external fun create(modelPath: String, threads: Int, useVulkan: Boolean): Long
    external fun transcribe(handle: Long, audio: FloatArray, language: String)
    external fun getText(handle: Long): String
    external fun getLanguage(handle: Long): String
    external fun getTokens(handle: Long): IntArray
    external fun getInferenceMillis(handle: Long): Long
    external fun decode(handle: Long, tokens: IntArray): String
    external fun runtimeInfo(): String
    external fun destroy(handle: Long)
}
