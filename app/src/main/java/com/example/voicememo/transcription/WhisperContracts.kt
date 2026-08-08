package com.example.voicememo.transcription

import com.example.voicememo.BuildConfig
import java.io.Closeable

sealed interface WhisperLanguage {
    data object Auto : WhisperLanguage

    data class Forced(val code: String) : WhisperLanguage {
        init {
            require(code.matches(Regex("[a-z]{2,3}"))) {
                "Whisper language codes must contain two or three lowercase ASCII letters."
            }
        }
    }
}

data class WhisperChunkResult(
    /** Generated tokens only. Prompt and EOS tokens are excluded. */
    val tokens: List<Int>,
    val text: String,
    val detectedLanguage: String,
    val inferenceMillis: Long,
    val truncated: Boolean,
    val warnings: List<String> = emptyList(),
    val timedSegments: List<WhisperTimedSegment> = emptyList(),
)

data class WhisperTimedSegment(
    val startMillis: Long,
    val endMillis: Long,
    val text: String,
)

interface WhisperEngine : Closeable {
    /**
     * Transcribes one mono, 16 kHz-or-resampleable PCM payload of at most 30 seconds.
     *
     * Implementations check [shouldCancel] between full decoder invocations. LiteRT
     * inference itself is also made cancellable when [close] is called.
     */
    fun transcribeChunk(
        audio: FloatArray,
        language: WhisperLanguage,
        shouldCancel: () -> Boolean = { false },
        onTokenProgress: (generated: Int, limit: Int) -> Unit = { _, _ -> },
    ): WhisperChunkResult

    /**
     * Decodes an already-generated token sequence when the implementation owns
     * the matching tokenizer. The nullable default keeps test/fake engines small.
     */
    fun decodeTokens(tokens: List<Int>): String? = null
}

class WhisperCancelledException : RuntimeException("Whisper transcription was cancelled.")

class WhisperModelContractException(message: String) : IllegalStateException(message)

object WhisperAssets {
    const val ROOT = "whisper"
    val MODEL: String = BuildConfig.WHISPER_MODEL_ASSET_PATH
    val SPLIT_MODEL_PRODUCTION: Boolean = BuildConfig.WHISPER_SPLIT_MODEL_PRODUCTION
    val ENCODER_MODEL: String = BuildConfig.WHISPER_ENCODER_MODEL_ASSET_PATH
    val DECODER_MODEL: String = BuildConfig.WHISPER_DECODER_MODEL_ASSET_PATH
    const val TOKENIZER = "$ROOT/tokenizer.json"
    const val GENERATION_CONFIG = "$ROOT/generation_config.json"
    const val PREPROCESSOR_CONFIG = "$ROOT/preprocessor_config.json"

    const val MODEL_REVISION = "4e3af6a1597a54b6e47d2899cf4f6c39932b0b7e"
    const val TOKENIZER_REVISION = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
    val MODEL_SIZE_BYTES: Long = BuildConfig.WHISPER_MODEL_SIZE_BYTES
    val MODEL_SHA256: String = BuildConfig.WHISPER_MODEL_SHA256
    val ENCODER_MODEL_SIZE_BYTES: Long = BuildConfig.WHISPER_ENCODER_MODEL_SIZE_BYTES
    val ENCODER_MODEL_SHA256: String = BuildConfig.WHISPER_ENCODER_MODEL_SHA256
    val DECODER_MODEL_SIZE_BYTES: Long = BuildConfig.WHISPER_DECODER_MODEL_SIZE_BYTES
    val DECODER_MODEL_SHA256: String = BuildConfig.WHISPER_DECODER_MODEL_SHA256

    /** Stateful KV-cache decoder assets; bundled with the split production models. */
    val KV_DECODER_PRODUCTION: Boolean = BuildConfig.WHISPER_KV_DECODER_PRODUCTION
    val KV_CROSS_CACHE_MODEL: String = BuildConfig.WHISPER_KV_CROSS_CACHE_ASSET_PATH
    val KV_STEP_MODEL: String = BuildConfig.WHISPER_KV_STEP_ASSET_PATH
    val KV_PROJECTION_MODEL: String = BuildConfig.WHISPER_KV_PROJECTION_ASSET_PATH
}
