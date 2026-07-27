package com.example.voicememo.transcription

enum class TranscriptionModel(
    val id: String,
    val displayName: String,
) {
    WHISPER_LARGE_V3_TURBO(
        id = "whisper-large-v3-turbo",
        displayName = "Whisper Large v3 Turbo",
    ),
    PARAKEET_TDT_0_6B_V3(
        id = "parakeet-tdt-0.6b-v3",
        displayName = "Parakeet TDT 0.6B v3",
    ),
    ;

    companion object {
        val DEFAULT = WHISPER_LARGE_V3_TURBO

        fun fromId(value: String?): TranscriptionModel =
            entries.firstOrNull { it.id == value } ?: DEFAULT
    }
}

data class TranscriptionWindowConfig(
    val payloadSamples: Int,
    val strideSamples: Int,
) {
    init {
        require(payloadSamples > 0)
        require(strideSamples in 1..payloadSamples)
    }
}

fun TranscriptionModel.windowConfig(): TranscriptionWindowConfig = when (this) {
    TranscriptionModel.WHISPER_LARGE_V3_TURBO -> TranscriptionWindowConfig(
        payloadSamples = WhisperChunker.PAYLOAD_SAMPLES,
        strideSamples = WhisperChunker.STRIDE_SAMPLES,
    )
    TranscriptionModel.PARAKEET_TDT_0_6B_V3 -> TranscriptionWindowConfig(
        payloadSamples = ParakeetAssets.MAX_SAMPLES,
        strideSamples = ParakeetAssets.STRIDE_SAMPLES,
    )
}

object ParakeetAssets {
    const val ROOT = "parakeet"
    const val MODEL = "$ROOT/parakeet_tdt_0.6b_v3_5s_i8_stateful.tflite"
    const val TOKENIZER = "$ROOT/tokenizer.json"
    const val PROCESSOR_CONFIG = "$ROOT/processor_config.json"
    const val CONFIG = "$ROOT/config.json"

    const val MODEL_REVISION = "e3a6f2dec6800733f97c87ff55822f32c405983a"
    const val TOKENIZER_REVISION = "7c35754d166cca382ad1e53e68b01e7c575f3a1d"
    const val MODEL_SIZE_BYTES = 614_261_072L
    const val MODEL_SHA256 = "334745b8bc7fd372b1c213516f0b6338bb827b1a2abb3e77ad35fe6fea5cd16b"
    const val TOKENIZER_SHA256 =
        "bd321b096832a3f270bd3b2a88823957920f1a5c5ada71114a26ea729d0cbe91"

    const val SAMPLE_RATE = 16_000
    const val INPUT_SECONDS = 5
    const val MAX_SAMPLES = SAMPLE_RATE * INPUT_SECONDS
    const val STRIDE_SAMPLES = MAX_SAMPLES / 2
    const val BLANK_TOKEN_ID = 8_192
}

internal object ParakeetInferenceFingerprint {
    fun forConfiguration(requestedBackend: WhisperBackendRequest): String = listOf(
        "model=${ParakeetAssets.MODEL_SHA256}",
        "tokenizer=${ParakeetAssets.TOKENIZER_SHA256}",
        "pipeline=parakeet-logmel-v1_tdt-stateful-v1_overlap-50-v1",
        "runtime=${requestedBackend.buildValue}-litert-compiled-model",
        "timestamps=token-duration",
        "vad=off",
    ).joinToString("|")
}
