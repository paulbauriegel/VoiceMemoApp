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

/**
 * Parakeet TDT 0.6B v3, exported by `scripts/export-parakeet-litert-30s.py --split
 * --fp16-safe-layernorm --quant fp16`. The encoder runs on the GPU in F32_F16 mode (FP16
 * storage, FP32 accumulation); the small predict/joint decoder runs on the CPU.
 *
 * The 1.3 GB model does not fit next to Whisper inside a 4 GiB APK, so it lives in
 * app-private storage (`files/parakeet-litert/`), installed by
 * `scripts/install-parakeet-model.py`. Only the tokenizer is bundled.
 */
object ParakeetAssets {
    const val TOKENIZER = "parakeet/tokenizer.json"
    const val MODEL_DIRECTORY = "parakeet-litert"
    const val ENCODER_FILE = "parakeet_tdt_0.6b_v3_30s_fp16_lnsafe_encoder.tflite"
    const val DECODER_FILE = "parakeet_tdt_0.6b_v3_30s_fp16_lnsafe_decoder.tflite"
    const val ENCODER_SIZE_BYTES = 1_245_335_280L
    const val DECODER_SIZE_BYTES = 69_903_132L
    const val ENCODER_SHA256 = "a9f81d5dc0a46241f45f0d3103db1d7552dae37e1f6a81c2014eff81278187ea"
    const val DECODER_SHA256 = "575417424ca4d55dee5ec918044debf852fe049b806b2dcac3453b14ad4f7857"

    /** nvidia/parakeet-tdt-0.6b-v3 revision the export and tokenizer come from. */
    const val MODEL_REVISION = "7c35754d166cca382ad1e53e68b01e7c575f3a1d"
    const val TOKENIZER_REVISION = "7c35754d166cca382ad1e53e68b01e7c575f3a1d"
    const val TOKENIZER_SHA256 =
        "bd321b096832a3f270bd3b2a88823957920f1a5c5ada71114a26ea729d0cbe91"

    const val SAMPLE_RATE = 16_000
    const val WINDOW_SECONDS = 30
    const val STRIDE_SECONDS = 25
    const val MAX_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS
    const val STRIDE_SAMPLES = SAMPLE_RATE * STRIDE_SECONDS
    const val BLANK_TOKEN_ID = 8_192
    const val FRAME_MILLIS = 80L
}

internal object ParakeetInferenceFingerprint {
    fun forConfiguration(requestedBackend: WhisperBackendRequest): String = listOf(
        "model=${ParakeetAssets.ENCODER_SHA256}+${ParakeetAssets.DECODER_SHA256}",
        "tokenizer=${ParakeetAssets.TOKENIZER_SHA256}",
        "pipeline=parakeet-hf-logmel-v1_tdt-split-greedy-v1_window-30s-stride-25s_" +
            "tight-gap-repair-v1_aligned-word-merge-v1",
        "runtime=${requestedBackend.buildValue}-litert-compiled-model-f32f16",
        "timestamps=word",
        "vad=off",
    ).joinToString("|")
}
