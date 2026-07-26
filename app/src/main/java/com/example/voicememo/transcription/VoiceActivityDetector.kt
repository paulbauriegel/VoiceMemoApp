package com.example.voicememo.transcription

/**
 * Whole-window speech gate used only to avoid unnecessary Whisper inference.
 *
 * Returning true keeps the original audio unchanged and sends it to Whisper.
 */
fun interface VoiceActivityDetector {
    fun hasSpeech(
        audio16Khz: FloatArray,
        shouldCancel: () -> Boolean,
    ): Boolean
}
