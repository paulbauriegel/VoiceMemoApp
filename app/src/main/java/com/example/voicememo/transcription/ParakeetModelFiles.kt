package com.example.voicememo.transcription

import android.content.Context
import java.io.File
import java.security.MessageDigest

/**
 * Locates and verifies the side-installed Parakeet model in app-private storage.
 *
 * [isInstalled] only compares file sizes, so it is cheap enough for the UI and the
 * recorder. [verified] also checks SHA-256 once per file version and remembers the
 * result next to the model, so later loads do not re-hash 1.3 GB.
 */
object ParakeetModelFiles {
    class Files(val encoder: File, val decoder: File)

    private const val VERIFIED_MARKER = ".verified"

    fun directory(context: Context): File =
        File(context.applicationContext.filesDir, ParakeetAssets.MODEL_DIRECTORY)

    fun isInstalled(context: Context): Boolean {
        val directory = directory(context)
        return File(directory, ParakeetAssets.ENCODER_FILE).length() ==
            ParakeetAssets.ENCODER_SIZE_BYTES &&
            File(directory, ParakeetAssets.DECODER_FILE).length() ==
            ParakeetAssets.DECODER_SIZE_BYTES
    }

    @Synchronized
    fun verified(context: Context): Files {
        val directory = directory(context)
        val encoder = File(directory, ParakeetAssets.ENCODER_FILE)
        val decoder = File(directory, ParakeetAssets.DECODER_FILE)
        if (!isInstalled(context)) {
            throw WhisperModelContractException(
                "The Parakeet model is not installed on this device. " +
                    "Run scripts/install-parakeet-model.py with the phone connected.",
            )
        }
        val marker = File(directory, VERIFIED_MARKER)
        val identity = listOf(encoder, decoder).joinToString("\n") { file ->
            "${file.name}:${file.length()}:${file.lastModified()}"
        }
        if (marker.isFile && runCatching { marker.readText() }.getOrNull() == identity) {
            return Files(encoder, decoder)
        }
        check(sha256(encoder) == ParakeetAssets.ENCODER_SHA256) {
            "The installed Parakeet encoder does not match the pinned SHA-256."
        }
        check(sha256(decoder) == ParakeetAssets.DECODER_SHA256) {
            "The installed Parakeet decoder does not match the pinned SHA-256."
        }
        runCatching { marker.writeText(identity) }
        return Files(encoder, decoder)
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().buffered(1 shl 20).use { input ->
            val buffer = ByteArray(1 shl 20)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}
