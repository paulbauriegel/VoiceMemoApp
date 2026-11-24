package com.example.voicememo.work

import com.example.voicememo.transcription.TimedWordsCodec
import com.example.voicememo.transcription.WhisperChunkResult
import java.io.File
import org.json.JSONArray
import org.json.JSONObject

internal object WhisperPreTranscriptionStorage {
    const val DIRECTORY_NAME = "whisper-pretranscription"
    const val FINISHED_MARKER = "recording-finished"

    fun directory(memoDirectory: File): File =
        File(memoDirectory, DIRECTORY_NAME)

    fun windowFile(directory: File, index: Int): File =
        File(directory, "window-${index.toString().padStart(6, '0')}.wav")

    fun resultFile(directory: File, index: Int): File =
        File(directory, "result-${index.toString().padStart(6, '0')}.json")

    fun finishedMarker(directory: File): File =
        File(directory, FINISHED_MARKER)

    fun writeResult(
        directory: File,
        index: Int,
        fingerprint: String,
        result: WhisperChunkResult,
        includeTimedWords: Boolean = false,
    ) {
        val destination = resultFile(directory, index)
        val temporary = File(directory, "${destination.name}.tmp")
        val json = JSONObject()
            .put("version", 1)
            .put("chunkIndex", index)
            .put("fingerprint", fingerprint)
            .put("text", result.text)
            .put("tokens", JSONArray(result.tokens))
            .put("detectedLanguage", result.detectedLanguage)
            .put("inferenceMillis", result.inferenceMillis)
            .put("truncated", result.truncated)
            .put("warnings", JSONArray(result.warnings))
        if (includeTimedWords) json.put("timedWords", TimedWordsCodec.encode(result.timedSegments))
        temporary.writeText(json.toString())
        check(temporary.renameTo(destination)) {
            "Could not publish pre-transcription result ${destination.name}."
        }
    }

    fun readResult(
        directory: File,
        index: Int,
        expectedFingerprint: String,
    ): WhisperChunkResult? {
        val file = resultFile(directory, index)
        if (!file.isFile) return null
        val json = runCatching { JSONObject(file.readText()) }.getOrNull() ?: return null
        if (json.optInt("version") != 1) return null
        if (json.optInt("chunkIndex", -1) != index) return null
        if (json.optString("fingerprint") != expectedFingerprint) return null
        val tokensJson = json.optJSONArray("tokens") ?: return null
        val warningsJson = json.optJSONArray("warnings") ?: JSONArray()
        return WhisperChunkResult(
            tokens = List(tokensJson.length(), tokensJson::getInt),
            text = json.optString("text"),
            detectedLanguage = json.optString("detectedLanguage", "und"),
            inferenceMillis = json.optLong("inferenceMillis"),
            truncated = json.optBoolean("truncated"),
            warnings = List(warningsJson.length(), warningsJson::getString),
            timedSegments = TimedWordsCodec.decode(json.optString("timedWords", "").ifEmpty { null })
                .orEmpty(),
        )
    }
}
