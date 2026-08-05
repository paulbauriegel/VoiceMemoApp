package com.example.voicememo.transcription

import android.content.Context
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.SamplerConfig
import java.io.File
import java.io.FileOutputStream
import java.util.UUID

data class RecordingMetadata(
    val title: String,
    val summary: String,
)

/**
 * Generates short recording metadata locally with Granite 4.0 350M.
 *
 * LiteRT-LM currently accepts a filesystem path, while Android assets expose
 * an offset within the APK. The verified asset is therefore copied once to a
 * versioned app-private model path and reused across recordings.
 */
class GraniteRecordingMetadataGenerator private constructor(
    private val context: Context,
) {
    fun generate(transcript: String): RecordingMetadata? {
        val excerpt = buildMetadataTranscriptExcerpt(
            WhisperTimestampTokens.withoutTimestampMarkers(transcript),
        )
        if (excerpt.isBlank()) return null

        val model = materializeModel()
        Engine(
            EngineConfig(
                modelPath = model.absolutePath,
                backend = Backend.CPU(threadCount = METADATA_THREAD_COUNT),
                maxNumTokens = MODEL_CONTEXT_TOKENS,
                cacheDir = File(context.cacheDir, "litertlm-granite").path,
            ),
        ).use { engine ->
            engine.initialize()
            val config = ConversationConfig(
                systemInstruction = Contents.of(
                    "You create faithful, concise metadata for voice recordings. " +
                        "Never invent facts that are absent from the transcript.",
                ),
                samplerConfig = SamplerConfig(
                    topK = 1,
                    topP = 1.0,
                    temperature = 0.0,
                    seed = 1,
                ),
            )
            engine.createConversation(config).use { conversation ->
                val response = conversation.sendMessage(metadataPrompt(excerpt))
                val text = response.contents.contents
                    .filterIsInstance<Content.Text>()
                    .joinToString(separator = "") { it.text }
                return parseRecordingMetadata(text)
            }
        }
    }

    private fun materializeModel(): File {
        val directory = File(context.noBackupFilesDir, "metadata-models")
        check(directory.isDirectory || directory.mkdirs()) {
            "The app-private metadata model directory could not be created."
        }
        val destination = File(directory, MATERIALIZED_MODEL_NAME)
        if (destination.isFile && destination.length() == MODEL_SIZE_BYTES) {
            return destination
        }
        if (destination.exists()) {
            check(destination.delete()) {
                "An invalid generated metadata model copy could not be replaced."
            }
        }
        check(directory.usableSpace >= MODEL_SIZE_BYTES + MINIMUM_FREE_SPACE_AFTER_COPY) {
            "At least 512 MB of free app storage is required to prepare the title model."
        }

        val temporary = File(directory, ".${destination.name}.${UUID.randomUUID()}.part")
        try {
            context.assets.open(MODEL_ASSET).use { input ->
                FileOutputStream(temporary).use { output ->
                    input.copyTo(output)
                    output.fd.sync()
                }
            }
            check(temporary.length() == MODEL_SIZE_BYTES) {
                "The bundled Granite model copy is incomplete."
            }
            check(temporary.renameTo(destination)) {
                "The prepared Granite model could not be activated."
            }
        } finally {
            if (temporary.exists()) temporary.delete()
        }
        return destination
    }

    companion object {
        const val MODEL_ID = "granite-4.0-350m-q8-ekv1280"
        const val MODEL_ASSET = "granite/granite-4.0-350m_q8_ekv1280.litertlm"
        const val MODEL_SIZE_BYTES = 468_209_584L

        private const val MATERIALIZED_MODEL_NAME =
            "granite-4.0-350m_q8_ekv1280-c8e9a294.litertlm"
        private const val MODEL_CONTEXT_TOKENS = 1_280
        private const val METADATA_THREAD_COUNT = 3
        private const val MINIMUM_FREE_SPACE_AFTER_COPY = 64L * 1024L * 1024L

        fun create(context: Context): GraniteRecordingMetadataGenerator =
            GraniteRecordingMetadataGenerator(context.applicationContext)
    }
}

internal fun metadataPrompt(transcriptExcerpt: String): String =
    """
    Create metadata in the same language as the transcript.
    Title: 3 to 8 words, specific, no ending punctuation.
    Summary: exactly one short sentence, at most 24 words.
    Return exactly two lines:
    TITLE: <title>
    SUMMARY: <summary>

    TRANSCRIPT:
    $transcriptExcerpt
    $METADATA_ANSWER_MARKER
    """.trimIndent()

internal fun buildMetadataTranscriptExcerpt(
    transcript: String,
    maximumCharacters: Int = 3_000,
): String {
    require(maximumCharacters >= 300)
    val normalized = transcript.replace(Regex("\\s+"), " ").trim()
    if (normalized.length <= maximumCharacters) return normalized

    val separator = "\n[…]\n"
    val available = maximumCharacters - separator.length * 2
    val firstLength = available * 2 / 5
    val middleLength = available * 3 / 10
    val lastLength = available - firstLength - middleLength
    val middleStart = (normalized.length - middleLength) / 2
    return listOf(
        normalized.take(firstLength).trimAtWordEnd(),
        normalized.substring(middleStart, middleStart + middleLength)
            .trimAtWordStart()
            .trimAtWordEnd(),
        normalized.takeLast(lastLength).trimAtWordStart(),
    ).joinToString(separator)
}

internal fun parseRecordingMetadata(raw: String): RecordingMetadata? {
    val answer = raw.substringAfterLast(
        delimiter = METADATA_ANSWER_MARKER,
        missingDelimiterValue = raw,
    )
    val cleaned = answer
        .replace("```json", "", ignoreCase = true)
        .replace("```", "")
        .trim()
    val titleMatches = METADATA_TITLE.findAll(cleaned).toList()
    val summaryMatches = METADATA_SUMMARY.findAll(cleaned).toList()
    val fallbackLines = cleaned.lineSequence()
        .map(String::trim)
        .map { it.trimStart('-', '*', '•').trim() }
        .filter(String::isNotBlank)
        .toList()
    val hasLabeledOutput = titleMatches.isNotEmpty() || summaryMatches.isNotEmpty()
    val title = titleMatches.lastOrNull()?.groupValues?.get(1)
        ?: fallbackLines.getOrNull(0).takeUnless { hasLabeledOutput }
    val summary = summaryMatches.lastOrNull()?.groupValues?.get(1)
        ?: fallbackLines.getOrNull(1).takeUnless { hasLabeledOutput }
    val normalizedTitle = cleanMetadataValue(title)
        ?.takeUnless(::isMetadataPromptInstruction)
        ?.boundedAtWord(80)
    val normalizedSummary = cleanMetadataValue(summary)
        ?.takeUnless(::isMetadataPromptInstruction)
        ?.boundedAtWord(220)
    if (normalizedTitle.isNullOrBlank() || normalizedSummary.isNullOrBlank()) return null
    return RecordingMetadata(
        title = normalizedTitle.trimEnd('.', '!', '?'),
        summary = normalizedSummary,
    )
}

private fun cleanMetadataValue(value: String?): String? = value
    ?.trim()
    ?.trimEnd(',')
    ?.trim()
    ?.removeSurrounding("\"")
    ?.removeSurrounding("'")
    ?.replace(Regex("\\s+"), " ")
    ?.trim()
    ?.takeIf(String::isNotBlank)

private fun isMetadataPromptInstruction(value: String): Boolean {
    val normalized = value.lowercase()
    return normalized == "<title>" ||
        normalized == "<summary>" ||
        "3 to 8 words" in normalized ||
        "no ending punctuation" in normalized ||
        "exactly one short sentence" in normalized ||
        "at most 24 words" in normalized
}

private fun String.boundedAtWord(maximumCharacters: Int): String {
    if (length <= maximumCharacters) return this
    return take(maximumCharacters + 1).trimAtWordEnd().take(maximumCharacters).trim()
}

private fun String.trimAtWordStart(): String =
    substringAfter(' ', missingDelimiterValue = this).trim()

private fun String.trimAtWordEnd(): String =
    substringBeforeLast(' ', missingDelimiterValue = this).trim()

private val METADATA_TITLE = Regex(
    pattern = """(?im)^\s*["']?title["']?\s*[:=-]\s*(.+?)\s*,?\s*$""",
)
private val METADATA_SUMMARY = Regex(
    pattern = """(?im)^\s*["']?summary["']?\s*[:=-]\s*(.+?)\s*,?\s*$""",
)
private const val METADATA_ANSWER_MARKER = "METADATA ANSWER:"
