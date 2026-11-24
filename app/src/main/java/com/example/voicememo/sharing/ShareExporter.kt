package com.example.voicememo.sharing

import android.content.ClipData
import android.content.Context
import android.content.Intent
import android.net.Uri
import androidx.core.content.FileProvider
import java.io.File

enum class ShareContent {
    Audio,
    Transcript,
    Both,
}

data class ShareableMemo(
    val id: String,
    val title: String,
    val audioPath: String,
    val effectiveTranscript: String?,
    val transcriptSegments: List<String> = emptyList(),
    /** Timed lines; when present the transcript is shared as a CSV table instead of plain text. */
    val transcriptRows: List<ShareTranscriptRow> = emptyList(),
)

/** One transcript line for the CSV export. [speaker] is "Me", "Other", "Both" or null when unknown. */
data class ShareTranscriptRow(
    val startMillis: Long,
    val endMillis: Long,
    val speaker: String?,
    val text: String,
)

class ShareExporter(
    context: Context,
) {
    private val applicationContext = context.applicationContext
    private val authority = "${context.packageName}.files"
    private val shareDirectory = File(context.cacheDir, "shares")

    fun createChooser(
        memos: List<ShareableMemo>,
        content: ShareContent,
    ): Intent {
        require(memos.isNotEmpty()) { "Select at least one memo to share." }
        cleanupStaleAttachments()
        shareDirectory.mkdirs()

        val attachments = buildList {
            memos.forEach { memo ->
                if (content == ShareContent.Audio || content == ShareContent.Both) {
                    val audio = File(memo.audioPath)
                    if (audio.isFile) add(FileProvider.getUriForFile(applicationContext, authority, audio))
                }
                if (content == ShareContent.Transcript || content == ShareContent.Both) {
                    val table = memo.transcriptCsvForShare()
                    val transcript = table ?: memo.transcriptForShare()
                    if (!transcript.isNullOrBlank()) {
                        val textFile = File(
                            shareDirectory,
                            "${safeShareFileName(memo.title)}-${memo.id.take(8)}." +
                                if (table != null) "csv" else "txt",
                        )
                        textFile.writeText(transcript, Charsets.UTF_8)
                        add(FileProvider.getUriForFile(applicationContext, authority, textFile))
                    }
                }
            }
        }
        require(attachments.isNotEmpty()) { "The selected content is not available to share." }

        val mimeType = when (content) {
            ShareContent.Audio -> "audio/wav"
            ShareContent.Transcript -> when {
                memos.all { it.transcriptCsvForShare() != null } -> "text/csv"
                memos.none { it.transcriptCsvForShare() != null } -> "text/plain"
                else -> "text/*"
            }
            ShareContent.Both -> "*/*"
        }
        val send = if (attachments.size == 1) {
            Intent(Intent.ACTION_SEND)
                .setType(mimeType)
                .putExtra(Intent.EXTRA_STREAM, attachments.single())
        } else {
            Intent(Intent.ACTION_SEND_MULTIPLE)
                .setType(mimeType)
                .putParcelableArrayListExtra(Intent.EXTRA_STREAM, ArrayList(attachments))
        }.apply {
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            putExtra(
                Intent.EXTRA_SUBJECT,
                if (memos.size == 1) memos.single().title else "${memos.size} voice memos",
            )
            clipData = ClipData.newUri(
                applicationContext.contentResolver,
                "Voice Memo attachments",
                attachments.first(),
            ).also { clip ->
                attachments.drop(1).forEach { clip.addItem(ClipData.Item(it)) }
            }
        }
        return Intent.createChooser(send, "Share voice memo")
    }

    fun cleanupStaleAttachments(nowMillis: Long = System.currentTimeMillis()) {
        shareDirectory.listFiles()
            ?.filter { nowMillis - it.lastModified() > STALE_AFTER_MILLIS }
            ?.forEach(File::delete)
    }

    private companion object {
        const val STALE_AFTER_MILLIS = 24L * 60L * 60L * 1_000L
    }
}

internal fun ShareableMemo.transcriptForShare(): String? {
    val segments = transcriptSegments.map(String::trim).filter(String::isNotBlank)
    return if (segments.isNotEmpty()) {
        segments.joinToString(separator = "\n\n")
    } else {
        effectiveTranscript
    }
}

/**
 * The timed transcript as a CSV table (`start,end,speaker,text`), or null when
 * the memo has no timed lines. UTF-8 with a byte-order mark so spreadsheet apps
 * read umlauts correctly; fields are quoted as in RFC 4180.
 */
internal fun ShareableMemo.transcriptCsvForShare(): String? {
    val rows = transcriptRows.filter { it.text.isNotBlank() }
    if (rows.isEmpty()) return null
    return buildString {
        append('\uFEFF')
        append("start,end,speaker,text\r\n")
        for (row in rows) {
            append(csvTime(row.startMillis)).append(',')
            append(csvTime(row.endMillis)).append(',')
            append(csvField(row.speaker.orEmpty())).append(',')
            append(csvField(row.text.trim())).append("\r\n")
        }
    }
}

private fun csvTime(millis: Long): String {
    val seconds = millis.coerceAtLeast(0L) / 1_000L
    return "%02d:%02d:%02d".format(seconds / 3_600L, seconds / 60L % 60L, seconds % 60L)
}

private fun csvField(value: String): String =
    if (value.any { it == ',' || it == '"' || it == '\n' || it == '\r' }) {
        "\"" + value.replace("\"", "\"\"") + "\""
    } else {
        value
    }

internal fun safeShareFileName(title: String): String {
    val cleaned = title
        .trim()
        .replace(Regex("[^\\p{L}\\p{N}._ -]+"), "_")
        .trim('.', ' ', '_')
        .take(80)
    return cleaned.ifBlank { "Voice memo" }
}
