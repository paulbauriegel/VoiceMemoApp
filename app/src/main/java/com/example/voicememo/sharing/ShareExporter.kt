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
                    val transcript = memo.effectiveTranscript
                    if (!transcript.isNullOrBlank()) {
                        val textFile = File(
                            shareDirectory,
                            "${safeShareFileName(memo.title)}-${memo.id.take(8)}.txt",
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
            ShareContent.Transcript -> "text/plain"
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

internal fun safeShareFileName(title: String): String {
    val cleaned = title
        .trim()
        .replace(Regex("[^\\p{L}\\p{N}._ -]+"), "_")
        .trim('.', ' ', '_')
        .take(80)
    return cleaned.ifBlank { "Voice memo" }
}
