package com.example.voicememo.data

import androidx.room.Embedded
import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey
import java.io.StringReader
import java.io.StringWriter
import java.util.Properties

/** Saved result; the selected version is mirrored into transcripts for search/sharing. */
@Entity(
    tableName = "transcript_versions",
    foreignKeys = [ForeignKey(
        entity = MemoRecord::class, parentColumns = ["id"],
        childColumns = ["memoId"], onDelete = ForeignKey.CASCADE,
    )],
    indices = [Index("memoId")],
)
data class TranscriptVersion(
    @PrimaryKey val id: String,
    val memoId: String,
    val jobId: String?,
    val modelId: String?,
    val selected: Boolean,
    @Embedded(prefix = "saved_") val transcript: TranscriptRecord,
    val segmentEdits: String = "",
)

internal object TranscriptEditSnapshot {
    fun encode(edits: List<TranscriptSegmentEdit>): String = Properties().apply {
        setProperty("count", edits.size.toString())
        edits.forEachIndexed { index, edit ->
            setProperty("$index.id", edit.segmentId)
            setProperty("$index.text", edit.text)
            setProperty("$index.time", edit.editedAtMillis.toString())
        }
    }.let { properties -> StringWriter().also { properties.store(it, null) }.toString() }

    fun decode(memoId: String, encoded: String): List<TranscriptSegmentEdit> {
        if (encoded.isEmpty()) return emptyList()
        val properties = Properties().apply { load(StringReader(encoded)) }
        return List(properties.getProperty("count").toInt()) { index ->
            TranscriptSegmentEdit(
                memoId, properties.getProperty("$index.id"),
                properties.getProperty("$index.text"), properties.getProperty("$index.time").toLong(),
            )
        }
    }
}

/** Portable transcript history; timing chunks are derived and aren't backed up. */
internal object TranscriptVersionArchive {
    fun encode(versions: List<TranscriptVersion>): ByteArray = Properties().apply {
        setProperty("count", versions.size.toString())
        versions.forEachIndexed { index, version ->
            val prefix = "$index."
            setProperty(prefix + "selected", version.selected.toString())
            version.modelId?.let { setProperty(prefix + "model", it) }
            val transcript = version.transcript
            transcript.generatedText?.let { setProperty(prefix + "generated", it) }
            transcript.editedText?.let { setProperty(prefix + "edited", it) }
            transcript.detectedLanguage?.let { setProperty(prefix + "language", it) }
            transcript.generatedAtMillis?.let { setProperty(prefix + "generatedAt", it.toString()) }
            transcript.editedAtMillis?.let { setProperty(prefix + "editedAt", it.toString()) }
            setProperty(prefix + "revision", transcript.generatedRevision.toString())
        }
    }.let { properties ->
        java.io.ByteArrayOutputStream().also { properties.store(it, null) }.toByteArray()
    }

    fun decode(memoId: String, bytes: ByteArray): List<TranscriptVersion> {
        val properties = Properties().apply { load(bytes.inputStream()) }
        val count = properties.getProperty("count").toInt()
        require(count in 0..10_000) { "Invalid transcript history size." }
        val versions = List(count) { index ->
            val prefix = "$index."
            val selected = properties.getProperty(prefix + "selected").toBooleanStrict()
            TranscriptVersion(
                id = java.util.UUID.randomUUID().toString(), memoId = memoId,
                jobId = null, modelId = properties.getProperty(prefix + "model"),
                selected = selected,
                transcript = TranscriptRecord(
                    memoId = memoId,
                    generatedText = properties.getProperty(prefix + "generated"),
                    editedText = properties.getProperty(prefix + "edited"),
                    detectedLanguage = properties.getProperty(prefix + "language"),
                    generatedAtMillis = properties.getProperty(prefix + "generatedAt")?.toLong(),
                    editedAtMillis = properties.getProperty(prefix + "editedAt")?.toLong(),
                    generatedRevision = properties.getProperty(prefix + "revision").toLong(),
                ),
            )
        }
        require(versions.isEmpty() || versions.count { it.selected } == 1) {
            "Transcript history must have one selected version."
        }
        return versions
    }
}
