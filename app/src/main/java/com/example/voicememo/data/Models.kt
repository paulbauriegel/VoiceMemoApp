package com.example.voicememo.data

import androidx.room.Embedded
import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Fts4
import androidx.room.FtsOptions
import androidx.room.Index
import androidx.room.Relation

enum class WaveformState {
    PENDING,
    READY,
    FAILED,
}

enum class TranscriptionJobStatus {
    QUEUED,
    RUNNING,
    PAUSED_FOR_RECORDING,
    SUCCEEDED,
    FAILED,
    CANCEL_REQUESTED,
    CANCELLED,
}

enum class TranscriptionPhase {
    QUEUED,
    LOADING_MODEL,
    PREPARING_AUDIO,
    ENCODING,
    DECODING,
    MERGING,
    SAVING,
    GENERATING_METADATA,
    PAUSED,
    COMPLETE,
    FAILED,
}

enum class TranscriptChunkStatus {
    PENDING,
    RUNNING,
    COMPLETED,
    FAILED,
}

enum class ProjectIcon {
    WORK,
    LIGHTBULB,
    PERSON,
    HOME,
    SCHOOL,
    MUSIC,
}

@Entity(
    tableName = "projects",
    indices = [
        Index(value = ["updatedAtMillis"]),
    ],
)
data class ProjectRecord(
    @androidx.room.PrimaryKey
    val id: String,
    val name: String,
    val createdAtMillis: Long,
    val updatedAtMillis: Long,
    val icon: ProjectIcon = ProjectIcon.WORK,
    val isArchived: Boolean = false,
)

@Entity(
    tableName = "memos",
    foreignKeys = [
        ForeignKey(
            entity = ProjectRecord::class,
            parentColumns = ["id"],
            childColumns = ["projectId"],
            onDelete = ForeignKey.SET_NULL,
        ),
    ],
    indices = [
        Index(value = ["createdAtMillis"]),
        Index(value = ["updatedAtMillis"]),
        Index(value = ["pendingDeletion"]),
        Index(value = ["projectId"]),
    ],
)
data class MemoRecord(
    @androidx.room.PrimaryKey
    val id: String,
    val title: String,
    val createdAtMillis: Long,
    val updatedAtMillis: Long,
    val durationMillis: Long,
    val audioPath: String,
    val waveformPath: String?,
    val waveformState: WaveformState,
    /** Null keeps the recording in Recents without assigning it to a project. */
    val projectId: String? = null,
    /**
     * Null means Auto. A non-null value is a Whisper language code such as
     * "en" or "de" and is applied to the next transcription/retry.
     */
    val languageOverride: String?,
    /**
     * A short-lived tombstone used while work and private files are removed.
     * Tombstoned rows are excluded from every user-facing query.
     */
    val pendingDeletion: Boolean = false,
    /** User-controlled favorite state used by the Starred recordings filter. */
    val isStarred: Boolean = false,
    /**
     * Legacy and imported titles are treated as user-owned. New recordings
     * start false so a generated title may replace their timestamp placeholder.
     */
    val titleEditedByUser: Boolean = true,
    /** One-line description generated locally after transcription. */
    val generatedSummary: String? = null,
    val metadataGeneratedAtMillis: Long? = null,
    val metadataModelId: String? = null,
)

@Entity(
    tableName = "transcripts",
    foreignKeys = [
        ForeignKey(
            entity = MemoRecord::class,
            parentColumns = ["id"],
            childColumns = ["memoId"],
            onDelete = ForeignKey.CASCADE,
        ),
    ],
)
data class TranscriptRecord(
    @androidx.room.PrimaryKey
    val memoId: String,
    val generatedText: String?,
    /**
     * Null means "show the generated copy". An empty string is an intentional
     * user edit and must remain distinct from null.
     */
    val editedText: String?,
    val detectedLanguage: String?,
    val generatedAtMillis: Long?,
    val editedAtMillis: Long?,
    val generatedRevision: Long = 0,
) {
    val effectiveText: String
        get() = editedText ?: generatedText.orEmpty()
}

/**
 * A user correction for one timed transcript segment. [segmentId] is the
 * presentation id derived from the chunk job, so edits belong to the
 * transcription they were made against. The joined effective text is mirrored
 * into [TranscriptRecord.editedText] so search and sharing keep working.
 */
@Entity(
    tableName = "transcript_segment_edits",
    primaryKeys = ["memoId", "segmentId"],
    foreignKeys = [
        ForeignKey(
            entity = MemoRecord::class,
            parentColumns = ["id"],
            childColumns = ["memoId"],
            onDelete = ForeignKey.CASCADE,
        ),
    ],
)
data class TranscriptSegmentEdit(
    val memoId: String,
    val segmentId: String,
    val text: String,
    val editedAtMillis: Long,
)

@Entity(
    tableName = "transcription_jobs",
    foreignKeys = [
        ForeignKey(
            entity = MemoRecord::class,
            parentColumns = ["id"],
            childColumns = ["memoId"],
            onDelete = ForeignKey.CASCADE,
        ),
    ],
    indices = [
        Index(value = ["memoId"]),
        Index(value = ["status", "createdAtMillis"]),
        Index(value = ["workId"], unique = true),
    ],
)
data class TranscriptionJob(
    @androidx.room.PrimaryKey
    val id: String,
    val memoId: String,
    /**
     * The UUID assigned by WorkManager, stored as text so a job can be
     * cancelled after process death. It is nullable until work is enqueued.
     */
    val workId: String?,
    val status: TranscriptionJobStatus,
    val phase: TranscriptionPhase,
    /** Null requests automatic language detection. */
    val requestedLanguage: String?,
    val detectedLanguage: String?,
    val totalChunks: Int,
    val completedChunks: Int,
    val currentChunkIndex: Int?,
    val currentToken: Int,
    val currentTokenLimit: Int,
    val errorCode: String?,
    val errorMessage: String?,
    val warning: String?,
    /**
     * Identifies the model, frontend, decoder, and runtime that produced
     * resumable chunk rows. A null legacy value is never trusted for resume.
     */
    val inferenceFingerprint: String? = null,
    val createdAtMillis: Long,
    val updatedAtMillis: Long,
    /** Stable model ID captured when the job is queued. */
    val transcriptionModelId: String = "whisper-large-v3-turbo",
    /** Wall time spent in the segment transcription pass, excluding model loading. */
    val transcriptionElapsedMillis: Long? = null,
    /** Highest Android thermal status sampled during the transcription pass. */
    val maximumThermalStatus: Int? = null,
)

@Entity(
    tableName = "transcript_chunks",
    primaryKeys = ["jobId", "chunkIndex"],
    foreignKeys = [
        ForeignKey(
            entity = TranscriptionJob::class,
            parentColumns = ["id"],
            childColumns = ["jobId"],
            onDelete = ForeignKey.CASCADE,
        ),
        ForeignKey(
            entity = MemoRecord::class,
            parentColumns = ["id"],
            childColumns = ["memoId"],
            onDelete = ForeignKey.CASCADE,
        ),
    ],
    indices = [
        Index(value = ["jobId"]),
        Index(value = ["memoId"]),
        Index(value = ["status"]),
    ],
)
data class TranscriptChunk(
    val jobId: String,
    val chunkIndex: Int,
    val memoId: String,
    val windowStartMillis: Long,
    val windowEndMillis: Long,
    val status: TranscriptChunkStatus,
    val rawText: String,
    val tokenIds: List<Int>,
    val detectedLanguage: String?,
    val truncated: Boolean,
    val warning: String?,
    val completedAtMillis: Long?,
    /** Model inference time reported for this segment. */
    val inferenceMillis: Long? = null,
)

/**
 * Standalone FTS4 index. It is updated in the same Room transaction as the
 * source row so the indexed transcript always follows the effective
 * (edited-preferred) text.
 */
@Fts4(
    tokenizer = FtsOptions.TOKENIZER_UNICODE61,
    tokenizerArgs = ["remove_diacritics=2"],
    prefix = [2, 3, 4],
)
@Entity(tableName = "memo_search_fts")
data class MemoSearchFts(
    val memoId: String,
    val title: String,
    val effectiveTranscript: String,
)

data class MemoAggregate(
    @Embedded
    val memo: MemoRecord,
    @Relation(
        parentColumn = "id",
        entityColumn = "memoId",
    )
    val transcripts: List<TranscriptRecord>,
    @Relation(
        parentColumn = "id",
        entityColumn = "memoId",
    )
    val transcriptionJobs: List<TranscriptionJob>,
)

data class NewMemo(
    val id: String? = null,
    val title: String,
    val createdAtMillis: Long,
    val durationMillis: Long,
    val audioPath: String,
    val waveformPath: String?,
    val waveformState: WaveformState,
    val titleEditedByUser: Boolean = true,
)

enum class TranscriptSource {
    NONE,
    GENERATED,
    EDITED,
}

data class MemoSummary(
    val id: String,
    val title: String,
    val createdAtMillis: Long,
    val updatedAtMillis: Long,
    val durationMillis: Long,
    val audioPath: String,
    val waveformPath: String?,
    val waveformState: WaveformState,
    val languageOverride: String?,
    val projectId: String? = null,
    val effectiveTranscript: String,
    val transcriptSource: TranscriptSource,
    val detectedLanguage: String?,
    val latestJob: TranscriptionJob?,
    val isStarred: Boolean = false,
    val generatedSummary: String? = null,
)

data class ProjectSummary(
    val id: String,
    val name: String,
    val createdAtMillis: Long,
    val updatedAtMillis: Long,
    val recordingCount: Int,
    val icon: ProjectIcon,
    val isArchived: Boolean,
)

data class MemoMonthGroup(
    val year: Int,
    /** Calendar-style month index in the range 1..12. */
    val month: Int,
    val monthStartMillis: Long,
    val memos: List<MemoSummary>,
)

data class MemoDetail(
    val memo: MemoRecord,
    val transcript: TranscriptRecord?,
    val latestJob: TranscriptionJob?,
    val chunks: List<TranscriptChunk>,
    val segmentEdits: List<TranscriptSegmentEdit> = emptyList(),
) {
    val effectiveTranscript: String
        get() = transcript?.effectiveText.orEmpty()

    val transcriptSource: TranscriptSource
        get() = when {
            transcript?.editedText != null -> TranscriptSource.EDITED
            transcript?.generatedText != null -> TranscriptSource.GENERATED
            else -> TranscriptSource.NONE
        }
}

internal fun MemoAggregate.toSummary(): MemoSummary {
    val transcript = transcripts.singleOrNull()
    val latestJob = transcriptionJobs.maxByOrNull { it.updatedAtMillis }
    val source = when {
        transcript?.editedText != null -> TranscriptSource.EDITED
        transcript?.generatedText != null -> TranscriptSource.GENERATED
        else -> TranscriptSource.NONE
    }
    return MemoSummary(
        id = memo.id,
        title = memo.title,
        createdAtMillis = memo.createdAtMillis,
        updatedAtMillis = memo.updatedAtMillis,
        durationMillis = memo.durationMillis,
        audioPath = memo.audioPath,
        waveformPath = memo.waveformPath,
        waveformState = memo.waveformState,
        languageOverride = memo.languageOverride,
        projectId = memo.projectId,
        effectiveTranscript = transcript?.effectiveText.orEmpty(),
        transcriptSource = source,
        detectedLanguage = transcript?.detectedLanguage,
        latestJob = latestJob,
        isStarred = memo.isStarred,
        generatedSummary = memo.generatedSummary,
    )
}
