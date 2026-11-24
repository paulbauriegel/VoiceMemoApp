package com.example.voicememo.data

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Transaction
import androidx.room.Update
import androidx.room.Upsert
import kotlinx.coroutines.flow.Flow

@Dao
interface VoiceMemoDao {
    @Query(
        """
        SELECT projects.id, projects.name, projects.createdAtMillis, projects.updatedAtMillis,
               projects.icon, projects.isArchived,
               COUNT(memos.id) AS recordingCount
        FROM projects
        LEFT JOIN memos
          ON memos.projectId = projects.id
         AND memos.pendingDeletion = 0
        GROUP BY projects.id
        ORDER BY projects.updatedAtMillis DESC, projects.name COLLATE NOCASE ASC
        """,
    )
    fun observeProjects(): Flow<List<ProjectSummary>>

    @Query("SELECT * FROM projects WHERE id = :projectId LIMIT 1")
    suspend fun getProject(projectId: String): ProjectRecord?

    @Query("SELECT * FROM projects")
    suspend fun getAllProjects(): List<ProjectRecord>

    @Insert(onConflict = OnConflictStrategy.ABORT)
    suspend fun insertProject(project: ProjectRecord)

    @Insert(onConflict = OnConflictStrategy.IGNORE)
    suspend fun insertProjectIfAbsent(project: ProjectRecord)

    @Update
    suspend fun updateProject(project: ProjectRecord)

    @Query("DELETE FROM projects WHERE id = :projectId")
    suspend fun deleteProject(projectId: String)

    @Transaction
    @Query(
        """
        SELECT * FROM memos
        WHERE pendingDeletion = 0
        ORDER BY createdAtMillis DESC
        """,
    )
    fun observeAllMemos(): Flow<List<MemoAggregate>>

    @Transaction
    @Query(
        """
        SELECT * FROM memos
        WHERE pendingDeletion = 0
        ORDER BY createdAtMillis DESC
        """,
    )
    suspend fun getAllMemoAggregates(): List<MemoAggregate>

    @Transaction
    @Query(
        """
        SELECT * FROM memos
        WHERE pendingDeletion = 0
          AND id IN (
              SELECT memoId
              FROM memo_search_fts
              WHERE memo_search_fts MATCH :matchQuery
          )
        ORDER BY createdAtMillis DESC
        """,
    )
    fun observeSearchResults(matchQuery: String): Flow<List<MemoAggregate>>

    @Transaction
    @Query("SELECT * FROM memos WHERE id = :memoId AND pendingDeletion = 0 LIMIT 1")
    fun observeMemo(memoId: String): Flow<MemoAggregate?>

    @Transaction
    @Query("SELECT * FROM memos WHERE id = :memoId LIMIT 1")
    suspend fun getMemoAggregate(memoId: String): MemoAggregate?

    @Query("SELECT * FROM memos WHERE id = :memoId LIMIT 1")
    suspend fun getMemo(memoId: String): MemoRecord?

    @Query("SELECT * FROM memos WHERE id IN (:memoIds)")
    suspend fun getMemos(memoIds: Collection<String>): List<MemoRecord>

    @Query(
        """
        SELECT id FROM memos
        WHERE pendingDeletion = 0
        ORDER BY createdAtMillis DESC
        """,
    )
    suspend fun getAllMemoIds(): List<String>

    @Query("SELECT id FROM memos")
    suspend fun getEveryMemoIds(): List<String>

    @Query(
        """
        SELECT id FROM memos
        WHERE pendingDeletion = 0
          AND id IN (
              SELECT memoId
              FROM memo_search_fts
              WHERE memo_search_fts MATCH :matchQuery
          )
        ORDER BY createdAtMillis DESC
        """,
    )
    suspend fun getSearchResultIds(matchQuery: String): List<String>

    @Insert(onConflict = OnConflictStrategy.ABORT)
    suspend fun insertMemo(memo: MemoRecord)

    @Update
    suspend fun updateMemo(memo: MemoRecord)

    @Query(
        """
        UPDATE memos
        SET projectId = :projectId, updatedAtMillis = :updatedAtMillis
        WHERE id IN (:memoIds) AND pendingDeletion = 0
        """,
    )
    suspend fun assignMemosToProject(
        memoIds: Collection<String>,
        projectId: String?,
        updatedAtMillis: Long,
    )

    @Query(
        """
        UPDATE memos
        SET isStarred = :starred
        WHERE id = :memoId AND pendingDeletion = 0
        """,
    )
    suspend fun setMemoStarred(memoId: String, starred: Boolean)

    @Query("UPDATE memos SET pendingDeletion = :pending, updatedAtMillis = :updatedAtMillis WHERE id IN (:memoIds)")
    suspend fun markPendingDeletion(
        memoIds: Collection<String>,
        pending: Boolean,
        updatedAtMillis: Long,
    )

    @Query("DELETE FROM memos WHERE id IN (:memoIds)")
    suspend fun deleteMemos(memoIds: Collection<String>)

    @Query("SELECT * FROM transcript_versions WHERE memoId = :memoId ORDER BY saved_generatedAtMillis DESC, id")
    suspend fun getTranscriptVersions(memoId: String): List<TranscriptVersion>

    @Query("SELECT * FROM transcript_versions WHERE memoId = :memoId ORDER BY saved_generatedAtMillis DESC, id")
    fun observeTranscriptVersions(memoId: String): Flow<List<TranscriptVersion>>

    @Upsert
    suspend fun upsertTranscriptVersion(version: TranscriptVersion)

    @Query("SELECT * FROM transcript_segment_edits WHERE memoId = :memoId")
    suspend fun getSegmentEdits(memoId: String): List<TranscriptSegmentEdit>

    @Query("SELECT * FROM transcripts WHERE memoId = :memoId LIMIT 1")
    suspend fun getTranscript(memoId: String): TranscriptRecord?

    @Upsert
    suspend fun upsertTranscript(transcript: TranscriptRecord)

    @Query("SELECT * FROM transcript_segment_edits WHERE memoId = :memoId")
    fun observeSegmentEdits(memoId: String): Flow<List<TranscriptSegmentEdit>>

    @Query("SELECT COUNT(*) FROM transcript_segment_edits WHERE memoId = :memoId")
    suspend fun countSegmentEdits(memoId: String): Int

    @Upsert
    suspend fun upsertSegmentEdit(edit: TranscriptSegmentEdit)

    @Query(
        "DELETE FROM transcript_segment_edits WHERE memoId = :memoId AND segmentId = :segmentId",
    )
    suspend fun deleteSegmentEdit(memoId: String, segmentId: String)

    @Query("DELETE FROM transcript_segment_edits WHERE memoId = :memoId")
    suspend fun deleteSegmentEdits(memoId: String)

    @Query("SELECT * FROM transcript_segment_embeddings WHERE memoId = :memoId")
    suspend fun getSegmentEmbeddings(memoId: String): List<TranscriptSegmentEmbedding>

    @Upsert
    suspend fun upsertSegmentEmbeddings(embeddings: List<TranscriptSegmentEmbedding>)

    @Query(
        "DELETE FROM transcript_segment_embeddings " +
            "WHERE memoId = :memoId AND segmentId IN (:segmentIds)",
    )
    suspend fun deleteSegmentEmbeddings(memoId: String, segmentIds: Collection<String>)

    @Query("DELETE FROM memo_search_fts WHERE memoId = :memoId")
    suspend fun deleteSearchEntry(memoId: String)

    @Insert
    suspend fun insertSearchEntry(searchEntry: MemoSearchFts)

    @Query("DELETE FROM memo_search_fts WHERE memoId IN (:memoIds)")
    suspend fun deleteSearchEntries(memoIds: Collection<String>)

    @Upsert
    suspend fun upsertTranscriptionJob(job: TranscriptionJob)

    @Query("SELECT * FROM transcription_jobs WHERE id = :jobId LIMIT 1")
    suspend fun getTranscriptionJob(jobId: String): TranscriptionJob?

    @Query(
        """
        SELECT * FROM transcription_jobs
        WHERE status IN ('QUEUED', 'RUNNING', 'PAUSED_FOR_RECORDING')
        ORDER BY createdAtMillis ASC
        LIMIT 1
        """,
    )
    suspend fun getOldestRunnableJob(): TranscriptionJob?

    @Query(
        """
        SELECT * FROM transcription_jobs
        WHERE memoId = :memoId
        ORDER BY updatedAtMillis DESC
        LIMIT 1
        """,
    )
    suspend fun getLatestJobForMemo(memoId: String): TranscriptionJob?

    @Query(
        """
        UPDATE transcription_jobs
        SET workId = :workId, updatedAtMillis = :updatedAtMillis
        WHERE id = :jobId
        """,
    )
    suspend fun attachWorkId(jobId: String, workId: String, updatedAtMillis: Long)

    @Upsert
    suspend fun upsertTranscriptChunk(chunk: TranscriptChunk)

    @Query(
        """
        SELECT * FROM transcript_chunks
        WHERE jobId = :jobId
        ORDER BY chunkIndex ASC
        """,
    )
    fun observeTranscriptChunks(jobId: String): Flow<List<TranscriptChunk>>

    @Query(
        """
        SELECT * FROM transcript_chunks
        WHERE jobId = :jobId
          AND status = 'COMPLETED'
        ORDER BY chunkIndex ASC
        """,
    )
    suspend fun getCompletedTranscriptChunks(jobId: String): List<TranscriptChunk>

    @Query("DELETE FROM transcript_chunks WHERE jobId = :jobId")
    suspend fun deleteTranscriptChunks(jobId: String)

    @Query("SELECT * FROM speaker_persons ORDER BY isSelf DESC, createdAtMillis")
    suspend fun getSpeakerPersons(): List<SpeakerPerson>

    @Upsert
    suspend fun upsertSpeakerPerson(person: SpeakerPerson)

    /** Voice samples go with the person (foreign key cascade). */
    @Query("DELETE FROM speaker_persons WHERE id = :personId")
    suspend fun deleteSpeakerPerson(personId: String)

    @Query("SELECT * FROM speaker_voice_samples ORDER BY createdAtMillis")
    suspend fun getSpeakerVoiceSamples(): List<SpeakerVoiceSample>

    @Upsert
    suspend fun upsertSpeakerVoiceSample(sample: SpeakerVoiceSample)

    @Query("DELETE FROM speaker_voice_samples WHERE id = :sampleId")
    suspend fun deleteSpeakerVoiceSample(sampleId: String)

    @Query(
        "SELECT * FROM speaker_local_voiceprints WHERE jobId = :jobId ORDER BY chunkIndex, localSpeaker",
    )
    suspend fun getSpeakerLocalVoiceprints(jobId: String): List<SpeakerLocalVoiceprint>

    @Upsert
    suspend fun upsertSpeakerLocalVoiceprints(voiceprints: List<SpeakerLocalVoiceprint>)

    @Query("DELETE FROM speaker_local_voiceprints WHERE jobId = :jobId")
    suspend fun deleteSpeakerLocalVoiceprints(jobId: String)

    @Query("SELECT * FROM speaker_analyses WHERE jobId = :jobId")
    suspend fun getSpeakerAnalysis(jobId: String): SpeakerAnalysis?

    @Query("SELECT jobId FROM speaker_analyses")
    suspend fun getSpeakerAnalysisJobIds(): List<String>

    @Upsert
    suspend fun upsertSpeakerAnalysis(analysis: SpeakerAnalysis)

    @Query("DELETE FROM speaker_analyses WHERE jobId = :jobId")
    suspend fun deleteSpeakerAnalysis(jobId: String)

    @Query("SELECT * FROM speaker_persons ORDER BY isSelf DESC, createdAtMillis")
    fun observeSpeakerPersons(): Flow<List<SpeakerPerson>>

    @Query("SELECT * FROM speaker_voice_samples ORDER BY createdAtMillis")
    fun observeSpeakerVoiceSamples(): Flow<List<SpeakerVoiceSample>>

    @Query("SELECT * FROM speaker_voice_samples WHERE id = :sampleId")
    suspend fun getSpeakerVoiceSample(sampleId: String): SpeakerVoiceSample?

    @Query("SELECT * FROM speaker_analyses WHERE jobId = :jobId")
    fun observeSpeakerAnalysis(jobId: String): Flow<SpeakerAnalysis?>

    @Query("SELECT * FROM speaker_segment_overrides WHERE jobId = :jobId")
    fun observeSpeakerSegmentOverrides(jobId: String): Flow<List<SpeakerSegmentOverride>>

    @Upsert
    suspend fun upsertSpeakerSegmentOverride(override: SpeakerSegmentOverride)

    @Query("DELETE FROM speaker_segment_overrides WHERE jobId = :jobId AND segmentId = :segmentId")
    suspend fun deleteSpeakerSegmentOverride(jobId: String, segmentId: String)

    @Query("SELECT * FROM speaker_segment_overrides WHERE jobId = :jobId")
    suspend fun getSpeakerSegmentOverrides(jobId: String): List<SpeakerSegmentOverride>
}
