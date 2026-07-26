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

    @Query("UPDATE memos SET pendingDeletion = :pending, updatedAtMillis = :updatedAtMillis WHERE id IN (:memoIds)")
    suspend fun markPendingDeletion(
        memoIds: Collection<String>,
        pending: Boolean,
        updatedAtMillis: Long,
    )

    @Query("DELETE FROM memos WHERE id IN (:memoIds)")
    suspend fun deleteMemos(memoIds: Collection<String>)

    @Query("SELECT * FROM transcripts WHERE memoId = :memoId LIMIT 1")
    suspend fun getTranscript(memoId: String): TranscriptRecord?

    @Upsert
    suspend fun upsertTranscript(transcript: TranscriptRecord)

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
}
