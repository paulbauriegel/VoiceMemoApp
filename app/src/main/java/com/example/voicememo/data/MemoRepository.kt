package com.example.voicememo.data

import androidx.room.withTransaction
import java.util.Calendar
import java.util.UUID
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.flow.map

/**
 * Coordinates side effects that must finish before Room permanently removes a
 * memo. Implementations cancel WorkManager jobs and delete only the app-private
 * audio/waveform files represented by [MemoRecord].
 */
interface MemoDeletionHooks {
    suspend fun cancelTranscriptions(memoIds: Set<String>)

    /**
     * Delete all paths or throw. Implementations should stage/rename memo
     * directories first when possible so a failure can be rolled back.
     */
    suspend fun deletePrivateFiles(memos: List<MemoRecord>)
}

interface MemoRepository {
    fun observeLibrary(query: String = ""): Flow<List<MemoSummary>>

    /**
     * Observes the memo corpus once and applies changing queries in memory.
     * This avoids restarting Room's aggregate query on every keystroke.
     */
    fun observeLibraryQueries(queries: Flow<String>): Flow<MemoLibrarySearchResult>

    fun observeMonthGroups(query: String = ""): Flow<List<MemoMonthGroup>>

    fun observeMemo(memoId: String): Flow<MemoDetail?>

    suspend fun getMemo(memoId: String): MemoRecord?

    suspend fun createMemo(newMemo: NewMemo): MemoRecord

    suspend fun updateTitle(memoId: String, title: String)

    suspend fun updateDurationAndWaveform(
        memoId: String,
        durationMillis: Long,
        waveformPath: String?,
        waveformState: WaveformState,
        updatedAtMillis: Long = System.currentTimeMillis(),
    )

    /** Null selects automatic language detection. */
    suspend fun setLanguageOverride(memoId: String, language: String?)

    /**
     * Replaces only the generated copy. An edited copy, including an
     * intentionally empty edit, is preserved.
     */
    suspend fun saveGeneratedTranscript(
        memoId: String,
        text: String,
        detectedLanguage: String?,
        generatedAtMillis: Long = System.currentTimeMillis(),
    )

    suspend fun saveEditedTranscript(
        memoId: String,
        text: String,
        editedAtMillis: Long = System.currentTimeMillis(),
    )

    suspend fun resetEditedTranscript(memoId: String)

    suspend fun createTranscriptionJob(
        memoId: String,
        requestedLanguage: String?,
        totalChunks: Int,
        transcriptionModelId: String = "whisper-large-v3-turbo",
        createdAtMillis: Long = System.currentTimeMillis(),
    ): TranscriptionJob

    suspend fun upsertTranscriptionJob(job: TranscriptionJob)

    suspend fun attachWorkId(
        jobId: String,
        workId: UUID,
        updatedAtMillis: Long = System.currentTimeMillis(),
    )

    suspend fun getTranscriptionJob(jobId: String): TranscriptionJob?

    suspend fun getOldestRunnableJob(): TranscriptionJob?

    /**
     * Atomically discards derived chunks when the inference implementation
     * changed, then pins the current fingerprint for safe process-death resume.
     */
    suspend fun prepareTranscriptionJobForInference(
        jobId: String,
        inferenceFingerprint: String,
    ): TranscriptionJob

    suspend fun upsertTranscriptChunk(chunk: TranscriptChunk)

    suspend fun completedTranscriptChunks(jobId: String): List<TranscriptChunk>

    suspend fun clearTranscriptChunks(jobId: String)

    /**
     * Atomically stores the merged generated result and marks its durable job
     * complete. Any existing edited transcript remains effective.
     */
    suspend fun completeTranscriptionJob(
        jobId: String,
        mergedText: String,
        detectedLanguage: String?,
        warning: String?,
        completedAtMillis: Long = System.currentTimeMillis(),
    )

    suspend fun selectAllIds(query: String = ""): Set<String>

    /**
     * Cancels active work, removes private files, and then atomically deletes
     * all Room rows (foreign keys cascade transcripts, chunks, and jobs).
     */
    suspend fun deleteMemos(memoIds: Set<String>)
}

class RoomMemoRepository(
    private val database: VoiceMemoDatabase,
    private val deletionHooks: MemoDeletionHooks,
) : MemoRepository {
    private val dao = database.voiceMemoDao()

    override fun observeLibrary(query: String): Flow<List<MemoSummary>> =
        observeLibraryQueries(flowOf(query)).map { result -> result.memos }

    override fun observeLibraryQueries(
        queries: Flow<String>,
    ): Flow<MemoLibrarySearchResult> {
        val summaries = dao.observeAllMemos()
            .map { rows -> rows.map(MemoAggregate::toSummary) }
            .distinctUntilChanged()
        return combine(
            summaries,
            queries.distinctUntilChanged(),
        ) { memos, query ->
            MemoLibrarySearchResult(
                query = query,
                memos = MemoSearchEngine.search(
                    memos = memos,
                    query = query,
                ),
            )
        }
            .distinctUntilChanged()
            .flowOn(Dispatchers.Default)
    }

    override fun observeMonthGroups(query: String): Flow<List<MemoMonthGroup>> =
        observeLibrary(query).map(::groupByMonth)

    override fun observeMemo(memoId: String): Flow<MemoDetail?> =
        dao.observeMemo(memoId)
            .flatMapLatest { aggregate ->
                if (aggregate == null) {
                    flowOf(null)
                } else {
                    val latestJob = aggregate.transcriptionJobs.maxByOrNull { it.updatedAtMillis }
                    if (latestJob == null) {
                        flowOf(
                            MemoDetail(
                                memo = aggregate.memo,
                                transcript = aggregate.transcripts.singleOrNull(),
                                latestJob = null,
                                chunks = emptyList(),
                            ),
                        )
                    } else {
                        dao.observeTranscriptChunks(latestJob.id).map { chunks ->
                            MemoDetail(
                                memo = aggregate.memo,
                                transcript = aggregate.transcripts.singleOrNull(),
                                latestJob = latestJob,
                                chunks = chunks,
                            )
                        }
                    }
                }
            }
            .distinctUntilChanged()

    override suspend fun getMemo(memoId: String): MemoRecord? = dao.getMemo(memoId)

    override suspend fun createMemo(newMemo: NewMemo): MemoRecord {
        require(newMemo.title.isNotBlank()) { "Memo title must not be blank." }
        require(newMemo.durationMillis >= 0) { "Memo duration must not be negative." }
        val memo = MemoRecord(
            id = UUID.randomUUID().toString(),
            title = newMemo.title.trim(),
            createdAtMillis = newMemo.createdAtMillis,
            updatedAtMillis = newMemo.createdAtMillis,
            durationMillis = newMemo.durationMillis,
            audioPath = newMemo.audioPath,
            waveformPath = newMemo.waveformPath,
            waveformState = newMemo.waveformState,
            languageOverride = null,
        )
        database.withTransaction {
            dao.insertMemo(memo)
            dao.replaceSearchEntry(memo, transcript = null)
        }
        return memo
    }

    override suspend fun updateTitle(memoId: String, title: String) {
        require(title.isNotBlank()) { "Memo title must not be blank." }
        database.withTransaction {
            val memo = requireMemo(memoId).copy(
                title = title.trim(),
                updatedAtMillis = System.currentTimeMillis(),
            )
            dao.updateMemo(memo)
            dao.replaceSearchEntry(memo, dao.getTranscript(memoId))
        }
    }

    override suspend fun updateDurationAndWaveform(
        memoId: String,
        durationMillis: Long,
        waveformPath: String?,
        waveformState: WaveformState,
        updatedAtMillis: Long,
    ) {
        require(durationMillis >= 0) { "Memo duration must not be negative." }
        database.withTransaction {
            val memo = requireMemo(memoId).copy(
                durationMillis = durationMillis,
                waveformPath = waveformPath,
                waveformState = waveformState,
                updatedAtMillis = updatedAtMillis,
            )
            dao.updateMemo(memo)
        }
    }

    override suspend fun setLanguageOverride(memoId: String, language: String?) {
        val normalizedLanguage = language?.trim()?.lowercase()?.takeIf(String::isNotEmpty)
        database.withTransaction {
            val memo = requireMemo(memoId).copy(
                languageOverride = normalizedLanguage,
                updatedAtMillis = System.currentTimeMillis(),
            )
            dao.updateMemo(memo)
        }
    }

    override suspend fun saveGeneratedTranscript(
        memoId: String,
        text: String,
        detectedLanguage: String?,
        generatedAtMillis: Long,
    ) {
        database.withTransaction {
            val memo = requireMemo(memoId)
            val current = dao.getTranscript(memoId)
            val updated = TranscriptRecord(
                memoId = memoId,
                generatedText = text,
                editedText = current?.editedText,
                detectedLanguage = detectedLanguage,
                generatedAtMillis = generatedAtMillis,
                editedAtMillis = current?.editedAtMillis,
                generatedRevision = (current?.generatedRevision ?: 0) + 1,
            )
            dao.upsertTranscript(updated)
            dao.updateMemo(memo.copy(updatedAtMillis = generatedAtMillis))
            dao.replaceSearchEntry(
                memo = memo.copy(updatedAtMillis = generatedAtMillis),
                transcript = updated,
            )
        }
    }

    override suspend fun saveEditedTranscript(
        memoId: String,
        text: String,
        editedAtMillis: Long,
    ) {
        database.withTransaction {
            val memo = requireMemo(memoId)
            val current = dao.getTranscript(memoId)
            val updated = TranscriptRecord(
                memoId = memoId,
                generatedText = current?.generatedText,
                editedText = text,
                detectedLanguage = current?.detectedLanguage,
                generatedAtMillis = current?.generatedAtMillis,
                editedAtMillis = editedAtMillis,
                generatedRevision = current?.generatedRevision ?: 0,
            )
            dao.upsertTranscript(updated)
            dao.updateMemo(memo.copy(updatedAtMillis = editedAtMillis))
            dao.replaceSearchEntry(
                memo = memo.copy(updatedAtMillis = editedAtMillis),
                transcript = updated,
            )
        }
    }

    override suspend fun resetEditedTranscript(memoId: String) {
        database.withTransaction {
            val memo = requireMemo(memoId)
            val current = dao.getTranscript(memoId) ?: return@withTransaction
            val updatedAt = System.currentTimeMillis()
            val updated = current.copy(
                editedText = null,
                editedAtMillis = null,
            )
            dao.upsertTranscript(updated)
            dao.updateMemo(memo.copy(updatedAtMillis = updatedAt))
            dao.replaceSearchEntry(
                memo = memo.copy(updatedAtMillis = updatedAt),
                transcript = updated,
            )
        }
    }

    override suspend fun createTranscriptionJob(
        memoId: String,
        requestedLanguage: String?,
        totalChunks: Int,
        transcriptionModelId: String,
        createdAtMillis: Long,
    ): TranscriptionJob {
        require(totalChunks > 0) { "A transcription job needs at least one chunk." }
        val memo = requireNotNull(dao.getMemo(memoId)) { "Unknown memo: $memoId" }
        check(!memo.pendingDeletion) { "Memo is being deleted: $memoId" }
        val job = TranscriptionJob(
            id = UUID.randomUUID().toString(),
            memoId = memoId,
            workId = null,
            status = TranscriptionJobStatus.QUEUED,
            phase = TranscriptionPhase.QUEUED,
            requestedLanguage = requestedLanguage
                ?.trim()
                ?.lowercase()
                ?.takeIf(String::isNotEmpty)
                ?: memo.languageOverride,
            detectedLanguage = null,
            totalChunks = totalChunks,
            completedChunks = 0,
            currentChunkIndex = null,
            currentToken = 0,
            currentTokenLimit = 128,
            errorCode = null,
            errorMessage = null,
            warning = null,
            createdAtMillis = createdAtMillis,
            updatedAtMillis = createdAtMillis,
            transcriptionModelId = transcriptionModelId,
        )
        dao.upsertTranscriptionJob(job)
        return job
    }

    override suspend fun upsertTranscriptionJob(job: TranscriptionJob) {
        require(job.totalChunks > 0) { "A transcription job needs at least one chunk." }
        require(job.completedChunks in 0..job.totalChunks) { "Invalid completed chunk count." }
        dao.upsertTranscriptionJob(job)
    }

    override suspend fun attachWorkId(jobId: String, workId: UUID, updatedAtMillis: Long) {
        checkNotNull(dao.getTranscriptionJob(jobId)) { "Unknown transcription job: $jobId" }
        dao.attachWorkId(jobId, workId.toString(), updatedAtMillis)
    }

    override suspend fun getTranscriptionJob(jobId: String): TranscriptionJob? =
        dao.getTranscriptionJob(jobId)

    override suspend fun getOldestRunnableJob(): TranscriptionJob? =
        dao.getOldestRunnableJob()

    override suspend fun prepareTranscriptionJobForInference(
        jobId: String,
        inferenceFingerprint: String,
    ): TranscriptionJob {
        require(inferenceFingerprint.isNotBlank()) {
            "Inference fingerprint must not be blank."
        }
        return database.withTransaction {
            val current = requireNotNull(dao.getTranscriptionJob(jobId)) {
                "Unknown transcription job: $jobId"
            }
            val preparation = planTranscriptionResume(
                current = current,
                inferenceFingerprint = inferenceFingerprint,
                updatedAtMillis = System.currentTimeMillis(),
            )
            if (!preparation.discardDerivedChunks) {
                return@withTransaction preparation.job
            }

            dao.deleteTranscriptChunks(jobId)
            preparation.job.also { updated ->
                dao.upsertTranscriptionJob(updated)
            }
        }
    }

    override suspend fun upsertTranscriptChunk(chunk: TranscriptChunk) {
        require(chunk.windowStartMillis >= 0) { "Chunk start must not be negative." }
        require(chunk.windowEndMillis >= chunk.windowStartMillis) { "Chunk window is reversed." }
        dao.upsertTranscriptChunk(chunk)
    }

    override suspend fun completedTranscriptChunks(jobId: String): List<TranscriptChunk> =
        dao.getCompletedTranscriptChunks(jobId)

    override suspend fun clearTranscriptChunks(jobId: String) {
        dao.deleteTranscriptChunks(jobId)
    }

    override suspend fun completeTranscriptionJob(
        jobId: String,
        mergedText: String,
        detectedLanguage: String?,
        warning: String?,
        completedAtMillis: Long,
    ) {
        database.withTransaction {
            val job = requireNotNull(dao.getTranscriptionJob(jobId)) {
                "Unknown transcription job: $jobId"
            }
            val memo = requireMemo(job.memoId)
            val currentTranscript = dao.getTranscript(job.memoId)
            val transcript = TranscriptRecord(
                memoId = job.memoId,
                generatedText = mergedText,
                editedText = currentTranscript?.editedText,
                detectedLanguage = detectedLanguage,
                generatedAtMillis = completedAtMillis,
                editedAtMillis = currentTranscript?.editedAtMillis,
                generatedRevision = (currentTranscript?.generatedRevision ?: 0) + 1,
            )
            val completedJob = job.copy(
                status = TranscriptionJobStatus.SUCCEEDED,
                phase = TranscriptionPhase.COMPLETE,
                detectedLanguage = detectedLanguage,
                completedChunks = job.totalChunks,
                currentChunkIndex = null,
                currentToken = 0,
                warning = warning,
                errorCode = null,
                errorMessage = null,
                updatedAtMillis = completedAtMillis,
            )
            val updatedMemo = memo.copy(updatedAtMillis = completedAtMillis)
            dao.upsertTranscript(transcript)
            dao.upsertTranscriptionJob(completedJob)
            dao.updateMemo(updatedMemo)
            dao.replaceSearchEntry(updatedMemo, transcript)
        }
    }

    override suspend fun selectAllIds(query: String): Set<String> {
        if (query.isBlank()) return dao.getAllMemoIds().toSet()
        return MemoSearchEngine.search(
            memos = dao.getAllMemoAggregates().map(MemoAggregate::toSummary),
            query = query,
        ).mapTo(linkedSetOf(), MemoSummary::id)
    }

    override suspend fun deleteMemos(memoIds: Set<String>) {
        if (memoIds.isEmpty()) return

        val memos = database.withTransaction {
            val records = dao.getMemos(memoIds)
            if (records.isEmpty()) return@withTransaction emptyList()
            val foundIds = records.mapTo(linkedSetOf()) { it.id }
            dao.markPendingDeletion(
                memoIds = foundIds,
                pending = true,
                updatedAtMillis = System.currentTimeMillis(),
            )
            dao.deleteSearchEntries(foundIds)
            records
        }
        if (memos.isEmpty()) return

        val foundIds = memos.mapTo(linkedSetOf()) { it.id }
        try {
            deletionHooks.cancelTranscriptions(foundIds)
            deletionHooks.deletePrivateFiles(memos)
            database.withTransaction {
                dao.deleteSearchEntries(foundIds)
                dao.deleteMemos(foundIds)
            }
        } catch (failure: Throwable) {
            // Make any records whose files were not removed visible again. A
            // staging deletion hook can roll its filesystem operation back
            // before throwing, keeping this recovery fully atomic to the user.
            database.withTransaction {
                dao.markPendingDeletion(
                    memoIds = foundIds,
                    pending = false,
                    updatedAtMillis = System.currentTimeMillis(),
                )
                for (memoId in foundIds) {
                    val memo = dao.getMemo(memoId) ?: continue
                    dao.replaceSearchEntry(memo, dao.getTranscript(memoId))
                }
            }
            throw failure
        }
    }

    private suspend fun requireMemo(memoId: String): MemoRecord =
        requireNotNull(dao.getMemo(memoId)) { "Unknown memo: $memoId" }

    private suspend fun VoiceMemoDao.replaceSearchEntry(
        memo: MemoRecord,
        transcript: TranscriptRecord?,
    ) {
        deleteSearchEntry(memo.id)
        if (!memo.pendingDeletion) {
            insertSearchEntry(
                MemoSearchFts(
                    memoId = memo.id,
                    title = memo.title,
                    effectiveTranscript = transcript?.effectiveText.orEmpty(),
                ),
            )
        }
    }
}

data class MemoLibrarySearchResult(
    val query: String,
    val memos: List<MemoSummary>,
)

internal object MemoSearchQuery {
    private val tokenPattern = Regex("[\\p{L}\\p{N}_]+")

    /**
     * Converts user input to a literal, prefix-matching FTS expression. Only
     * Unicode letters/numbers enter the expression, so FTS operators typed by
     * a user cannot alter the query.
     */
    fun toMatchExpression(query: String): String? {
        val tokens = tokenPattern
            .findAll(query)
            .map { it.value }
            .take(MAX_QUERY_TOKENS)
            .toList()
        if (tokens.isEmpty()) return null
        return tokens.joinToString(separator = " AND ") { token -> "\"$token\"*" }
    }

    private const val MAX_QUERY_TOKENS = 20
}

internal fun groupByMonth(memos: List<MemoSummary>): List<MemoMonthGroup> {
    if (memos.isEmpty()) return emptyList()
    return memos
        .groupBy { memo ->
            val calendar = Calendar.getInstance().apply {
                timeInMillis = memo.createdAtMillis
            }
            calendar.get(Calendar.YEAR) to calendar.get(Calendar.MONTH)
        }
        .map { (yearAndZeroBasedMonth, monthMemos) ->
            val (year, zeroBasedMonth) = yearAndZeroBasedMonth
            val start = Calendar.getInstance().apply {
                clear()
                set(Calendar.YEAR, year)
                set(Calendar.MONTH, zeroBasedMonth)
                set(Calendar.DAY_OF_MONTH, 1)
            }.timeInMillis
            MemoMonthGroup(
                year = year,
                month = zeroBasedMonth + 1,
                monthStartMillis = start,
                memos = monthMemos.sortedByDescending(MemoSummary::createdAtMillis),
            )
        }
        .sortedByDescending(MemoMonthGroup::monthStartMillis)
}
