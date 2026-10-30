package com.example.voicememo.embedding

import android.os.SystemClock
import android.util.Log
import com.example.voicememo.data.SemanticHit
import com.example.voicememo.data.SemanticTranscriptRanking
import com.example.voicememo.data.TimedTranscriptSegment
import com.example.voicememo.data.TranscriptSegmentEmbedding
import com.example.voicememo.data.VoiceMemoDao
import java.security.MessageDigest
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/**
 * Stores per-segment sentence embeddings in Room and answers semantic
 * find-in-transcript queries.
 *
 * The ~120 MB embedding session is loaded on demand and released after
 * [idleReleaseMillis] without use, so it never sits next to Whisper in memory.
 */
class SemanticTranscriptIndex(
    private val dao: VoiceMemoDao,
    private val openEmbedder: () -> SentenceEmbedder,
    private val idleReleaseMillis: Long = 60_000L,
) {
    data class IndexingResult(
        val embedded: Int,
        val reused: Int,
        val removed: Int,
        val modelLoadMillis: Long,
        val embeddingMillis: Long,
    )

    private class LoadedModel(val embedder: SentenceEmbedder, val hubCentroid: FloatArray)

    private val mutex = Mutex()
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var model: LoadedModel? = null
    private var releaseJob: Job? = null

    /**
     * Embeds segments whose text changed or that were never embedded, and
     * drops rows for segments that no longer exist (e.g. after transcribing
     * again).
     */
    suspend fun index(
        memoId: String,
        segments: List<TimedTranscriptSegment>,
    ): IndexingResult = withContext(Dispatchers.Default) {
        mutex.withLock {
            try {
                indexLocked(memoId, segments)
            } finally {
                scheduleRelease()
            }
        }
    }

    /** Ranked semantic hits among [segments], indexing them first if needed. */
    suspend fun search(
        memoId: String,
        segments: List<TimedTranscriptSegment>,
        query: String,
    ): List<SemanticHit> = withContext(Dispatchers.Default) {
        val trimmed = query.trim()
        if (trimmed.length < SemanticTranscriptRanking.MIN_QUERY_LENGTH || segments.isEmpty()) {
            return@withContext emptyList()
        }
        mutex.withLock {
            try {
                indexLocked(memoId, segments)
                val loaded = loadedModel()
                val queryVector = loaded.embedder.embedQuery(trimmed)
                val current = segments.associateBy({ it.id }, { sha256(it.text) })
                val vectors = dao.getSegmentEmbeddings(memoId)
                    .filter { row ->
                        row.modelId == loaded.embedder.modelId &&
                            current[row.segmentId] == row.textSha256
                    }
                    .associateBy(TranscriptSegmentEmbedding::segmentId)
                val ordered = segments.mapNotNull { segment ->
                    vectors[segment.id]?.let {
                        segment.id to SemanticTranscriptRanking.fromBytes(it.vector)
                    }
                }
                SemanticTranscriptRanking.rank(queryVector, ordered, loaded.hubCentroid)
            } finally {
                scheduleRelease()
            }
        }
    }

    /** Closes the embedding session now, e.g. when a worker finishes. */
    suspend fun release() {
        mutex.withLock { releaseLocked() }
    }

    private suspend fun indexLocked(
        memoId: String,
        segments: List<TimedTranscriptSegment>,
    ): IndexingResult {
        val existing = dao.getSegmentEmbeddings(memoId).associateBy(TranscriptSegmentEmbedding::segmentId)
        val wanted = segments.filter { it.text.isNotBlank() }.associateBy(TimedTranscriptSegment::id)
        val stale = existing.keys - wanted.keys
        stale.chunked(SQL_VARIABLE_BATCH).forEach { ids -> dao.deleteSegmentEmbeddings(memoId, ids) }

        var modelLoadMillis = 0L
        val pending = wanted.values.filter { segment ->
            val row = existing[segment.id]
            row == null || row.modelId != MultilingualE5Embedder.MODEL_ID ||
                row.textSha256 != sha256(segment.text)
        }
        if (pending.isEmpty()) {
            return IndexingResult(0, wanted.size, stale.size, 0L, 0L)
        }
        if (model == null) {
            val loadStarted = SystemClock.elapsedRealtime()
            loadedModel()
            modelLoadMillis = SystemClock.elapsedRealtime() - loadStarted
        }
        val embedder = loadedModel().embedder
        val started = SystemClock.elapsedRealtime()
        val now = System.currentTimeMillis()
        val rows = pending.map { segment ->
            TranscriptSegmentEmbedding(
                memoId = memoId,
                segmentId = segment.id,
                textSha256 = sha256(segment.text),
                modelId = embedder.modelId,
                vector = SemanticTranscriptRanking.toBytes(embedder.embedPassage(segment.text.trim())),
                embeddedAtMillis = now,
            )
        }
        val embeddingMillis = SystemClock.elapsedRealtime() - started
        dao.upsertSegmentEmbeddings(rows)
        Log.i(
            TAG,
            "Embedded ${rows.size} segments in $embeddingMillis ms " +
                "(model load $modelLoadMillis ms, reused ${wanted.size - rows.size}).",
        )
        return IndexingResult(
            embedded = rows.size,
            reused = wanted.size - rows.size,
            removed = stale.size,
            modelLoadMillis = modelLoadMillis,
            embeddingMillis = embeddingMillis,
        )
    }

    private fun loadedModel(): LoadedModel {
        model?.let { return it }
        val embedder = openEmbedder()
        val hubCentroid = try {
            SemanticTranscriptRanking.centroid(
                SemanticTranscriptRanking.HUB_ANCHOR_QUERIES.map(embedder::embedQuery),
            )
        } catch (failure: Throwable) {
            embedder.close()
            throw failure
        }
        return LoadedModel(embedder, hubCentroid).also { model = it }
    }

    private fun scheduleRelease() {
        releaseJob?.cancel()
        releaseJob = scope.launch {
            delay(idleReleaseMillis)
            mutex.withLock { releaseLocked() }
        }
    }

    private fun releaseLocked() {
        releaseJob?.cancel()
        releaseJob = null
        model?.embedder?.close()
        model = null
    }

    companion object {
        private const val TAG = "SemanticIndex"
        private const val SQL_VARIABLE_BATCH = 500

        internal fun sha256(text: String): String =
            MessageDigest.getInstance("SHA-256")
                .digest(text.trim().toByteArray(Charsets.UTF_8))
                .joinToString("") { "%02x".format(it) }
    }
}
