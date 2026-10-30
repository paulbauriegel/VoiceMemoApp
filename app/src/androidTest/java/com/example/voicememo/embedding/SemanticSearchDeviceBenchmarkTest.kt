package com.example.voicememo.embedding

import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import androidx.room.Room
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.voicememo.data.MemoRecord
import com.example.voicememo.data.SemanticHit
import com.example.voicememo.data.SemanticTranscriptRanking
import com.example.voicememo.data.TimedTranscriptSegment
import com.example.voicememo.data.VoiceMemoDatabase
import com.example.voicememo.data.WaveformState
import kotlinx.coroutines.runBlocking
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Device parity and indexing cost for semantic find-in-transcript.
 *
 * Compares on-device embeddings (Kotlin tokenizer + Android ONNX Runtime)
 * with desktop references from scripts/generate-semantic-search-device-fixture.py
 * and indexes its five synthetic transcripts through [SemanticTranscriptIndex]
 * into an in-memory Room database. Results are reported as an instrumentation
 * status bundle and logged under the `SemanticBenchmark` tag.
 */
@RunWith(AndroidJUnit4::class)
class SemanticSearchDeviceBenchmarkTest {
    @Test
    fun deviceEmbeddingsMatchDesktopAndIndexingIsTimed() = runBlocking {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val fixture = instrumentation.context.assets
            .open("semantic-search/device-reference.json")
            .bufferedReader()
            .use { JSONObject(it.readText()) }
        val samples = fixture.getJSONArray("samples")
        val report = JSONObject()

        // Parity and per-segment timing on a directly opened embedder.
        val loadStarted = SystemClock.elapsedRealtime()
        val embedder = MultilingualE5Embedder.open(context)
        report.put("modelLoadMillis", SystemClock.elapsedRealtime() - loadStarted)
        var minimumCosine = 1f
        var segmentCount = 0
        var embeddingNanos = 0L
        val hub = SemanticTranscriptRanking.centroid(
            SemanticTranscriptRanking.HUB_ANCHOR_QUERIES.map(embedder::embedQuery),
        )
        val queryReports = JSONArray()
        embedder.use {
            for (sampleIndex in 0 until samples.length()) {
                val sample = samples.getJSONObject(sampleIndex)
                val segments = sample.getJSONArray("segments")
                val references = sample.getJSONArray("vectors")
                val deviceVectors = (0 until segments.length()).map { index ->
                    val started = System.nanoTime()
                    val vector = embedder.embedPassage(segments.getString(index))
                    embeddingNanos += System.nanoTime() - started
                    val reference = references.getJSONArray(index).toFloatArray()
                    minimumCosine = minOf(minimumCosine, SemanticTranscriptRanking.dot(vector, reference))
                    segmentCount += 1
                    "$index" to vector
                }
                val queries = sample.getJSONArray("queries")
                for (queryIndex in 0 until queries.length()) {
                    val query = queries.getJSONObject(queryIndex)
                    val expected = query.getJSONArray("hits").toIntList()
                    val expectedScores = query.getJSONArray("scores")
                    val started = System.nanoTime()
                    val hits = SemanticTranscriptRanking.rank(
                        embedder.embedQuery(query.getString("query")),
                        deviceVectors,
                        hub,
                    ).map { it.segmentId.toInt() }
                    val queryMillis = (System.nanoTime() - started) / 1_000_000.0
                    queryReports.put(
                        JSONObject()
                            .put("sample", sample.getString("id"))
                            .put("query", query.getString("query"))
                            .put("expected", JSONArray(expected))
                            .put("device", JSONArray(hits))
                            .put("queryMillis", queryMillis),
                    )
                    // A clear desktop winner must also win on device; scores
                    // near the threshold may legitimately flip with int8 kernels.
                    if (expected.isNotEmpty() && expectedScores.getDouble(0) >= 0.04) {
                        assertEquals("Top hit for ${query.getString("query")}", expected.first(), hits.firstOrNull())
                    }
                }
            }
        }
        report.put("segments", segmentCount)
        report.put("minimumCosineToDesktop", minimumCosine.toDouble())
        report.put("meanEmbeddingMillis", embeddingNanos / 1_000_000.0 / segmentCount)
        report.put("queries", queryReports)

        // End-to-end indexing through Room, as the worker runs it after a job.
        val database = Room.inMemoryDatabaseBuilder(context, VoiceMemoDatabase::class.java).build()
        val index = SemanticTranscriptIndex(
            dao = database.voiceMemoDao(),
            openEmbedder = { MultilingualE5Embedder.open(context) },
        )
        val indexing = JSONArray()
        try {
            for (sampleIndex in 0 until samples.length()) {
                val sample = samples.getJSONObject(sampleIndex)
                val memoId = sample.getString("id")
                database.voiceMemoDao().insertMemo(
                    MemoRecord(
                        id = memoId,
                        title = memoId,
                        createdAtMillis = 0L,
                        updatedAtMillis = 0L,
                        durationMillis = 0L,
                        audioPath = "",
                        waveformPath = null,
                        waveformState = WaveformState.PENDING,
                        languageOverride = null,
                    ),
                )
                val segments = sample.getJSONArray("segments").let { array ->
                    (0 until array.length()).map { position ->
                        TimedTranscriptSegment("$memoId:$position", 0L, 0L, array.getString(position))
                    }
                }
                val started = SystemClock.elapsedRealtime()
                val result = index.index(memoId, segments)
                val wallMillis = SystemClock.elapsedRealtime() - started
                val searchStarted = SystemClock.elapsedRealtime()
                val hits: List<SemanticHit> = index.search(memoId, segments, "what happened")
                val searchMillis = SystemClock.elapsedRealtime() - searchStarted
                assertEquals(segments.size, result.embedded)
                assertTrue(index.index(memoId, segments).embedded == 0)
                indexing.put(
                    JSONObject()
                        .put("sample", memoId)
                        .put("segments", segments.size)
                        .put("indexWallMillis", wallMillis)
                        .put("modelLoadMillis", result.modelLoadMillis)
                        .put("embeddingMillis", result.embeddingMillis)
                        .put("warmSearchMillis", searchMillis)
                        .put("searchHits", hits.size),
                )
                // Cold-start every sample, like a worker that releases the model.
                index.release()
            }
        } finally {
            index.release()
            database.close()
        }
        report.put("indexing", indexing)

        Log.i(TAG, report.toString())
        instrumentation.sendStatus(0, Bundle().apply { putString("semanticBenchmark", report.toString()) })
        assertTrue("Device embeddings drifted: $minimumCosine", minimumCosine >= 0.99f)
    }

    private fun JSONArray.toFloatArray() = FloatArray(length()) { getDouble(it).toFloat() }

    private fun JSONArray.toIntList() = (0 until length()).map(::getInt)

    private companion object {
        const val TAG = "SemanticBenchmark"
    }
}
