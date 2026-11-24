package com.example.voicememo.speaker

import androidx.room.Room
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.voicememo.data.MemoRecord
import com.example.voicememo.data.SpeakerPerson
import com.example.voicememo.data.SpeakerVoiceSample
import com.example.voicememo.data.VoiceMemoDatabase
import com.example.voicememo.data.WaveformState
import com.example.voicememo.transcription.ParakeetBenchmarkSupport
import java.io.File
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith

/**
 * End-to-end check of [SpeakerLabelIndex] in the isolated benchmark package:
 * analyse a staged public sample into an in-memory database, pause and resume,
 * enrol the first voiceprint found as "me" and label. The host runner
 * (`scripts/run-speaker-label-benchmark.py`) compares the timeline with the
 * numpy/scipy reference. Nothing touches the app's real database.
 */
@RunWith(AndroidJUnit4::class)
class SpeakerLabelIndexBenchmarkTest {
    @Test
    fun runBenchmark(): Unit = runBlocking<Unit> {
        val arguments = InstrumentationRegistry.getArguments()
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        check(context.packageName == BENCHMARK_PACKAGE) {
            "Refusing to run against non-isolated package ${context.packageName}."
        }
        val runId = arguments.getString("runId")?.takeIf(SAFE_VALUE::matches) ?: error("runId is missing or invalid.")
        val audioFile = ParakeetBenchmarkSupport.stagedFile(
            context, DIRECTORY, arguments.getString("audioName").orEmpty(), ".wav",
        )
        val reportDirectory = File(context.cacheDir, DIRECTORY).apply { mkdirs() }
        val reportFile = File(reportDirectory, "$runId.json")
        val temporaryReport = File(reportDirectory, "$runId.json.tmp")
        val timelineFile = File(reportDirectory, "$runId.timeline")
        listOf(reportFile, temporaryReport, timelineFile).forEach(File::delete)
        val report = JSONObject().put("run_id", runId).put("package_name", context.packageName).put("status", "running")

        fun publish() {
            temporaryReport.writeText(report.toString(1))
            reportFile.delete()
            check(temporaryReport.renameTo(reportFile)) { "Could not publish the report." }
        }
        publish()
        val database = Room.inMemoryDatabaseBuilder(context, VoiceMemoDatabase::class.java).build()
        try {
            val dao = database.voiceMemoDao()
            val index = SpeakerLabelIndex(
                dao = dao,
                openSegmenter = { PyannoteSegmenter.open(context) },
                openEmbedder = { CamPlusPlusEmbedder.open(context) },
            )
            dao.insertMemo(
                MemoRecord(
                    id = MEMO,
                    title = "benchmark",
                    createdAtMillis = 0,
                    updatedAtMillis = 0,
                    durationMillis = 0,
                    audioPath = audioFile.absolutePath,
                    waveformPath = null,
                    waveformState = WaveformState.entries.first(),
                    languageOverride = null,
                ),
            )

            // 1. Pause after a few windows, then continue: must equal one uninterrupted run.
            var seen = 0
            val paused = index.analyse(MEMO, "paused", audioFile, shouldPause = { seen >= PAUSE_AFTER }) { _, _ -> seen++ }
            val storedAtPause = dao.getSpeakerLocalVoiceprints("paused").size
            val resumed = checkNotNull(index.analyse(MEMO, "paused", audioFile))

            // 2. One uninterrupted run, timed.
            var windows = 0
            val full = checkNotNull(index.analyse(MEMO, JOB, audioFile) { _, total -> windows = total })
            check(!full.labelled) { "Labelled without any voice profile." }
            val voiceprints = dao.getSpeakerLocalVoiceprints(JOB)
            check(voiceprints.size == full.voiceprints)

            // 3. Enrol the first voiceprint as "me" and label from the stored voiceprints.
            dao.upsertSpeakerPerson(SpeakerPerson(id = "me", name = "Me", isSelf = true, createdAtMillis = 0))
            dao.upsertSpeakerVoiceSample(
                SpeakerVoiceSample(
                    id = "sample",
                    personId = "me",
                    modelId = CamPlusPlusEmbedder.MODEL_ID,
                    vector = voiceprints.first().vector,
                    origin = SpeakerVoiceSample.ORIGIN_SEGMENT,
                    memoId = MEMO,
                    startMillis = null,
                    endMillis = null,
                    inputChannel = null,
                    audioFileName = null,
                    speechMillis = 0,
                    createdAtMillis = 0,
                ),
            )
            val relabelStarted = System.nanoTime()
            val relabelled = index.relabelAll()
            val relabelMillis = (System.nanoTime() - relabelStarted) / 1e6
            val analysis = checkNotNull(dao.getSpeakerAnalysis(JOB))
            val timeline = checkNotNull(analysis.timeline)
            timelineFile.writeBytes(timeline)
            val decoded = SpeakerTimeline.decode(timeline)

            report.put(
                "result",
                JSONObject()
                    .put("windows", windows)
                    .put("voiceprints", full.voiceprints)
                    .put("analysis_ms", full.analysisMillis)
                    .put("sample_count", analysis.sampleCount)
                    .put("timeline_frames", decoded.frameCount)
                    .put("timeline_bytes", timeline.size)
                    .put("relabelled_runs", relabelled)
                    .put("relabel_ms", relabelMillis)
                    .put("paused_returned_null", paused == null)
                    .put("voiceprints_stored_at_pause", storedAtPause)
                    .put("voiceprints_after_resume", resumed.voiceprints)
                    .put("windows_run_after_resume", resumed.windows)
                    .put("memory", ParakeetBenchmarkSupport.memory()),
            )
            report.put("status", "completed")
            Unit
        } catch (error: Throwable) {
            report.put("status", "failed")
                .put("error_type", error.javaClass.name)
                .put("error_message", error.message?.lineSequence()?.firstOrNull()?.take(500))
            throw error
        } finally {
            database.close()
            publish()
        }
    }

    private companion object {
        private const val BENCHMARK_PACKAGE = "com.example.voicememo.benchmark"
        private const val DIRECTORY = "speaker-chain-benchmark"
        private const val MEMO = "benchmark-memo"
        private const val JOB = "benchmark-job"
        private const val PAUSE_AFTER = 7
        private val SAFE_VALUE = Regex("[A-Za-z0-9-]{1,128}")
    }
}
