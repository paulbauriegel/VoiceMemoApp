package com.example.voicememo.work

import android.content.Context
import android.os.Build
import android.os.Debug
import android.os.PowerManager
import android.os.Process
import android.os.SystemClock
import android.util.Log
import com.example.voicememo.transcription.LongFormChunk
import com.example.voicememo.transcription.LongFormTranscript
import com.example.voicememo.transcription.WhisperBackendSelection
import java.io.File
import org.json.JSONObject

/**
 * Low-overhead, content-free telemetry for repeatable on-device comparisons.
 *
 * Reports contain timings and device/runtime state only. Memo titles, audio,
 * generated tokens, and transcript text are deliberately excluded.
 */
internal class TranscriptionBenchmark(
    context: Context,
    private val audioDurationMillis: Long,
    private val segmentCount: Int,
    private val precompletedSegmentCount: Int,
    private val workerCount: Int,
    private val threadsPerWorker: Int,
) {
    private val applicationContext = context.applicationContext
    private val powerManager = applicationContext.getSystemService(PowerManager::class.java)
    private val totalStartedWallMillis = SystemClock.elapsedRealtime()
    private val totalStartedCpuMillis = Process.getElapsedCpuTime()
    private val startPssKb = Debug.getPss()
    private val startThermalStatus = currentThermalStatus()

    private var maximumThermalStatus = startThermalStatus
    private var modelStartedWallMillis: Long? = null
    private var modelStartedCpuMillis: Long? = null
    private var modelLoadWallMillis: Long? = null
    private var modelLoadCpuMillis: Long? = null
    private var transcriptionStartedWallMillis: Long? = null
    private var transcriptionStartedCpuMillis: Long? = null
    private var transcriptionWallMillis: Long? = null
    private var transcriptionCpuMillis: Long? = null
    private var summedSegmentInferenceMillis = 0L
    private var completedSegments = 0
    private var vadSkippedSegments = 0
    private var backendSelection = WhisperBackendSelection.cpuRequested()
    private var finished = false

    @Synchronized
    fun backendSelectionChanged(selection: WhisperBackendSelection) {
        backendSelection = selection
    }

    @Synchronized
    fun modelLoadingStarted() {
        sampleThermalStatus()
        modelStartedWallMillis = SystemClock.elapsedRealtime()
        modelStartedCpuMillis = Process.getElapsedCpuTime()
    }

    @Synchronized
    fun modelLoadingCompleted() {
        val wallStart = modelStartedWallMillis ?: return
        val cpuStart = modelStartedCpuMillis ?: return
        modelLoadWallMillis = SystemClock.elapsedRealtime() - wallStart
        modelLoadCpuMillis = Process.getElapsedCpuTime() - cpuStart
        sampleThermalStatus()
    }

    @Synchronized
    fun transcriptionStarted() {
        sampleThermalStatus()
        transcriptionStartedWallMillis = SystemClock.elapsedRealtime()
        transcriptionStartedCpuMillis = Process.getElapsedCpuTime()
    }

    @Synchronized
    fun segmentCompleted(chunk: LongFormChunk) {
        completedSegments++
        if (chunk.skippedByVad) vadSkippedSegments++
        summedSegmentInferenceMillis += chunk.result.inferenceMillis
        sampleThermalStatus()
    }

    @Synchronized
    fun transcriptionCompleted(transcript: LongFormTranscript) {
        val wallStart = transcriptionStartedWallMillis ?: return
        val cpuStart = transcriptionStartedCpuMillis ?: return
        transcriptionWallMillis = SystemClock.elapsedRealtime() - wallStart
        transcriptionCpuMillis = Process.getElapsedCpuTime() - cpuStart
        // Resume can supply already-completed chunks that do not pass through
        // segmentCompleted in this process. Use the final transcript as the
        // authoritative count while retaining only this run's inference sum.
        completedSegments = transcript.chunks.size
        sampleThermalStatus()
    }

    @Synchronized
    fun finish(
        outcome: String,
        errorCode: String? = null,
    ) {
        if (finished) return
        finished = true
        sampleThermalStatus()

        val totalWallMillis = SystemClock.elapsedRealtime() - totalStartedWallMillis
        val totalCpuMillis = Process.getElapsedCpuTime() - totalStartedCpuMillis
        val transcribeWall = transcriptionWallMillis
        val backend = backendSelection
        val report = JSONObject()
            .put("schema_version", 1)
            .put("timestamp_millis", System.currentTimeMillis())
            .put("outcome", outcome)
            .put("error_code", errorCode)
            .put("backend_requested", backend.requested.reportValue)
            .put("backend", backend.actual.reportValue)
            .put(
                "backend_fallback_stage",
                backend.fallbackStage?.reportValue ?: JSONObject.NULL,
            )
            .put(
                "backend_fallback_reason",
                backend.fallbackReason ?: JSONObject.NULL,
            )
            .put("gpu_delegate", backend.actual.usesGpuAccelerator)
            .put("gpu_accelerator", backend.actual.usesGpuAccelerator)
            .put("xnnpack", backend.actual.usesCpuXnnpack)
            .put("workers", workerCount)
            .put("threads_per_worker", threadsPerWorker)
            .put(
                "maximum_inference_threads",
                if (backend.actual.usesCpuXnnpack) {
                    workerCount * threadsPerWorker
                } else {
                    JSONObject.NULL
                },
            )
            .put("device", "${Build.MANUFACTURER} ${Build.MODEL}")
            .put(
                "soc",
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) Build.SOC_MODEL else "unknown",
            )
            .put("android_sdk", Build.VERSION.SDK_INT)
            .put("audio_duration_ms", audioDurationMillis)
            .put("segment_count", segmentCount)
            .put("precompleted_segments", precompletedSegmentCount)
            .put("fresh_full_run", precompletedSegmentCount == 0)
            .put("completed_segments", completedSegments)
            .put("vad_skipped_segments", vadSkippedSegments)
            .put("model_load_wall_ms", modelLoadWallMillis)
            .put("model_load_process_cpu_ms", modelLoadCpuMillis)
            .put("transcription_wall_ms", transcribeWall)
            .put("transcription_process_cpu_ms", transcriptionCpuMillis)
            .put("summed_segment_inference_ms", summedSegmentInferenceMillis)
            .put("total_job_wall_ms", totalWallMillis)
            .put("total_job_process_cpu_ms", totalCpuMillis)
            .put("start_pss_kb", startPssKb)
            .put("end_pss_kb", Debug.getPss())
            .put("start_thermal_status", thermalStatusName(startThermalStatus))
            .put("maximum_thermal_status", thermalStatusName(maximumThermalStatus))
            .put("end_thermal_status", thermalStatusName(currentThermalStatus()))
            .put(
                "real_time_factor",
                if (precompletedSegmentCount == 0) {
                    ratio(transcribeWall, audioDurationMillis)
                } else {
                    JSONObject.NULL
                },
            )
            .put(
                "audio_speed_x",
                if (precompletedSegmentCount == 0) {
                    ratio(audioDurationMillis, transcribeWall)
                } else {
                    JSONObject.NULL
                },
            )
            .put(
                "mean_active_inference_lanes",
                ratio(summedSegmentInferenceMillis, transcribeWall),
            )

        persist(report)
        Log.i(LOG_TAG, report.toString())
    }

    private fun persist(report: JSONObject) {
        runCatching {
            val directory = File(applicationContext.filesDir, "benchmarks").apply {
                check(exists() || mkdirs()) { "Could not create benchmark directory." }
            }
            val compact = report.toString()
            val temporary = File(directory, "latest.json.tmp")
            temporary.writeText(report.toString(2))
            check(temporary.renameTo(File(directory, "latest.json"))) {
                "Could not finalize benchmark report."
            }
            File(directory, "history.ndjson").appendText("$compact\n")
        }.onFailure { error ->
            Log.w(LOG_TAG, "Could not persist transcription benchmark.", error)
        }
    }

    private fun sampleThermalStatus() {
        maximumThermalStatus = maxOf(maximumThermalStatus, currentThermalStatus())
    }

    private fun currentThermalStatus(): Int =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            powerManager.currentThermalStatus
        } else {
            THERMAL_STATUS_UNAVAILABLE
        }

    private fun thermalStatusName(status: Int): String = when (status) {
        PowerManager.THERMAL_STATUS_NONE -> "NONE"
        PowerManager.THERMAL_STATUS_LIGHT -> "LIGHT"
        PowerManager.THERMAL_STATUS_MODERATE -> "MODERATE"
        PowerManager.THERMAL_STATUS_SEVERE -> "SEVERE"
        PowerManager.THERMAL_STATUS_CRITICAL -> "CRITICAL"
        PowerManager.THERMAL_STATUS_EMERGENCY -> "EMERGENCY"
        PowerManager.THERMAL_STATUS_SHUTDOWN -> "SHUTDOWN"
        else -> "UNAVAILABLE"
    }

    private fun ratio(numerator: Long?, denominator: Long?): Any {
        if (numerator == null || denominator == null || denominator <= 0L) {
            return JSONObject.NULL
        }
        return numerator.toDouble() / denominator
    }

    companion object {
        const val LOG_TAG = "VoiceMemoBench"
        private const val THERMAL_STATUS_UNAVAILABLE = -1
    }
}
