package com.example.voicememo.work

import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.content.pm.ServiceInfo
import android.os.Build
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.work.CoroutineWorker
import androidx.work.Data
import androidx.work.ExistingWorkPolicy
import androidx.work.ForegroundInfo
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkInfo
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import com.example.voicememo.BuildConfig
import com.example.voicememo.R
import com.example.voicememo.VoiceMemoApplication
import com.example.voicememo.coordination.InferenceGate
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Labels the speakers of an already transcribed recording. New transcriptions
 * are labelled by [TranscriptionWorker] itself; this covers recordings from
 * before the feature was switched on. It waits while a transcription runs or a
 * recording is in progress, and continues where it stopped.
 */
class SpeakerLabelWorker(
    context: Context,
    parameters: WorkerParameters,
) : CoroutineWorker(context, parameters) {
    override suspend fun doWork(): Result {
        if (!BuildConfig.SPEAKER_LABELS_ENABLED) return Result.success()
        val memoId = inputData.getString(KEY_MEMO_ID) ?: return Result.failure()
        val jobId = inputData.getString(KEY_JOB_ID) ?: return Result.failure()
        val container = (applicationContext as VoiceMemoApplication).container
        val memo = container.memoRepository.getMemo(memoId) ?: return Result.failure()
        if (InferenceGate.isRecording || transcriptionRunning(container.workManager)) return Result.retry()
        return try {
            setForeground(foregroundInfo(0f))
            val result = container.speakerLabelIndex.analyse(
                memoId = memoId,
                jobId = jobId,
                audioFile = File(memo.audioPath),
                shouldPause = { isStopped || InferenceGate.isRecording },
            ) { window, windows ->
                if (window % PROGRESS_WINDOWS == 0) setForeground(foregroundInfo(window.toFloat() / windows))
            }
            if (result == null) {
                Result.retry()
            } else {
                Log.i(TAG, "Speaker labels: ${result.voiceprints} voiceprints in ${result.analysisMillis} ms.")
                Result.success()
            }
        } catch (failure: Exception) {
            if (isStopped) throw failure
            Log.w(TAG, "Speaker labelling failed.", failure)
            Result.failure()
        }
    }

    private suspend fun transcriptionRunning(workManager: WorkManager): Boolean = withContext(Dispatchers.IO) {
        workManager.getWorkInfosByTag(WorkManagerTranscriptionCoordinator.QUEUE_TAG).get()
            .any { it.state == WorkInfo.State.RUNNING }
    }

    private fun foregroundInfo(progress: Float): ForegroundInfo {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            applicationContext.getSystemService(NotificationManager::class.java).createNotificationChannel(
                NotificationChannel(
                    CHANNEL_ID,
                    applicationContext.getString(R.string.transcription_channel_name),
                    NotificationManager.IMPORTANCE_LOW,
                ),
            )
        }
        val notification = NotificationCompat.Builder(applicationContext, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setContentTitle("Identifying speakers")
            .setContentText("Working on this device")
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setProgress(100, (progress.coerceIn(0f, 1f) * 100).toInt(), false)
            .build()
        return ForegroundInfo(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
    }

    companion object {
        private const val TAG = "SpeakerLabelWorker"
        private const val KEY_MEMO_ID = "memo_id"
        private const val KEY_JOB_ID = "job_id"
        /** Same channel as transcription progress. */
        private const val CHANNEL_ID = "local_transcription"
        private const val NOTIFICATION_ID = 1003
        private const val PROGRESS_WINDOWS = 20
        private const val WORK_NAME = "speaker-labels"

        /** One recording at a time, in the order requested. Deleting the memo cancels it (memo tag). */
        fun enqueue(workManager: WorkManager, memoId: String, jobId: String) {
            val request = OneTimeWorkRequestBuilder<SpeakerLabelWorker>()
                .setInputData(Data.Builder().putString(KEY_MEMO_ID, memoId).putString(KEY_JOB_ID, jobId).build())
                .addTag(WorkManagerTranscriptionCoordinator.memoTag(memoId))
                .addTag(TAG_SPEAKER_LABELS)
                .addTag(jobTag(jobId))
                .build()
            workManager.beginUniqueWork(WORK_NAME, ExistingWorkPolicy.APPEND_OR_REPLACE, request).enqueue()
        }

        const val TAG_SPEAKER_LABELS = "speaker-labels"
        fun jobTag(jobId: String): String = "speaker-labels-job:$jobId"
    }
}
