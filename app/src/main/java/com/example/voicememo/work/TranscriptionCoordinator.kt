package com.example.voicememo.work

import androidx.work.Data
import androidx.work.ExistingWorkPolicy
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import com.example.voicememo.data.MemoRepository
import com.example.voicememo.data.TranscriptionJobStatus
import com.example.voicememo.data.TranscriptionPhase
import com.example.voicememo.transcription.TranscriptionModel
import java.util.UUID
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

interface TranscriptionCoordinator {
    fun enqueueWhileRecording(
        memoId: String,
        memoDirectory: String,
        timestampsEnabled: Boolean,
        voiceActivityDetectionEnabled: Boolean,
        gpuAccelerationEnabled: Boolean,
    )

    suspend fun enqueue(
        memoId: String,
        languageCode: String?,
        timestampsEnabled: Boolean = false,
        voiceActivityDetectionEnabled: Boolean = true,
        gpuAccelerationEnabled: Boolean = true,
        transcriptionModelId: String = TranscriptionModel.DEFAULT.id,
        reuseRecordingStageResults: Boolean = false,
    ): String
    suspend fun cancel(memoId: String, jobId: String?)
}

class WorkManagerTranscriptionCoordinator(
    private val workManager: WorkManager,
    private val repository: MemoRepository,
) : TranscriptionCoordinator {
    override fun enqueueWhileRecording(
        memoId: String,
        memoDirectory: String,
        timestampsEnabled: Boolean,
        voiceActivityDetectionEnabled: Boolean,
        gpuAccelerationEnabled: Boolean,
    ) {
        val request = OneTimeWorkRequestBuilder<WhisperPreTranscriptionWorker>()
            .setInputData(
                Data.Builder()
                    .putString(WhisperPreTranscriptionWorker.KEY_MEMO_ID, memoId)
                    .putString(
                        WhisperPreTranscriptionWorker.KEY_MEMO_DIRECTORY,
                        memoDirectory,
                    )
                    .putBoolean(
                        WhisperPreTranscriptionWorker.KEY_TIMESTAMPS_ENABLED,
                        timestampsEnabled,
                    )
                    .putBoolean(
                        WhisperPreTranscriptionWorker.KEY_VAD_ENABLED,
                        voiceActivityDetectionEnabled,
                    )
                    .putBoolean(
                        WhisperPreTranscriptionWorker.KEY_GPU_ACCELERATION_ENABLED,
                        gpuAccelerationEnabled,
                    )
                    .build(),
            )
            .addTag(QUEUE_TAG)
            .addTag(memoTag(memoId))
            .build()
        workManager.beginUniqueWork(
            FIFO_WORK_NAME,
            ExistingWorkPolicy.APPEND_OR_REPLACE,
            request,
        ).enqueue()
    }

    override suspend fun enqueue(
        memoId: String,
        languageCode: String?,
        timestampsEnabled: Boolean,
        voiceActivityDetectionEnabled: Boolean,
        gpuAccelerationEnabled: Boolean,
        transcriptionModelId: String,
        reuseRecordingStageResults: Boolean,
    ): String {
        val normalizedLanguage = languageCode?.takeUnless { it == "auto" }
        val job = repository.createTranscriptionJob(
            memoId = memoId,
            requestedLanguage = normalizedLanguage,
            // The worker replaces this placeholder after reading the WAV header.
            totalChunks = 1,
            transcriptionModelId = TranscriptionModel.fromId(transcriptionModelId).id,
        )
        val request = OneTimeWorkRequestBuilder<TranscriptionWorker>()
            .setInputData(
                Data.Builder()
                    .putString(TranscriptionWorker.KEY_JOB_ID, job.id)
                    .putString(TranscriptionWorker.KEY_MEMO_ID, memoId)
                    .putBoolean(
                        TranscriptionWorker.KEY_TIMESTAMPS_ENABLED,
                        timestampsEnabled,
                    )
                    .putBoolean(
                        TranscriptionWorker.KEY_VAD_ENABLED,
                        voiceActivityDetectionEnabled,
                    )
                    .putBoolean(
                        TranscriptionWorker.KEY_GPU_ACCELERATION_ENABLED,
                        gpuAccelerationEnabled,
                    )
                    .putString(
                        TranscriptionWorker.KEY_TRANSCRIPTION_MODEL_ID,
                        job.transcriptionModelId,
                    )
                    .putBoolean(
                        TranscriptionWorker.KEY_USE_PRETRANSCRIBED_WINDOWS,
                        reuseRecordingStageResults,
                    )
                    .build(),
            )
            .addTag(QUEUE_TAG)
            .addTag(memoTag(memoId))
            .build()

        workManager.beginUniqueWork(
            FIFO_WORK_NAME,
            ExistingWorkPolicy.APPEND_OR_REPLACE,
            request,
        ).enqueue()
        repository.attachWorkId(job.id, request.id)
        return job.id
    }

    override suspend fun cancel(memoId: String, jobId: String?) {
        if (jobId != null) {
            repository.getTranscriptionJob(jobId)?.let { job ->
                repository.upsertTranscriptionJob(
                    job.copy(
                        status = TranscriptionJobStatus.CANCEL_REQUESTED,
                        phase = TranscriptionPhase.PAUSED,
                        updatedAtMillis = System.currentTimeMillis(),
                    ),
                )
            }
        }
        withContext(Dispatchers.IO) {
            workManager.cancelAllWorkByTag(memoTag(memoId)).result.get()
        }
        if (jobId != null) {
            repository.getTranscriptionJob(jobId)?.let { job ->
                repository.upsertTranscriptionJob(
                    job.copy(
                        status = TranscriptionJobStatus.CANCELLED,
                        phase = TranscriptionPhase.PAUSED,
                        currentChunkIndex = null,
                        currentToken = 0,
                        errorCode = null,
                        errorMessage = null,
                        updatedAtMillis = System.currentTimeMillis(),
                    ),
                )
            }
        }
    }

    companion object {
        const val FIFO_WORK_NAME = "offline-whisper-fifo"
        const val QUEUE_TAG = "offline-whisper"
        fun memoTag(memoId: String): String = "memo:$memoId"
    }
}
