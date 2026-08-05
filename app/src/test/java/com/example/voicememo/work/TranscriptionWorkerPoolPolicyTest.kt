package com.example.voicememo.work

import com.example.voicememo.transcription.TranscriptionModel
import com.example.voicememo.transcription.WhisperBackendRequest
import org.junit.Assert.assertEquals
import org.junit.Test

class TranscriptionWorkerPoolPolicyTest {
    @Test
    fun gpuAlwaysUsesExactlyOneEngine() {
        assertEquals(
            1,
            transcriptionWorkerCount(
                transcriptionModel = TranscriptionModel.WHISPER_LARGE_V3_TURBO,
                requestedBackend = WhisperBackendRequest.GPU,
                usePreTranscribedWindows = false,
                configuredCpuWorkerCount = 2,
            ),
        )
    }

    @Test
    fun cpuCanUseConfiguredParallelEnginePool() {
        assertEquals(
            2,
            transcriptionWorkerCount(
                transcriptionModel = TranscriptionModel.WHISPER_LARGE_V3_TURBO,
                requestedBackend = WhisperBackendRequest.CPU,
                usePreTranscribedWindows = false,
                configuredCpuWorkerCount = 2,
            ),
        )
    }

    @Test
    fun stagedWindowHandoffUsesOneCpuEngine() {
        assertEquals(
            1,
            transcriptionWorkerCount(
                transcriptionModel = TranscriptionModel.WHISPER_LARGE_V3_TURBO,
                requestedBackend = WhisperBackendRequest.CPU,
                usePreTranscribedWindows = true,
                configuredCpuWorkerCount = 2,
            ),
        )
    }
}
