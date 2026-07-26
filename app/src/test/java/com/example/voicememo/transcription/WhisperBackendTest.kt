package com.example.voicememo.transcription

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

class WhisperBackendTest {
    @Test
    fun gpuAccelerationPreferenceSelectsRequestedBackend() {
        assertEquals(
            WhisperBackendRequest.GPU,
            WhisperBackendRequest.fromGpuAccelerationEnabled(true),
        )
        assertEquals(
            WhisperBackendRequest.CPU,
            WhisperBackendRequest.fromGpuAccelerationEnabled(false),
        )
    }

    @Test
    fun buildValueParsing_isStrictAndCaseInsensitive() {
        assertEquals(
            WhisperBackendRequest.CPU,
            WhisperBackendRequest.fromBuildValue(" CPU "),
        )
        assertEquals(
            WhisperBackendRequest.GPU,
            WhisperBackendRequest.fromBuildValue("gpu"),
        )

        try {
            WhisperBackendRequest.fromBuildValue("nnapi")
            fail("Unknown backend should be rejected.")
        } catch (_: IllegalArgumentException) {
            // Expected.
        }
    }

    @Test
    fun cpuRequest_neverAttemptsGpu() {
        var gpuOpened = false
        val result = WhisperBackendSelector.open(
            requested = WhisperBackendRequest.CPU,
            openCpu = { "cpu" },
            openGpu = {
                gpuOpened = true
                "gpu"
            },
        )

        assertEquals("cpu", result.value)
        assertEquals(WhisperRuntimeBackend.CPU_XNNPACK, result.selection.actual)
        assertFalse(gpuOpened)
    }

    @Test
    fun gpuOpenSuccess_reportsOnlyGpu() {
        val result = WhisperBackendSelector.open(
            requested = WhisperBackendRequest.GPU,
            openCpu = { "cpu" },
            openGpu = { "gpu" },
        )

        assertEquals("gpu", result.value)
        assertEquals(
            WhisperRuntimeBackend.GPU_COMPILED_MODEL_AUTO,
            result.selection.actual,
        )
        assertEquals(null, result.selection.fallbackReason)
    }

    @Test
    fun gpuOpenFailure_opensCpuAndReportsReason() {
        val result = WhisperBackendSelector.open(
            requested = WhisperBackendRequest.GPU,
            openCpu = { "cpu" },
            openGpu = { error("GPU delegate rejected op 12") },
        )

        assertEquals("cpu", result.value)
        assertEquals(WhisperRuntimeBackend.CPU_XNNPACK, result.selection.actual)
        assertEquals(WhisperBackendFallbackStage.OPEN, result.selection.fallbackStage)
        assertEquals(
            "IllegalStateException: GPU delegate rejected op 12",
            result.selection.fallbackReason,
        )
    }

    @Test
    fun parityGuard_reportsRequestedGpuButRunsOnlyCpu() {
        val selection = WhisperBackendSelection.gpuParityGuarded()

        assertEquals(WhisperBackendRequest.GPU, selection.requested)
        assertEquals(WhisperRuntimeBackend.CPU_XNNPACK, selection.actual)
        assertEquals(
            WhisperBackendFallbackStage.PARITY_GUARD,
            selection.fallbackStage,
        )
        assertEquals(GPU_PARITY_GUARD_REASON, selection.fallbackReason)
        assertFalse(selection.actual.usesGpuAccelerator)
        assertTrue(selection.actual.usesCpuXnnpack)
    }

    @Test
    fun explicitHybridProbe_reportsBothAccelerators() {
        val result = WhisperBackendSelector.open(
            requested = WhisperBackendRequest.GPU,
            openCpu = { "cpu" },
            openGpu = { "hybrid" },
            selectedGpuBackend = WhisperRuntimeBackend.MIXED_GPU_CPU_XNNPACK,
        )

        assertEquals("hybrid", result.value)
        assertEquals(
            WhisperRuntimeBackend.MIXED_GPU_CPU_XNNPACK,
            result.selection.actual,
        )
        assertTrue(result.selection.actual.usesGpuAccelerator)
        assertTrue(result.selection.actual.usesCpuXnnpack)
    }

    @Test
    fun inferenceFingerprint_separatesSafeAndDiagnosticRuntimes() {
        val cpu = WhisperInferenceFingerprint.forConfiguration(
            requestedBackend = WhisperBackendRequest.CPU,
            gpuDecoderMode = "disabled",
            workerCount = 2,
            threadsPerWorker = 3,
        )
        val guarded = WhisperInferenceFingerprint.forConfiguration(
            requestedBackend = WhisperBackendRequest.GPU,
            gpuDecoderMode = "disabled",
            workerCount = 2,
            threadsPerWorker = 3,
        )
        val hybridProbe = WhisperInferenceFingerprint.forConfiguration(
            requestedBackend = WhisperBackendRequest.GPU,
            gpuDecoderMode = "cpu",
            workerCount = 1,
            threadsPerWorker = 3,
        )
        val fullGpuProbe = WhisperInferenceFingerprint.forConfiguration(
            requestedBackend = WhisperBackendRequest.GPU,
            gpuDecoderMode = "gpu",
            workerCount = 1,
            threadsPerWorker = 3,
        )

        assertEquals(4, setOf(cpu, guarded, hybridProbe, fullGpuProbe).size)
        listOf(cpu, guarded, hybridProbe, fullGpuProbe).forEach { fingerprint ->
            assertTrue(fingerprint.contains(WhisperAssets.MODEL_SHA256))
            assertTrue(fingerprint.contains(WhisperAssets.TOKENIZER_REVISION))
        }
        assertTrue(WhisperInferenceFingerprint.isCurrentCpuReference(cpu))
        assertTrue(WhisperInferenceFingerprint.isCurrentCpuReference(guarded))
        assertFalse(WhisperInferenceFingerprint.isCurrentCpuReference(hybridProbe))
        assertFalse(WhisperInferenceFingerprint.isCurrentCpuReference(fullGpuProbe))
        assertFalse(WhisperInferenceFingerprint.isCurrentCpuReference(null))
        assertFalse(
            WhisperInferenceFingerprint.isCurrentCpuReference(
                cpu.replace(WhisperAssets.MODEL_SHA256, "stale-model"),
            ),
        )

        val timestamped = WhisperInferenceFingerprint.forConfiguration(
            requestedBackend = WhisperBackendRequest.CPU,
            gpuDecoderMode = "disabled",
            workerCount = 2,
            threadsPerWorker = 3,
            timestampsEnabled = true,
        )
        assertFalse(cpu == timestamped)
        assertTrue(timestamped.contains("timestamps=on"))

        val vadDisabled = WhisperInferenceFingerprint.forConfiguration(
            requestedBackend = WhisperBackendRequest.CPU,
            gpuDecoderMode = "disabled",
            workerCount = 2,
            threadsPerWorker = 3,
            voiceActivityDetectionEnabled = false,
        )
        assertFalse(cpu == vadDisabled)
        assertTrue(vadDisabled.contains("vad=off"))
    }

    @Test
    fun fatalGpuOpenFailure_isNotHiddenByFallback() {
        val fatal = OutOfMemoryError("model allocation")
        var cpuOpened = false

        try {
            WhisperBackendSelector.open(
                requested = WhisperBackendRequest.GPU,
                openCpu = {
                    cpuOpened = true
                    "cpu"
                },
                openGpu = { throw fatal },
            )
            fail("Fatal VM errors must be propagated.")
        } catch (actual: OutOfMemoryError) {
            assertSame(fatal, actual)
        }
        assertFalse(cpuOpened)
    }

    @Test
    fun firstGpuInvocationFailure_retriesOnCpuAndReportsCpu() {
        val gpu = FakeEngine { throw gpuFailure("unsupported decoder op") }
        val cpu = FakeEngine { chunkResult("cpu") }
        val selections = mutableListOf<WhisperBackendSelection>()
        val engine = RuntimeFallbackWhisperEngine(
            initialGpuEngine = gpu,
            cpuFactory = { cpu },
            onBackendSelectionChanged = selections::add,
        )

        val result = engine.transcribeChunk(
            audio = FloatArray(1),
            language = WhisperLanguage.Auto,
        )

        assertEquals("cpu", result.text)
        assertTrue(gpu.closed)
        assertEquals(WhisperRuntimeBackend.UNAVAILABLE, selections.first().actual)
        assertEquals(WhisperRuntimeBackend.CPU_XNNPACK, selections.last().actual)
        assertEquals(
            "IllegalStateException: unsupported decoder op",
            selections.last().fallbackReason,
        )

        engine.close()
        assertTrue(cpu.closed)
    }

    @Test
    fun laterGpuInvocationFailure_reportsMixedAfterCompletedGpuChunk() {
        val gpu = FakeEngine { invocation ->
            if (invocation == 1) {
                chunkResult("gpu")
            } else {
                throw gpuFailure("driver lost context")
            }
        }
        val cpu = FakeEngine { chunkResult("cpu") }
        val selections = mutableListOf<WhisperBackendSelection>()
        val engine = RuntimeFallbackWhisperEngine(
            initialGpuEngine = gpu,
            cpuFactory = { cpu },
            onBackendSelectionChanged = selections::add,
        )

        assertEquals(
            "gpu",
            engine.transcribeChunk(FloatArray(1), WhisperLanguage.Auto).text,
        )
        assertEquals(
            "cpu",
            engine.transcribeChunk(FloatArray(1), WhisperLanguage.Auto).text,
        )

        assertEquals(
            WhisperRuntimeBackend.MIXED_GPU_CPU_XNNPACK,
            selections.last().actual,
        )
        assertEquals(WhisperBackendFallbackStage.INFERENCE, selections.last().fallbackStage)
        engine.close()
    }

    private class FakeEngine(
        private val transcribe: (invocation: Int) -> WhisperChunkResult,
    ) : WhisperEngine {
        var closed = false
            private set
        private var invocations = 0

        override fun transcribeChunk(
            audio: FloatArray,
            language: WhisperLanguage,
            shouldCancel: () -> Boolean,
            onTokenProgress: (generated: Int, limit: Int) -> Unit,
        ): WhisperChunkResult = transcribe(++invocations)

        override fun close() {
            closed = true
        }
    }

    private fun chunkResult(text: String) = WhisperChunkResult(
        tokens = emptyList(),
        text = text,
        detectedLanguage = "en",
        inferenceMillis = 1L,
        truncated = false,
    )

    private fun gpuFailure(detail: String) = WhisperGpuBackendException(
        message = "GPU invocation failed.",
        cause = IllegalStateException(detail),
    )
}
