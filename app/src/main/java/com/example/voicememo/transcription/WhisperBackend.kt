package com.example.voicememo.transcription

import android.content.Context
import java.util.concurrent.CancellationException
import java.util.concurrent.atomic.AtomicBoolean

enum class WhisperBackendRequest(
    val buildValue: String,
    val reportValue: String,
) {
    CPU(buildValue = "cpu", reportValue = "CPU"),
    GPU(buildValue = "gpu", reportValue = "GPU"),
    ;

    companion object {
        fun fromBuildValue(value: String): WhisperBackendRequest =
            entries.singleOrNull { it.buildValue == value.trim().lowercase() }
                ?: throw IllegalArgumentException(
                    "Unknown Whisper backend '$value'; expected cpu or gpu.",
                )

        fun fromGpuAccelerationEnabled(enabled: Boolean): WhisperBackendRequest =
            if (enabled) GPU else CPU
    }
}

enum class WhisperRuntimeBackend(
    val reportValue: String,
    val usesGpuAccelerator: Boolean,
    val usesCpuXnnpack: Boolean,
) {
    UNAVAILABLE(
        reportValue = "UNAVAILABLE",
        usesGpuAccelerator = false,
        usesCpuXnnpack = false,
    ),
    CPU_XNNPACK(
        reportValue = "CPU_XNNPACK",
        usesGpuAccelerator = false,
        usesCpuXnnpack = true,
    ),
    GPU_COMPILED_MODEL_AUTO(
        reportValue = "GPU_COMPILED_MODEL_AUTO",
        usesGpuAccelerator = true,
        usesCpuXnnpack = false,
    ),
    MIXED_GPU_CPU_XNNPACK(
        reportValue = "MIXED_GPU_CPU_XNNPACK",
        usesGpuAccelerator = true,
        usesCpuXnnpack = true,
    ),
}

enum class WhisperBackendFallbackStage(
    val reportValue: String,
) {
    PARITY_GUARD("PARITY_GUARD"),
    OPEN("OPEN"),
    INFERENCE("INFERENCE"),
}

data class WhisperBackendSelection(
    val requested: WhisperBackendRequest,
    val actual: WhisperRuntimeBackend,
    val fallbackStage: WhisperBackendFallbackStage? = null,
    val fallbackReason: String? = null,
) {
    init {
        require((fallbackStage == null) == (fallbackReason == null)) {
            "A backend fallback stage and reason must be reported together."
        }
    }

    companion object {
        fun pending(requested: WhisperBackendRequest) = WhisperBackendSelection(
            requested = requested,
            actual = WhisperRuntimeBackend.UNAVAILABLE,
        )

        fun cpuRequested() = WhisperBackendSelection(
            requested = WhisperBackendRequest.CPU,
            actual = WhisperRuntimeBackend.CPU_XNNPACK,
        )

        fun gpuSelected(
            actual: WhisperRuntimeBackend = WhisperRuntimeBackend.GPU_COMPILED_MODEL_AUTO,
        ) = WhisperBackendSelection(
            requested = WhisperBackendRequest.GPU,
            actual = actual,
        )

        fun gpuFallback(
            actual: WhisperRuntimeBackend,
            stage: WhisperBackendFallbackStage,
            reason: String,
        ) = WhisperBackendSelection(
            requested = WhisperBackendRequest.GPU,
            actual = actual,
            fallbackStage = stage,
            fallbackReason = reason,
        )

        fun gpuParityGuarded() = gpuFallback(
            actual = WhisperRuntimeBackend.CPU_XNNPACK,
            stage = WhisperBackendFallbackStage.PARITY_GUARD,
            reason = GPU_PARITY_GUARD_REASON,
        )
    }
}

internal const val GPU_PARITY_GUARD_REASON =
    "The pinned Whisper model failed exact GPU token parity; using CPU/XNNPACK."

internal object WhisperInferenceFingerprint {
    private const val PIPELINE_REVISION =
        "frontend-mel-v1_longform-silero-vad-v6.2.1_overlap-merge-v1_greedy-v2"
    private const val CPU_RUNTIME_PREFIX = "cpu-xnnpack-"
    private const val GUARDED_CPU_RUNTIME_PREFIX =
        "gpu-parity-guard_cpu-xnnpack-"

    fun forConfiguration(
        requestedBackend: WhisperBackendRequest,
        gpuDecoderMode: String,
        workerCount: Int,
        threadsPerWorker: Int,
        timestampsEnabled: Boolean = false,
        voiceActivityDetectionEnabled: Boolean = true,
    ): String {
        require(workerCount > 0)
        require(threadsPerWorker > 0)
        val runtime = when {
            requestedBackend == WhisperBackendRequest.CPU ->
                "cpu-xnnpack-${workerCount}x$threadsPerWorker"
            gpuDecoderMode == "disabled" ->
                "gpu-parity-guard_cpu-xnnpack-${workerCount}x$threadsPerWorker"
            gpuDecoderMode == "cpu" ->
                "gpu-encoder-qd8-fp32-opencl_cpu-decoder-xnnpack-$threadsPerWorker"
            gpuDecoderMode == "gpu" ->
                "gpu-full-float-decoder-fp32-opencl_no-source-quant_cpu-gather"
            else -> throw IllegalArgumentException(
                "Unknown GPU decoder mode '$gpuDecoderMode'.",
            )
        }
        return listOf(
            "model=${WhisperAssets.MODEL_SHA256}",
            "tokenizer=${WhisperAssets.TOKENIZER_REVISION}",
            "pipeline=$PIPELINE_REVISION",
            "runtime=$runtime",
            "timestamps=${if (timestampsEnabled) "on" else "off"}",
            "vad=${if (voiceActivityDetectionEnabled) "on" else "off"}",
        ).joinToString("|")
    }

    /**
     * Diagnostic parity reports may use only output produced by the current
     * model, tokenizer, pipeline, and the trusted Interpreter/XNNPACK runtime.
     * A merely successful historical job is not a valid CPU oracle.
     */
    fun isCurrentCpuReference(fingerprint: String?): Boolean {
        val fields = fingerprint
            ?.split('|')
            ?.mapNotNull { field ->
                val separator = field.indexOf('=')
                if (separator <= 0) null else {
                    field.substring(0, separator) to field.substring(separator + 1)
                }
            }
            ?.toMap()
            ?: return false
        val runtime = fields["runtime"] ?: return false
        return fields["model"] == WhisperAssets.MODEL_SHA256 &&
            fields["tokenizer"] == WhisperAssets.TOKENIZER_REVISION &&
            fields["pipeline"] == PIPELINE_REVISION &&
            (
                runtime.startsWith(CPU_RUNTIME_PREFIX) ||
                    runtime.startsWith(GUARDED_CPU_RUNTIME_PREFIX)
                )
    }
}

internal data class WhisperBackendOpenResult<T>(
    val value: T,
    val selection: WhisperBackendSelection,
)

internal object WhisperBackendSelector {
    fun <T> open(
        requested: WhisperBackendRequest,
        openCpu: () -> T,
        openGpu: () -> T,
        selectedGpuBackend: WhisperRuntimeBackend =
            WhisperRuntimeBackend.GPU_COMPILED_MODEL_AUTO,
    ): WhisperBackendOpenResult<T> = when (requested) {
        WhisperBackendRequest.CPU -> WhisperBackendOpenResult(
            value = openCpu(),
            selection = WhisperBackendSelection.cpuRequested(),
        )

        WhisperBackendRequest.GPU -> {
            try {
                WhisperBackendOpenResult(
                    value = openGpu(),
                    selection = WhisperBackendSelection.gpuSelected(selectedGpuBackend),
                )
            } catch (gpuFailure: Throwable) {
                if (!isRecoverableGpuFailure(gpuFailure)) throw gpuFailure
                val reason = failureReason(gpuFailure)
                val cpu = try {
                    openCpu()
                } catch (cpuFailure: Throwable) {
                    cpuFailure.addSuppressed(gpuFailure)
                    throw cpuFailure
                }
                WhisperBackendOpenResult(
                    value = cpu,
                    selection = WhisperBackendSelection.gpuFallback(
                        actual = WhisperRuntimeBackend.CPU_XNNPACK,
                        stage = WhisperBackendFallbackStage.OPEN,
                        reason = reason,
                    ),
                )
            }
        }
    }

    fun isRecoverableGpuFailure(error: Throwable): Boolean = when (error) {
        is WhisperCancelledException,
        is CancellationException,
        is InterruptedException,
        is VirtualMachineError,
        is ThreadDeath,
        -> false

        else -> true
    }

    fun failureReason(error: Throwable): String {
        val reported = if (error is WhisperGpuBackendException) {
            error.cause ?: error
        } else {
            error
        }
        val type = reported.javaClass.simpleName.ifBlank { reported.javaClass.name }
        val detail = reported.message
            ?.lineSequence()
            ?.firstOrNull()
            ?.trim()
            ?.take(MAX_REASON_CHARACTERS)
            ?.takeIf(String::isNotEmpty)
        return if (detail == null) type else "$type: $detail"
    }

    private const val MAX_REASON_CHARACTERS = 240
}

/**
 * Opens the LiteRT 2.x GPU backend while keeping Interpreter/XNNPACK as both
 * creation- and run-time fallback.
 */
internal object LiteRtWhisperBackendFactory {
    fun openExperimentalGpuWithCpuFallback(
        context: Context,
        cpuThreadCount: Int = LiteRtWhisperEngine.DEFAULT_THREAD_COUNT,
        useFullGpuDecoder: Boolean = false,
        allowSourceQuantizedFcConvOps: Boolean = true,
        timestampsEnabled: Boolean = false,
        onBackendSelectionChanged: (WhisperBackendSelection) -> Unit = {},
    ): WhisperEngine {
        require(cpuThreadCount > 0) { "LiteRT thread count must be positive." }
        val applicationContext = context.applicationContext
        onBackendSelectionChanged(
            WhisperBackendSelection.pending(WhisperBackendRequest.GPU),
        )

        val cpuFactory: () -> WhisperEngine = {
            LiteRtWhisperEngine.open(
                context = applicationContext,
                threadCount = cpuThreadCount,
                timestampsEnabled = timestampsEnabled,
            )
        }
        val opened = WhisperBackendSelector.open(
            requested = WhisperBackendRequest.GPU,
            openCpu = cpuFactory,
            openGpu = {
                if (useFullGpuDecoder) {
                    LiteRtFullGpuWhisperEngine.open(
                        context = applicationContext,
                        allowSourceQuantizedFcConvOps =
                            allowSourceQuantizedFcConvOps,
                        timestampsEnabled = timestampsEnabled,
                    )
                } else {
                    LiteRtGpuWhisperEngine.open(
                        context = applicationContext,
                        cpuThreadCount = cpuThreadCount,
                        timestampsEnabled = timestampsEnabled,
                    )
                }
            },
            selectedGpuBackend = if (useFullGpuDecoder) {
                // LiteRT retains the decoder token GATHER on CPU.
                WhisperRuntimeBackend.MIXED_GPU_CPU_XNNPACK
            } else {
                WhisperRuntimeBackend.MIXED_GPU_CPU_XNNPACK
            },
        )
        onBackendSelectionChanged(opened.selection)
        return if (opened.selection.actual.usesGpuAccelerator) {
            RuntimeFallbackWhisperEngine(
                initialGpuEngine = opened.value,
                cpuFactory = cpuFactory,
                onBackendSelectionChanged = onBackendSelectionChanged,
            )
        } else {
            opened.value
        }
    }
}

internal class RuntimeFallbackWhisperEngine(
    initialGpuEngine: WhisperEngine,
    private val cpuFactory: () -> WhisperEngine,
    private val onBackendSelectionChanged: (WhisperBackendSelection) -> Unit,
) : WhisperEngine {
    private val closed = AtomicBoolean(false)
    private val stateLock = Any()
    private var activeEngine: WhisperEngine = initialGpuEngine
    private var gpuEngine: WhisperEngine? = initialGpuEngine
    private var successfulGpuChunks = 0

    override fun transcribeChunk(
        audio: FloatArray,
        language: WhisperLanguage,
        shouldCancel: () -> Boolean,
        onTokenProgress: (generated: Int, limit: Int) -> Unit,
    ): WhisperChunkResult {
        val selected = synchronized(stateLock) {
            check(!closed.get()) { "WhisperEngine is closed." }
            activeEngine
        }
        return try {
            val result = selected.transcribeChunk(
                audio = audio,
                language = language,
                shouldCancel = shouldCancel,
                onTokenProgress = onTokenProgress,
            )
            synchronized(stateLock) {
                if (selected === gpuEngine) successfulGpuChunks++
            }
            result
        } catch (gpuFailure: Throwable) {
            val currentGpu = synchronized(stateLock) { gpuEngine }
            if (
                selected !== currentGpu ||
                gpuFailure !is WhisperGpuBackendException ||
                shouldCancel() ||
                !WhisperBackendSelector.isRecoverableGpuFailure(gpuFailure)
            ) {
                throw gpuFailure
            }
            switchToCpu(gpuFailure).transcribeChunk(
                audio = audio,
                language = language,
                shouldCancel = shouldCancel,
                onTokenProgress = onTokenProgress,
            )
        }
    }

    override fun decodeTokens(tokens: List<Int>): String? =
        synchronized(stateLock) {
            check(!closed.get()) { "WhisperEngine is closed." }
            activeEngine.decodeTokens(tokens)
        }

    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        val engine = synchronized(stateLock) {
            gpuEngine = null
            activeEngine
        }
        engine.close()
    }

    private fun switchToCpu(gpuFailure: Throwable): WhisperEngine {
        val reason = WhisperBackendSelector.failureReason(gpuFailure)
        synchronized(stateLock) {
            check(!closed.get()) { "WhisperEngine is closed." }
            if (gpuEngine == null) return activeEngine

            onBackendSelectionChanged(
                WhisperBackendSelection.gpuFallback(
                    actual = WhisperRuntimeBackend.UNAVAILABLE,
                    stage = WhisperBackendFallbackStage.INFERENCE,
                    reason = reason,
                ),
            )
            runCatching { gpuEngine?.close() }
                .exceptionOrNull()
                ?.let(gpuFailure::addSuppressed)
            gpuEngine = null

            val cpuEngine = try {
                cpuFactory()
            } catch (cpuFailure: Throwable) {
                cpuFailure.addSuppressed(gpuFailure)
                throw cpuFailure
            }
            activeEngine = cpuEngine
            val actualBackend = if (successfulGpuChunks == 0) {
                WhisperRuntimeBackend.CPU_XNNPACK
            } else {
                WhisperRuntimeBackend.MIXED_GPU_CPU_XNNPACK
            }
            onBackendSelectionChanged(
                WhisperBackendSelection.gpuFallback(
                    actual = actualBackend,
                    stage = WhisperBackendFallbackStage.INFERENCE,
                    reason = reason,
                ),
            )
            return cpuEngine
        }
    }
}
