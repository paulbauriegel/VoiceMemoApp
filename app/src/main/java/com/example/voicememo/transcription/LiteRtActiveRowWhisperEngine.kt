package com.example.voicememo.transcription

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Research-only engine using the normal GPU encoder, a decoder core ending at
 * final LayerNorm, and an FP32 XNNPACK projection of only the active row.
 */
internal class LiteRtActiveRowWhisperEngine private constructor(
    private val encoderModel: CompiledModel,
    private val features: TensorBuffer,
    private val encoderState: TensorBuffer,
    private val decoder: LiteRtSplitDecoderDiagnostic,
    private val featureExtractor: WhisperFeatureExtractor,
) : WhisperEngine {
    private val closed = AtomicBoolean(false)
    private val inferenceLock = Any()

    override fun transcribeChunk(
        audio: FloatArray,
        language: WhisperLanguage,
        shouldCancel: () -> Boolean,
        onTokenProgress: (generated: Int, limit: Int) -> Unit,
    ): WhisperChunkResult = synchronized(inferenceLock) {
        check(!closed.get()) { "Active-row Whisper engine is closed." }
        require(language is WhisperLanguage.Forced) {
            "The active-row benchmark requires a forced language."
        }
        require(audio.size <= WhisperFeatureExtractor.MAX_SAMPLES) {
            "transcribeChunk accepts at most 30 seconds of 16 kHz PCM."
        }
        if (shouldCancel()) throw WhisperCancelledException()
        val started = System.nanoTime()
        features.writeFloat(featureExtractor.extract(audio))
        encoderModel.run(
            mapOf(ENCODE_INPUT to features),
            mapOf(ENCODE_OUTPUT to encoderState),
            ENCODE_SIGNATURE,
        )
        if (shouldCancel()) throw WhisperCancelledException()
        val state = encoderState.readFloat()
        require(state.size == WhisperSplitDecoderTensors.ENCODER_STATE_FLOATS) {
            "GPU encoder returned ${state.size} floats."
        }
        decoder.transcribeEncodedChunk(
            encoderState = state,
            language = language,
            shouldCancel = shouldCancel,
            onTokenProgress = onTokenProgress,
            startedNanos = started,
        )
    }

    override fun decodeTokens(tokens: List<Int>): String = decoder.decodeTokens(tokens)

    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        synchronized(inferenceLock) {
            runCatching(decoder::close)
            runCatching(encoderState::close)
            runCatching(features::close)
            encoderModel.close()
        }
    }

    companion object {
        private const val ENCODE_SIGNATURE = "encode"
        private const val ENCODE_INPUT = "args_0"
        private const val ENCODE_OUTPUT = "output_0"

        fun open(
            context: Context,
            encoderModelPath: String,
            decoderCorePath: String,
            activeRowProjectionPath: String,
            encoderPrecision: CompiledModel.GpuOptions.Precision,
            decoderPrecision: CompiledModel.GpuOptions.Precision,
        ): LiteRtActiveRowWhisperEngine {
            fun gpuOptions(precision: CompiledModel.GpuOptions.Precision) =
                CompiledModel.Options(Accelerator.GPU, Accelerator.CPU).apply {
                    gpuOptions = CompiledModel.GpuOptions(
                        constantTensorSharing = true,
                        allowSrcQuantizedFcConvOps = false,
                        precision = precision,
                        backend = CompiledModel.GpuOptions.Backend.OPENCL,
                    )
                }
            val encoder = CompiledModel.create(encoderModelPath, gpuOptions(encoderPrecision))
            val allocated = mutableListOf<TensorBuffer>()
            var split: LiteRtSplitDecoderDiagnostic? = null
            return try {
                val features = encoder.createInputBuffer(ENCODE_INPUT, ENCODE_SIGNATURE)
                    .also(allocated::add)
                val encoderState = encoder.createOutputBuffer(ENCODE_OUTPUT, ENCODE_SIGNATURE)
                    .also(allocated::add)
                split = LiteRtSplitDecoderDiagnostic.openFiles(
                    configurationAssets = context.applicationContext.assets,
                    decoderCorePath = decoderCorePath,
                    activeRowProjectionPath = activeRowProjectionPath,
                    coreBackend = WhisperSplitDecoderCoreBackend.GPU_COMPILED_MODEL,
                    allowSourceQuantizedFcConvOps = false,
                    coreGpuPrecision = decoderPrecision,
                )
                LiteRtActiveRowWhisperEngine(
                    encoderModel = encoder,
                    features = features,
                    encoderState = encoderState,
                    decoder = split,
                    featureExtractor = WhisperFeatureExtractor(),
                )
            } catch (error: Throwable) {
                runCatching { split?.close() }
                allocated.asReversed().forEach { buffer -> runCatching(buffer::close) }
                encoder.close()
                throw error
            }
        }
    }
}
