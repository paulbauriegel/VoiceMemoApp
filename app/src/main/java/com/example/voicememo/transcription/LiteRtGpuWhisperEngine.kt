package com.example.voicememo.transcription

import android.content.Context
import android.os.Trace
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import com.google.ai.edge.litert.TensorType
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Diagnostic LiteRT 2.x GPU-encoder/XNNPACK-decoder implementation.
 *
 * A fully delegated quantized GPU decoder must not be used unless it produces the
 * same greedy token stream as the reference Interpreter/XNNPACK path on the
 * supported test device. This diagnostic hybrid keeps the expensive encoder
 * on LiteRT's accelerator, then bridges its FP32 state to the reference CPU
 * decoder to evaluate whether a GPU encoder can match CPU tokens. It is not
 * selected by normal app transcription: device tests proved the encoder state
 * still changes later token choices for the pinned quantized model.
 */
internal class LiteRtGpuWhisperEngine private constructor(
    private val model: CompiledModel,
    private val cpuDecoder: LiteRtWhisperEngine,
    private val buffers: InferenceBuffers,
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
        check(!closed.get()) { "WhisperEngine is closed." }
        require(audio.size <= WhisperFeatureExtractor.MAX_SAMPLES) {
            "transcribeChunk accepts at most 30 seconds of 16 kHz PCM."
        }
        val started = System.nanoTime()

        checkCancellation(shouldCancel)
        val features = traced("WhisperGpuFeatureExtraction") {
            featureExtractor.extract(audio)
        }
        gpuOperation("writing encoder input") {
            buffers.features.writeFloat(features)
        }
        checkCancellation(shouldCancel)
        tracedGpu("WhisperGpuEncode", "encoder invocation") {
            model.run(
                mapOf(ENCODE_INPUT to buffers.features),
                mapOf(ENCODE_OUTPUT to buffers.encoderState),
                ENCODE_SIGNATURE,
            )
        }
        val encoderState = gpuOperation("reading encoder state") {
            buffers.encoderState.readFloat()
        }
        if (encoderState.size != ENCODER_STATE_FLOATS) {
            throw WhisperGpuBackendException(
                "GPU encoder returned ${encoderState.size} floats; expected " +
                    "$ENCODER_STATE_FLOATS.",
            )
        }
        checkCancellation(shouldCancel)
        cpuDecoder.transcribeEncodedChunk(
            encoderState = encoderState,
            language = language,
            shouldCancel = shouldCancel,
            onTokenProgress = onTokenProgress,
            startedNanos = started,
        )
    }

    override fun decodeTokens(tokens: List<Int>): String? = cpuDecoder.decodeTokens(tokens)

    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        synchronized(inferenceLock) {
            buffers.close()
            model.close()
            cpuDecoder.close()
        }
    }

    private fun checkCancellation(shouldCancel: () -> Boolean) {
        if (closed.get() || shouldCancel()) throw WhisperCancelledException()
    }

    private inline fun <T> traced(section: String, block: () -> T): T {
        Trace.beginSection(section)
        return try {
            block()
        } finally {
            Trace.endSection()
        }
    }

    private inline fun <T> tracedGpu(
        section: String,
        operation: String,
        block: () -> T,
    ): T = traced(section) {
        gpuOperation(operation, block)
    }

    private inline fun <T> gpuOperation(
        operation: String,
        block: () -> T,
    ): T = try {
        block()
    } catch (error: Throwable) {
        if (!WhisperBackendSelector.isRecoverableGpuFailure(error)) throw error
        if (error is WhisperGpuBackendException) throw error
        throw WhisperGpuBackendException(
            message = "LiteRT GPU $operation failed.",
            cause = error,
        )
    }

    private class InferenceBuffers private constructor(
        val features: TensorBuffer,
        val encoderState: TensorBuffer,
    ) : AutoCloseable {
        override fun close() {
            listOf(
                encoderState,
                features,
            ).forEach { buffer ->
                runCatching(buffer::close)
            }
        }

        companion object {
            fun allocate(model: CompiledModel): InferenceBuffers {
                val allocated = mutableListOf<TensorBuffer>()
                fun input(name: String, signature: String): TensorBuffer =
                    model.createInputBuffer(name, signature).also(allocated::add)
                fun output(name: String, signature: String): TensorBuffer =
                    model.createOutputBuffer(name, signature).also(allocated::add)

                return try {
                    val features = input(ENCODE_INPUT, ENCODE_SIGNATURE)
                    val encoderState = output(ENCODE_OUTPUT, ENCODE_SIGNATURE)
                    InferenceBuffers(
                        features = features,
                        encoderState = encoderState,
                    )
                } catch (error: Throwable) {
                    allocated.asReversed().forEach { buffer ->
                        runCatching(buffer::close)
                    }
                    throw error
                }
            }
        }
    }

    companion object {
        private const val ENCODE_SIGNATURE = "encode"
        private const val DECODE_SIGNATURE = "decode"
        private const val ENCODE_INPUT = "args_0"
        private const val ENCODE_OUTPUT = "output_0"
        private const val DECODE_ENCODER_STATE = "args_0"
        private const val DECODE_TOKENS = "args_1"
        private const val DECODE_MASK = "args_2"
        private const val DECODE_OUTPUT = "output_0"
        private const val ENCODER_FRAMES = 1_500
        private const val ENCODER_WIDTH = 1_280
        private const val ENCODER_STATE_FLOATS = ENCODER_FRAMES * ENCODER_WIDTH

        fun open(
            context: Context,
            cpuThreadCount: Int = LiteRtWhisperEngine.DEFAULT_THREAD_COUNT,
            timestampsEnabled: Boolean = false,
        ): LiteRtGpuWhisperEngine {
            require(cpuThreadCount > 0) { "LiteRT thread count must be positive." }
            val applicationContext = context.applicationContext
            val config = applicationContext.assets.open(WhisperAssets.GENERATION_CONFIG).use {
                WhisperGenerationConfig.load(it)
            }
            validateConfig(config)

            // The model's decode signature contains one GATHER that LiteRT's
            // GPU accelerator does not currently support. Include CPU so model
            // compilation can retain that residual operation even though this
            // diagnostic hybrid invokes only the GPU encode signature.
            val options = CompiledModel.Options(
                Accelerator.GPU,
                Accelerator.CPU,
            ).apply {
                // XNNPACK dynamically quantizes FLOAT inputs before this
                // weight-quantized model's FC operations. Enable the analogous
                // GPU source-quantized path, with the constant sharing it
                // requires, rather than comparing two different arithmetic
                // recipes. Keep all remaining floating-point calculations in FP32.
                gpuOptions = CompiledModel.GpuOptions(
                    constantTensorSharing = true,
                    allowSrcQuantizedFcConvOps = true,
                    precision = CompiledModel.GpuOptions.Precision.FP32,
                    backend = CompiledModel.GpuOptions.Backend.OPENCL,
                )
            }
            val model = CompiledModel.create(
                applicationContext.assets,
                WhisperAssets.MODEL,
                options,
            )
            var cpuDecoder: LiteRtWhisperEngine? = null
            var inferenceBuffers: InferenceBuffers? = null
            return try {
                validateModelContract(model)
                val buffers = InferenceBuffers.allocate(model)
                    .also { inferenceBuffers = it }
                val decoder = LiteRtWhisperEngine.open(
                    context = applicationContext,
                    threadCount = cpuThreadCount,
                    timestampsEnabled = timestampsEnabled,
                ).also { cpuDecoder = it }
                LiteRtGpuWhisperEngine(
                    model = model,
                    cpuDecoder = decoder,
                    buffers = buffers,
                    featureExtractor = WhisperFeatureExtractor(),
                )
            } catch (error: Throwable) {
                runCatching { cpuDecoder?.close() }
                runCatching { inferenceBuffers?.close() }
                model.close()
                throw error
            }
        }

        private fun validateConfig(config: WhisperGenerationConfig) {
            if (
                config.eosTokenId != 50_257 ||
                config.decoderStartTokenId != 50_258 ||
                config.transcribeTokenId != 50_360 ||
                config.noTimestampsTokenId != 50_364
            ) {
                throw WhisperModelContractException(
                    "generation_config.json does not match Whisper large-v3-turbo.",
                )
            }
            if (config.languageToId.isEmpty()) {
                throw WhisperModelContractException("No language tokens found.")
            }
        }

        private fun validateModelContract(model: CompiledModel) {
            validateTensor(
                label = "$ENCODE_SIGNATURE/$ENCODE_INPUT",
                tensor = model.getInputTensorType(ENCODE_INPUT, ENCODE_SIGNATURE),
                elementType = TensorType.ElementType.FLOAT,
                shape = intArrayOf(1, 128, 3_000),
            )
            validateTensor(
                label = "$ENCODE_SIGNATURE/$ENCODE_OUTPUT",
                tensor = model.getOutputTensorType(ENCODE_OUTPUT, ENCODE_SIGNATURE),
                elementType = TensorType.ElementType.FLOAT,
                shape = intArrayOf(1, 1_500, 1_280),
            )
            validateTensor(
                label = "$DECODE_SIGNATURE/$DECODE_ENCODER_STATE",
                tensor = model.getInputTensorType(DECODE_ENCODER_STATE, DECODE_SIGNATURE),
                elementType = TensorType.ElementType.FLOAT,
                shape = intArrayOf(1, 1_500, 1_280),
            )
            validateTensor(
                label = "$DECODE_SIGNATURE/$DECODE_TOKENS",
                tensor = model.getInputTensorType(DECODE_TOKENS, DECODE_SIGNATURE),
                elementType = TensorType.ElementType.INT,
                shape = intArrayOf(1, 128),
            )
            validateTensor(
                label = "$DECODE_SIGNATURE/$DECODE_MASK",
                tensor = model.getInputTensorType(DECODE_MASK, DECODE_SIGNATURE),
                elementType = TensorType.ElementType.FLOAT,
                shape = intArrayOf(1, 1, 128, 128),
            )
            validateTensor(
                label = "$DECODE_SIGNATURE/$DECODE_OUTPUT",
                tensor = model.getOutputTensorType(DECODE_OUTPUT, DECODE_SIGNATURE),
                elementType = TensorType.ElementType.FLOAT,
                shape = intArrayOf(1, 128, 51_866),
            )
        }

        private fun validateTensor(
            label: String,
            tensor: TensorType,
            elementType: TensorType.ElementType,
            shape: IntArray,
        ) {
            val actualShape = tensor.layout?.dimensions
            if (tensor.elementType != elementType || actualShape != shape.toList()) {
                throw WhisperModelContractException(
                    "$label is ${tensor.elementType} $actualShape; expected " +
                        "$elementType ${shape.contentToString()}.",
                )
            }
        }
    }
}

internal class WhisperGpuBackendException(
    message: String,
    cause: Throwable? = null,
) : RuntimeException(message, cause)
