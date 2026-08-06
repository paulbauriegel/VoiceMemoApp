package com.example.voicememo.transcription

import android.content.res.AssetManager
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import com.google.ai.edge.litert.TensorType
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Backend used only by the split-decoder parity diagnostic.
 *
 * Neither option participates in normal backend selection. In particular,
 * choosing [GPU_COMPILED_MODEL] here cannot make production transcription use
 * the generated diagnostic model.
 */
internal enum class WhisperSplitDecoderCoreBackend {
    CPU_XNNPACK,
    GPU_COMPILED_MODEL,
}

/**
 * GPU implementation requested by the split-decoder diagnostic.
 *
 * This intentionally mirrors LiteRT's public GPU backend choices without
 * exposing LiteRT option construction to instrumentation callers.
 */
enum class WhisperSplitDecoderGpuBackend {
    AUTOMATIC,
    OPENCL,
    WEBGPU,
    OPENGL,
}

/**
 * Pure description of the GPU options selected for a diagnostic run.
 *
 * Keeping selection separate from [CompiledModel.Options] makes the exact
 * accelerator order and backend mapping locally testable without opening a
 * model.
 */
internal data class WhisperSplitDecoderGpuOptionSelection(
    val accelerators: List<Accelerator>,
    val constantTensorSharing: Boolean,
    val allowSourceQuantizedFcConvOps: Boolean,
    val backend: CompiledModel.GpuOptions.Backend,
)

internal fun selectWhisperSplitDecoderGpuOptions(
    constantTensorSharing: Boolean = true,
    allowSourceQuantizedFcConvOps: Boolean = true,
    gpuBackend: WhisperSplitDecoderGpuBackend = WhisperSplitDecoderGpuBackend.OPENCL,
    strictGpuOnly: Boolean = false,
): WhisperSplitDecoderGpuOptionSelection =
    WhisperSplitDecoderGpuOptionSelection(
        accelerators = if (strictGpuOnly) {
            listOf(Accelerator.GPU)
        } else {
            listOf(Accelerator.GPU, Accelerator.CPU)
        },
        constantTensorSharing = constantTensorSharing,
        allowSourceQuantizedFcConvOps = allowSourceQuantizedFcConvOps,
        backend = gpuBackend.toLiteRtBackend(),
    )

private fun WhisperSplitDecoderGpuBackend.toLiteRtBackend():
    CompiledModel.GpuOptions.Backend =
    when (this) {
        WhisperSplitDecoderGpuBackend.AUTOMATIC ->
            CompiledModel.GpuOptions.Backend.AUTOMATIC
        WhisperSplitDecoderGpuBackend.OPENCL ->
            CompiledModel.GpuOptions.Backend.OPENCL
        WhisperSplitDecoderGpuBackend.WEBGPU ->
            CompiledModel.GpuOptions.Backend.WEBGPU
        WhisperSplitDecoderGpuBackend.OPENGL ->
            CompiledModel.GpuOptions.Backend.OPENGL
    }

/**
 * Asset paths supplied by an instrumentation-only [AssetManager].
 *
 * The generated models remain under build/ by default and are never copied
 * over the pinned production model. A diagnostic APK may package them under
 * [DEFAULT_DIRECTORY] and pass its own AssetManager to [open].
 */
internal data class WhisperSplitDecoderAssets(
    val decoderCore: String,
    val activeRowProjection: String,
) {
    companion object {
        const val DEFAULT_DIRECTORY = "whisper-decoder-split"

        val Default = WhisperSplitDecoderAssets(
            decoderCore =
                "$DEFAULT_DIRECTORY/whisper_large_v3_turbo_30s_i4_decode_hidden.tflite",
            activeRowProjection =
                "$DEFAULT_DIRECTORY/whisper_large_v3_turbo_active_row_projection_i8.tflite",
        )
    }
}

/**
 * One forced decoder step split at the final vocabulary projection.
 *
 * The hidden row is retained so a parity report can independently compare the
 * decoder core and the CPU projection. Token text is deliberately absent.
 */
internal data class WhisperSplitDecoderStep(
    val promptLength: Int,
    val activeHiddenRow: FloatArray,
    val rankedLogits: WhisperTeacherForcedLogitsRow,
    val decoderCoreMillis: Long,
    val projectionMillis: Long,
)

/**
 * Diagnostic runtime for the generated decoder split artifacts.
 *
 * The patched decoder core produces `[1, 128, 1280]`. Only the row that
 * predicts the next token is copied into the separate `[1, 1280] -> [1,
 * 51866]` projection, which always runs through Interpreter/XNNPACK with three
 * threads. This isolates GPU decoder-core drift from the terminal INT8
 * vocabulary projection while preserving the CPU greedy-token policy.
 */
internal class LiteRtSplitDecoderDiagnostic private constructor(
    private val decoderCore: DecoderCore,
    private val projection: ActiveRowProjection,
    private val tokenizer: WhisperTokenizer,
    private val generationConfig: WhisperGenerationConfig,
    val coreBackend: WhisperSplitDecoderCoreBackend,
) : AutoCloseable {
    private val closed = AtomicBoolean(false)
    private val inferenceLock = Any()

    fun transcribeEncodedChunk(
        encoderState: FloatArray,
        language: WhisperLanguage.Forced,
        shouldCancel: () -> Boolean = { false },
        onTokenProgress: (generated: Int, limit: Int) -> Unit = { _, _ -> },
        startedNanos: Long = System.nanoTime(),
    ): WhisperChunkResult {
        val generated = mutableListOf<Int>()
        var reachedEos = false
        val generationLimit =
            WhisperSplitDecoderTensors.TARGET_LENGTH -
                WhisperSplitDecoderTensors.STANDARD_PROMPT_LENGTH
        onTokenProgress(0, generationLimit)
        while (generated.size < generationLimit) {
            val step = teacherForcedStep(
                encoderState = encoderState,
                language = language,
                referenceTokenPrefix = generated,
                shouldCancel = shouldCancel,
            )
            val nextToken = step.rankedLogits.selectableTop1?.tokenId
                ?: throw WhisperModelContractException(
                    "Active-row decoder produced no selectable token.",
                )
            if (nextToken == generationConfig.eosTokenId) {
                reachedEos = true
                break
            }
            generated += nextToken
            onTokenProgress(generated.size, generationLimit)
        }
        val truncated = !reachedEos && generated.size >= generationLimit
        return WhisperChunkResult(
            tokens = generated,
            text = tokenizer.decode(generated).trim(),
            detectedLanguage = language.code,
            inferenceMillis = (System.nanoTime() - startedNanos) / NANOS_PER_MILLISECOND,
            truncated = truncated,
            warnings = if (truncated) {
                listOf("Decoder reached its 128-token limit before EOS.")
            } else {
                emptyList()
            },
        )
    }

    fun decodeTokens(tokens: List<Int>): String = tokenizer.decode(tokens)

    fun teacherForcedStep(
        encoderState: FloatArray,
        language: WhisperLanguage.Forced,
        referenceTokenPrefix: List<Int>,
        shouldCancel: () -> Boolean = { false },
    ): WhisperSplitDecoderStep = synchronized(inferenceLock) {
        check(!closed.get()) { "Split-decoder diagnostic is closed." }
        require(encoderState.size == WhisperSplitDecoderTensors.ENCODER_STATE_FLOATS) {
            "Whisper encoder state has ${encoderState.size} floats; expected " +
                "${WhisperSplitDecoderTensors.ENCODER_STATE_FLOATS}."
        }
        checkCancellation(shouldCancel)
        val prompt = WhisperSplitDecoderTensors.forcedPrompt(
            generationConfig = generationConfig,
            language = language,
            referenceTokenPrefix = referenceTokenPrefix,
        )
        val paddedTokens = WhisperSplitDecoderTensors.paddedTokens(
            prompt = prompt,
            paddingToken = generationConfig.eosTokenId,
        )

        val coreStarted = System.nanoTime()
        val hidden = decoderCore.decode(
            encoderState = encoderState,
            tokens = paddedTokens,
            shouldCancel = shouldCancel,
        )
        val coreMillis = (System.nanoTime() - coreStarted) / NANOS_PER_MILLISECOND
        checkCancellation(shouldCancel)
        val activeHidden = WhisperSplitDecoderTensors.activeHiddenRow(
            decoderHidden = hidden,
            promptLength = prompt.size,
        )

        val projectionStarted = System.nanoTime()
        val logits = projection.project(
            activeHidden = activeHidden,
            shouldCancel = shouldCancel,
        )
        val projectionMillis =
            (System.nanoTime() - projectionStarted) / NANOS_PER_MILLISECOND
        checkCancellation(shouldCancel)

        WhisperSplitDecoderStep(
            promptLength = prompt.size,
            activeHiddenRow = activeHidden,
            rankedLogits = WhisperTeacherForcedTokenRanking.forCpu(
                logits = logits,
                generationConfig = generationConfig,
                firstGeneratedToken = referenceTokenPrefix.isEmpty(),
            ),
            decoderCoreMillis = coreMillis,
            projectionMillis = projectionMillis,
        )
    }

    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        synchronized(inferenceLock) {
            runCatching(projection::close)
            decoderCore.close()
        }
    }

    private fun checkCancellation(shouldCancel: () -> Boolean) {
        if (closed.get() || shouldCancel()) throw WhisperCancelledException()
    }

    companion object {
        private const val NANOS_PER_MILLISECOND = 1_000_000L
        private const val PROJECTION_THREAD_COUNT = 3

        /**
         * [configurationAssets] normally comes from the target app so the
         * pinned generation config remains authoritative. [diagnosticAssets]
         * may come from the instrumentation APK that packages the generated
         * split artifacts. [allowSourceQuantizedFcConvOps] exposes LiteRT's
         * diagnostic source-quantization switch only for the GPU core.
         * [constantTensorSharing], [gpuBackend], and [strictGpuOnly] expose
         * additional diagnostic-only GPU compilation choices. Their defaults
         * retain the original OpenCL GPU+CPU configuration.
         */
        fun open(
            configurationAssets: AssetManager,
            diagnosticAssets: AssetManager = configurationAssets,
            assets: WhisperSplitDecoderAssets = WhisperSplitDecoderAssets.Default,
            coreBackend: WhisperSplitDecoderCoreBackend,
            coreCpuThreadCount: Int = LiteRtWhisperEngine.DEFAULT_THREAD_COUNT,
            allowSourceQuantizedFcConvOps: Boolean = true,
            constantTensorSharing: Boolean = true,
            gpuBackend: WhisperSplitDecoderGpuBackend =
                WhisperSplitDecoderGpuBackend.OPENCL,
            strictGpuOnly: Boolean = false,
            coreGpuPrecision: CompiledModel.GpuOptions.Precision =
                CompiledModel.GpuOptions.Precision.FP32,
        ): LiteRtSplitDecoderDiagnostic {
            require(coreCpuThreadCount > 0) {
                "Split decoder CPU thread count must be positive."
            }
            val generationConfig =
                configurationAssets.open(WhisperAssets.GENERATION_CONFIG).use {
                    WhisperGenerationConfig.load(it)
                }
            val tokenizer = configurationAssets.open(WhisperAssets.TOKENIZER).use {
                WhisperTokenizer.load(it)
            }
            validateGenerationConfig(generationConfig)

            val decoderCore = when (coreBackend) {
                WhisperSplitDecoderCoreBackend.CPU_XNNPACK ->
                    InterpreterDecoderCore.open(
                        assetManager = diagnosticAssets,
                        assetPath = assets.decoderCore,
                        threadCount = coreCpuThreadCount,
                    )

                WhisperSplitDecoderCoreBackend.GPU_COMPILED_MODEL ->
                    GpuDecoderCore.open(
                        assetManager = diagnosticAssets,
                        assetPath = assets.decoderCore,
                        constantTensorSharing = constantTensorSharing,
                        allowSourceQuantizedFcConvOps = allowSourceQuantizedFcConvOps,
                        gpuBackend = gpuBackend,
                        strictGpuOnly = strictGpuOnly,
                        precision = coreGpuPrecision,
                    )
            }
            return try {
                val projection = ActiveRowProjection.open(
                    assetManager = diagnosticAssets,
                    assetPath = assets.activeRowProjection,
                    threadCount = PROJECTION_THREAD_COUNT,
                )
                LiteRtSplitDecoderDiagnostic(
                    decoderCore = decoderCore,
                    projection = projection,
                    tokenizer = tokenizer,
                    generationConfig = generationConfig,
                    coreBackend = coreBackend,
                )
            } catch (error: Throwable) {
                decoderCore.close()
                throw error
            }
        }

        fun openFiles(
            configurationAssets: AssetManager,
            decoderCorePath: String,
            activeRowProjectionPath: String,
            coreBackend: WhisperSplitDecoderCoreBackend,
            coreCpuThreadCount: Int = LiteRtWhisperEngine.DEFAULT_THREAD_COUNT,
            allowSourceQuantizedFcConvOps: Boolean = true,
            constantTensorSharing: Boolean = true,
            gpuBackend: WhisperSplitDecoderGpuBackend = WhisperSplitDecoderGpuBackend.OPENCL,
            strictGpuOnly: Boolean = false,
            coreGpuPrecision: CompiledModel.GpuOptions.Precision =
                CompiledModel.GpuOptions.Precision.FP32,
        ): LiteRtSplitDecoderDiagnostic {
            require(coreCpuThreadCount > 0) {
                "Split decoder CPU thread count must be positive."
            }
            val generationConfig = configurationAssets.open(WhisperAssets.GENERATION_CONFIG).use {
                WhisperGenerationConfig.load(it)
            }
            validateGenerationConfig(generationConfig)
            val tokenizer = configurationAssets.open(WhisperAssets.TOKENIZER).use {
                WhisperTokenizer.load(it)
            }
            val decoderCore = when (coreBackend) {
                WhisperSplitDecoderCoreBackend.CPU_XNNPACK ->
                    InterpreterDecoderCore.openFile(decoderCorePath, coreCpuThreadCount)
                WhisperSplitDecoderCoreBackend.GPU_COMPILED_MODEL ->
                    GpuDecoderCore.openFile(
                        path = decoderCorePath,
                        constantTensorSharing = constantTensorSharing,
                        allowSourceQuantizedFcConvOps = allowSourceQuantizedFcConvOps,
                        gpuBackend = gpuBackend,
                        strictGpuOnly = strictGpuOnly,
                        precision = coreGpuPrecision,
                    )
            }
            return try {
                LiteRtSplitDecoderDiagnostic(
                    decoderCore = decoderCore,
                    projection = ActiveRowProjection.openFile(
                        activeRowProjectionPath,
                        PROJECTION_THREAD_COUNT,
                    ),
                    tokenizer = tokenizer,
                    generationConfig = generationConfig,
                    coreBackend = coreBackend,
                )
            } catch (error: Throwable) {
                decoderCore.close()
                throw error
            }
        }

        private fun validateGenerationConfig(config: WhisperGenerationConfig) {
            if (
                config.eosTokenId != 50_257 ||
                config.decoderStartTokenId != 50_258 ||
                config.transcribeTokenId != 50_360 ||
                config.noTimestampsTokenId != 50_364 ||
                config.languageToId.isEmpty()
            ) {
                throw WhisperModelContractException(
                    "generation_config.json does not match Whisper large-v3-turbo.",
                )
            }
        }
    }
}

/**
 * Pure tensor preparation/extraction kept separate from LiteRT allocation so
 * the exact split boundary is covered by local unit tests.
 */
internal object WhisperSplitDecoderTensors {
    const val ENCODER_FRAMES = 1_500
    const val ENCODER_WIDTH = 1_280
    const val TARGET_LENGTH = 128
    const val VOCABULARY_SIZE = 51_866
    const val STANDARD_PROMPT_LENGTH = 4
    const val ENCODER_STATE_FLOATS = ENCODER_FRAMES * ENCODER_WIDTH
    const val DECODER_HIDDEN_FLOATS = TARGET_LENGTH * ENCODER_WIDTH

    fun forcedPrompt(
        generationConfig: WhisperGenerationConfig,
        language: WhisperLanguage.Forced,
        referenceTokenPrefix: List<Int>,
    ): List<Int> {
        require(referenceTokenPrefix.size < TARGET_LENGTH - STANDARD_PROMPT_LENGTH) {
            "Teacher-forced prefix has ${referenceTokenPrefix.size} tokens; expected fewer " +
                "than ${TARGET_LENGTH - STANDARD_PROMPT_LENGTH}."
        }
        require(referenceTokenPrefix.all { it in 0 until VOCABULARY_SIZE }) {
            "Teacher-forced prefix contains an out-of-vocabulary token."
        }
        return buildList(STANDARD_PROMPT_LENGTH + referenceTokenPrefix.size) {
            add(generationConfig.decoderStartTokenId)
            add(generationConfig.languageToken(language.code))
            add(generationConfig.transcribeTokenId)
            add(generationConfig.noTimestampsTokenId)
            addAll(referenceTokenPrefix)
        }
    }

    fun paddedTokens(
        prompt: List<Int>,
        paddingToken: Int,
    ): IntArray {
        require(prompt.isNotEmpty()) { "Decoder prompt must not be empty." }
        require(prompt.size <= TARGET_LENGTH) {
            "Decoder prompt has ${prompt.size} tokens; maximum is $TARGET_LENGTH."
        }
        return IntArray(TARGET_LENGTH) { index ->
            if (index < prompt.size) prompt[index] else paddingToken
        }
    }

    fun causalMask(): FloatArray = FloatArray(TARGET_LENGTH * TARGET_LENGTH) { index ->
        val row = index / TARGET_LENGTH
        val column = index % TARGET_LENGTH
        if (column <= row) 0.0f else -1.0e9f
    }

    fun activeHiddenRow(
        decoderHidden: FloatArray,
        promptLength: Int,
    ): FloatArray {
        require(decoderHidden.size == DECODER_HIDDEN_FLOATS) {
            "Split decoder returned ${decoderHidden.size} hidden values; expected " +
                "$DECODER_HIDDEN_FLOATS."
        }
        require(promptLength in 1..TARGET_LENGTH) {
            "Prompt length must be between 1 and $TARGET_LENGTH."
        }
        val start = (promptLength - 1) * ENCODER_WIDTH
        return decoderHidden.copyOfRange(start, start + ENCODER_WIDTH)
    }
}

private interface DecoderCore : AutoCloseable {
    fun decode(
        encoderState: FloatArray,
        tokens: IntArray,
        shouldCancel: () -> Boolean,
    ): FloatArray
}

private class InterpreterDecoderCore private constructor(
    private val interpreter: Interpreter,
    private val bindings: DecoderBindings,
) : DecoderCore {
    private val encoderState =
        directFloatBuffer(WhisperSplitDecoderTensors.ENCODER_STATE_FLOATS)
    private val tokens = directIntBuffer(WhisperSplitDecoderTensors.TARGET_LENGTH)
    private val causalMask =
        directFloatBuffer(
            WhisperSplitDecoderTensors.TARGET_LENGTH *
                WhisperSplitDecoderTensors.TARGET_LENGTH,
        ).also { buffer ->
            buffer.asFloatBuffer().put(WhisperSplitDecoderTensors.causalMask())
            buffer.rewind()
        }
    private val hidden =
        directFloatBuffer(WhisperSplitDecoderTensors.DECODER_HIDDEN_FLOATS)

    override fun decode(
        encoderState: FloatArray,
        tokens: IntArray,
        shouldCancel: () -> Boolean,
    ): FloatArray {
        if (shouldCancel()) throw WhisperCancelledException()
        interpreter.setCancelled(false)
        this.encoderState.clear()
        this.encoderState.asFloatBuffer().put(encoderState)
        this.encoderState.rewind()
        this.tokens.clear()
        this.tokens.asIntBuffer().put(tokens)
        this.tokens.rewind()
        causalMask.rewind()
        hidden.clear()
        interpreter.runSignature(
            mapOf(
                bindings.encoderState to this.encoderState,
                bindings.tokens to this.tokens,
                bindings.causalMask to causalMask,
            ),
            mapOf(bindings.hidden to hidden),
            DECODE_SIGNATURE,
        )
        if (shouldCancel()) throw WhisperCancelledException()
        hidden.rewind()
        return FloatArray(WhisperSplitDecoderTensors.DECODER_HIDDEN_FLOATS).also {
            values ->
            hidden.asFloatBuffer().get(values)
            hidden.rewind()
        }
    }

    override fun close() {
        interpreter.setCancelled(true)
        interpreter.close()
    }

    companion object {
        fun openFile(path: String, threadCount: Int): InterpreterDecoderCore =
            create(mapModelFile(path, "split decoder core"), threadCount)

        fun open(
            assetManager: AssetManager,
            assetPath: String,
            threadCount: Int,
        ): InterpreterDecoderCore {
            val model = mapUncompressedAsset(
                assetManager = assetManager,
                assetPath = assetPath,
                label = "split decoder core",
            )
            return create(model, threadCount)
        }

        private fun create(
            model: MappedByteBuffer,
            threadCount: Int,
        ): InterpreterDecoderCore {
            val interpreter = Interpreter(
                model,
                Interpreter.Options()
                    .setNumThreads(threadCount)
                    .setUseXNNPACK(true)
                    .setCancellable(true),
            )
            return try {
                InterpreterDecoderCore(
                    interpreter = interpreter,
                    bindings = resolveDecoderBindings(interpreter),
                )
            } catch (error: Throwable) {
                interpreter.close()
                throw error
            }
        }
    }
}

private class GpuDecoderCore private constructor(
    private val model: CompiledModel,
    private val encoderState: TensorBuffer,
    private val tokens: TensorBuffer,
    private val causalMask: TensorBuffer,
    private val hidden: TensorBuffer,
) : DecoderCore {
    override fun decode(
        encoderState: FloatArray,
        tokens: IntArray,
        shouldCancel: () -> Boolean,
    ): FloatArray {
        if (shouldCancel()) throw WhisperCancelledException()
        gpuOperation("writing split decoder encoder state") {
            this.encoderState.writeFloat(encoderState)
        }
        gpuOperation("writing split decoder tokens") {
            this.tokens.writeInt(tokens)
        }
        gpuOperation("running split decoder core") {
            model.run(
                mapOf(
                    DECODE_ENCODER_STATE to this.encoderState,
                    DECODE_TOKENS to this.tokens,
                    DECODE_MASK to causalMask,
                ),
                mapOf(DECODE_OUTPUT to hidden),
                DECODE_SIGNATURE,
            )
        }
        if (shouldCancel()) throw WhisperCancelledException()
        val values = gpuOperation("reading split decoder hidden state") {
            hidden.readFloat()
        }
        if (values.size != WhisperSplitDecoderTensors.DECODER_HIDDEN_FLOATS) {
            throw WhisperGpuBackendException(
                "GPU split decoder returned ${values.size} hidden values; expected " +
                    "${WhisperSplitDecoderTensors.DECODER_HIDDEN_FLOATS}.",
            )
        }
        return values
    }

    override fun close() {
        listOf(hidden, causalMask, tokens, encoderState).forEach { buffer ->
            runCatching(buffer::close)
        }
        model.close()
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

    companion object {
        fun openFile(
            path: String,
            constantTensorSharing: Boolean,
            allowSourceQuantizedFcConvOps: Boolean,
            gpuBackend: WhisperSplitDecoderGpuBackend,
            strictGpuOnly: Boolean,
            precision: CompiledModel.GpuOptions.Precision,
        ): GpuDecoderCore = openModel(
            createModel = { options -> CompiledModel.create(path, options) },
            constantTensorSharing = constantTensorSharing,
            allowSourceQuantizedFcConvOps = allowSourceQuantizedFcConvOps,
            gpuBackend = gpuBackend,
            strictGpuOnly = strictGpuOnly,
            precision = precision,
        )

        fun open(
            assetManager: AssetManager,
            assetPath: String,
            constantTensorSharing: Boolean,
            allowSourceQuantizedFcConvOps: Boolean,
            gpuBackend: WhisperSplitDecoderGpuBackend,
            strictGpuOnly: Boolean,
            precision: CompiledModel.GpuOptions.Precision,
        ): GpuDecoderCore = openModel(
            createModel = { options -> CompiledModel.create(assetManager, assetPath, options) },
            constantTensorSharing = constantTensorSharing,
            allowSourceQuantizedFcConvOps = allowSourceQuantizedFcConvOps,
            gpuBackend = gpuBackend,
            strictGpuOnly = strictGpuOnly,
            precision = precision,
        )

        private fun openModel(
            createModel: (CompiledModel.Options) -> CompiledModel,
            constantTensorSharing: Boolean,
            allowSourceQuantizedFcConvOps: Boolean,
            gpuBackend: WhisperSplitDecoderGpuBackend,
            strictGpuOnly: Boolean,
            precision: CompiledModel.GpuOptions.Precision,
        ): GpuDecoderCore {
            val selection = selectWhisperSplitDecoderGpuOptions(
                constantTensorSharing = constantTensorSharing,
                allowSourceQuantizedFcConvOps = allowSourceQuantizedFcConvOps,
                gpuBackend = gpuBackend,
                strictGpuOnly = strictGpuOnly,
            )
            val options =
                CompiledModel.Options(*selection.accelerators.toTypedArray()).apply {
                    gpuOptions = CompiledModel.GpuOptions(
                        constantTensorSharing = selection.constantTensorSharing,
                        allowSrcQuantizedFcConvOps =
                            selection.allowSourceQuantizedFcConvOps,
                        precision = precision,
                        backend = selection.backend,
                    )
                }
            val model = createModel(options)
            val allocated = mutableListOf<TensorBuffer>()
            return try {
                validateGpuDecoderContract(model)
                fun input(name: String): TensorBuffer =
                    model.createInputBuffer(name, DECODE_SIGNATURE).also(allocated::add)
                fun output(name: String): TensorBuffer =
                    model.createOutputBuffer(name, DECODE_SIGNATURE).also(allocated::add)

                val mask = input(DECODE_MASK)
                mask.writeFloat(WhisperSplitDecoderTensors.causalMask())
                GpuDecoderCore(
                    model = model,
                    encoderState = input(DECODE_ENCODER_STATE),
                    tokens = input(DECODE_TOKENS),
                    causalMask = mask,
                    hidden = output(DECODE_OUTPUT),
                )
            } catch (error: Throwable) {
                allocated.asReversed().forEach { buffer ->
                    runCatching(buffer::close)
                }
                model.close()
                throw error
            }
        }
    }
}

internal class ActiveRowProjection private constructor(
    private val interpreter: Interpreter,
    private val bindings: ProjectionBindings,
) : AutoCloseable {
    private val hidden = directFloatBuffer(WhisperSplitDecoderTensors.ENCODER_WIDTH)
    private val logits = directFloatBuffer(WhisperSplitDecoderTensors.VOCABULARY_SIZE)

    fun project(
        activeHidden: FloatArray,
        shouldCancel: () -> Boolean,
    ): FloatArray {
        require(activeHidden.size == WhisperSplitDecoderTensors.ENCODER_WIDTH) {
            "Active decoder row has ${activeHidden.size} floats; expected " +
                "${WhisperSplitDecoderTensors.ENCODER_WIDTH}."
        }
        if (shouldCancel()) throw WhisperCancelledException()
        interpreter.setCancelled(false)
        hidden.clear()
        hidden.asFloatBuffer().put(activeHidden)
        hidden.rewind()
        logits.clear()
        interpreter.runSignature(
            mapOf(bindings.hidden to hidden),
            mapOf(bindings.logits to logits),
            bindings.signature,
        )
        if (shouldCancel()) throw WhisperCancelledException()
        logits.rewind()
        return FloatArray(WhisperSplitDecoderTensors.VOCABULARY_SIZE).also { values ->
            logits.asFloatBuffer().get(values)
            logits.rewind()
        }
    }

    override fun close() {
        interpreter.setCancelled(true)
        interpreter.close()
    }

    companion object {
        fun openFile(path: String, threadCount: Int): ActiveRowProjection =
            create(mapModelFile(path, "active-row projection"), threadCount)

        fun open(
            assetManager: AssetManager,
            assetPath: String,
            threadCount: Int,
        ): ActiveRowProjection {
            val model = mapUncompressedAsset(
                assetManager = assetManager,
                assetPath = assetPath,
                label = "active-row projection",
            )
            return create(model, threadCount)
        }

        private fun create(
            model: MappedByteBuffer,
            threadCount: Int,
        ): ActiveRowProjection {
            val interpreter = Interpreter(
                model,
                Interpreter.Options()
                    .setNumThreads(threadCount)
                    .setUseXNNPACK(true)
                    .setCancellable(true),
            )
            return try {
                ActiveRowProjection(
                    interpreter = interpreter,
                    bindings = resolveProjectionBindings(interpreter),
                )
            } catch (error: Throwable) {
                interpreter.close()
                throw error
            }
        }
    }
}

private data class DecoderBindings(
    val encoderState: String,
    val tokens: String,
    val causalMask: String,
    val hidden: String,
)

private data class ProjectionBindings(
    val signature: String,
    val hidden: String,
    val logits: String,
)

private const val DECODE_SIGNATURE = "decode"
private const val DECODE_ENCODER_STATE = "args_0"
private const val DECODE_TOKENS = "args_1"
private const val DECODE_MASK = "args_2"
private const val DECODE_OUTPUT = "output_0"
private const val PROJECTION_SIGNATURE = "project"
private const val NORMALIZE_AND_PROJECT_SIGNATURE = "normalize_and_project"
private const val PROJECTION_INPUT = "hidden"
private const val NORMALIZE_AND_PROJECT_INPUT = "hidden_pre_layer_norm"
private const val PROJECTION_OUTPUT = "logits"

private fun resolveDecoderBindings(interpreter: Interpreter): DecoderBindings {
    requireSignature(interpreter, DECODE_SIGNATURE)
    return DecoderBindings(
        encoderState = resolveInterpreterTensor(
            interpreter = interpreter,
            signature = DECODE_SIGNATURE,
            input = true,
            preferredName = DECODE_ENCODER_STATE,
            type = DataType.FLOAT32,
            shape = intArrayOf(
                1,
                WhisperSplitDecoderTensors.ENCODER_FRAMES,
                WhisperSplitDecoderTensors.ENCODER_WIDTH,
            ),
        ),
        tokens = resolveInterpreterTensor(
            interpreter = interpreter,
            signature = DECODE_SIGNATURE,
            input = true,
            preferredName = DECODE_TOKENS,
            type = DataType.INT32,
            shape = intArrayOf(1, WhisperSplitDecoderTensors.TARGET_LENGTH),
        ),
        causalMask = resolveInterpreterTensor(
            interpreter = interpreter,
            signature = DECODE_SIGNATURE,
            input = true,
            preferredName = DECODE_MASK,
            type = DataType.FLOAT32,
            shape = intArrayOf(
                1,
                1,
                WhisperSplitDecoderTensors.TARGET_LENGTH,
                WhisperSplitDecoderTensors.TARGET_LENGTH,
            ),
        ),
        hidden = resolveInterpreterTensor(
            interpreter = interpreter,
            signature = DECODE_SIGNATURE,
            input = false,
            preferredName = DECODE_OUTPUT,
            type = DataType.FLOAT32,
            shape = intArrayOf(
                1,
                WhisperSplitDecoderTensors.TARGET_LENGTH,
                WhisperSplitDecoderTensors.ENCODER_WIDTH,
            ),
        ),
    )
}

private fun resolveProjectionBindings(interpreter: Interpreter): ProjectionBindings {
    val signature = selectProjectionSignature(interpreter.signatureKeys.asList())
    val preferredInput = when (signature) {
        PROJECTION_SIGNATURE -> PROJECTION_INPUT
        NORMALIZE_AND_PROJECT_SIGNATURE -> NORMALIZE_AND_PROJECT_INPUT
        else -> error("Unsupported projection signature $signature.")
    }
    return ProjectionBindings(
        signature = signature,
        hidden = resolveInterpreterTensor(
            interpreter = interpreter,
            signature = signature,
            input = true,
            preferredName = preferredInput,
            type = DataType.FLOAT32,
            shape = intArrayOf(1, WhisperSplitDecoderTensors.ENCODER_WIDTH),
        ),
        logits = resolveInterpreterTensor(
            interpreter = interpreter,
            signature = signature,
            input = false,
            preferredName = PROJECTION_OUTPUT,
            type = DataType.FLOAT32,
            shape = intArrayOf(1, WhisperSplitDecoderTensors.VOCABULARY_SIZE),
        ),
    )
}

internal fun selectProjectionSignature(signatureKeys: Collection<String>): String {
    val supported = listOf(
        PROJECTION_SIGNATURE,
        NORMALIZE_AND_PROJECT_SIGNATURE,
    ).filter(signatureKeys::contains)
    if (supported.size != 1) {
        throw WhisperModelContractException(
            "Model signatures are ${signatureKeys.toList()}; exactly one of " +
                "$PROJECTION_SIGNATURE or $NORMALIZE_AND_PROJECT_SIGNATURE is required.",
        )
    }
    return supported.single()
}

private fun requireSignature(
    interpreter: Interpreter,
    signature: String,
) {
    if (signature !in interpreter.signatureKeys) {
        throw WhisperModelContractException(
            "Model signatures are ${interpreter.signatureKeys.toList()}; " +
                "$signature is required.",
        )
    }
}

private fun resolveInterpreterTensor(
    interpreter: Interpreter,
    signature: String,
    input: Boolean,
    preferredName: String,
    type: DataType,
    shape: IntArray,
): String {
    val names = if (input) {
        interpreter.getSignatureInputs(signature)
    } else {
        interpreter.getSignatureOutputs(signature)
    }
    fun tensor(name: String): Tensor =
        if (input) {
            interpreter.getInputTensorFromSignature(name, signature)
        } else {
            interpreter.getOutputTensorFromSignature(name, signature)
        }

    if (preferredName in names) {
        validateInterpreterTensor(
            label = "$signature/$preferredName",
            tensor = tensor(preferredName),
            type = type,
            shape = shape,
        )
        return preferredName
    }
    val matching = names.filter { name ->
        val candidate = tensor(name)
        candidate.dataType() == type && candidate.shape().contentEquals(shape)
    }
    if (matching.size != 1) {
        throw WhisperModelContractException(
            "Cannot uniquely bind $signature/$preferredName with $type " +
                "${shape.contentToString()}; available names are ${names.toList()}.",
        )
    }
    return matching.single()
}

private fun validateInterpreterTensor(
    label: String,
    tensor: Tensor,
    type: DataType,
    shape: IntArray,
) {
    if (tensor.dataType() != type || !tensor.shape().contentEquals(shape)) {
        throw WhisperModelContractException(
            "$label is ${tensor.dataType()} ${tensor.shape().contentToString()}; " +
                "expected $type ${shape.contentToString()}.",
        )
    }
}

private fun validateGpuDecoderContract(model: CompiledModel) {
    validateGpuTensor(
        label = "$DECODE_SIGNATURE/$DECODE_ENCODER_STATE",
        tensor = model.getInputTensorType(DECODE_ENCODER_STATE, DECODE_SIGNATURE),
        elementType = TensorType.ElementType.FLOAT,
        shape = intArrayOf(
            1,
            WhisperSplitDecoderTensors.ENCODER_FRAMES,
            WhisperSplitDecoderTensors.ENCODER_WIDTH,
        ),
    )
    validateGpuTensor(
        label = "$DECODE_SIGNATURE/$DECODE_TOKENS",
        tensor = model.getInputTensorType(DECODE_TOKENS, DECODE_SIGNATURE),
        elementType = TensorType.ElementType.INT,
        shape = intArrayOf(1, WhisperSplitDecoderTensors.TARGET_LENGTH),
    )
    validateGpuTensor(
        label = "$DECODE_SIGNATURE/$DECODE_MASK",
        tensor = model.getInputTensorType(DECODE_MASK, DECODE_SIGNATURE),
        elementType = TensorType.ElementType.FLOAT,
        shape = intArrayOf(
            1,
            1,
            WhisperSplitDecoderTensors.TARGET_LENGTH,
            WhisperSplitDecoderTensors.TARGET_LENGTH,
        ),
    )
    validateGpuTensor(
        label = "$DECODE_SIGNATURE/$DECODE_OUTPUT",
        tensor = model.getOutputTensorType(DECODE_OUTPUT, DECODE_SIGNATURE),
        elementType = TensorType.ElementType.FLOAT,
        shape = intArrayOf(
            1,
            WhisperSplitDecoderTensors.TARGET_LENGTH,
            WhisperSplitDecoderTensors.ENCODER_WIDTH,
        ),
    )
}

private fun validateGpuTensor(
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

private fun mapUncompressedAsset(
    assetManager: AssetManager,
    assetPath: String,
    label: String,
): MappedByteBuffer = try {
    assetManager.openFd(assetPath).use { descriptor ->
        FileInputStream(descriptor.fileDescriptor).channel.use { channel ->
            channel.map(
                FileChannel.MapMode.READ_ONLY,
                descriptor.startOffset,
                descriptor.declaredLength,
            )
        }
    }
} catch (error: IOException) {
    throw WhisperModelContractException(
        "The diagnostic $label asset '$assetPath' is unavailable or compressed: " +
            (error.message ?: error.javaClass.simpleName),
    )
}

private fun mapModelFile(path: String, label: String): MappedByteBuffer {
    val file = java.io.File(path)
    require(file.isFile) { "Diagnostic $label is missing." }
    return FileInputStream(file).channel.use { channel ->
        channel.map(FileChannel.MapMode.READ_ONLY, 0L, channel.size())
    }
}

private fun directFloatBuffer(elements: Int): ByteBuffer =
    ByteBuffer.allocateDirect(Math.multiplyExact(elements, Float.SIZE_BYTES))
        .order(ByteOrder.nativeOrder())

private fun directIntBuffer(elements: Int): ByteBuffer =
    ByteBuffer.allocateDirect(Math.multiplyExact(elements, Int.SIZE_BYTES))
        .order(ByteOrder.nativeOrder())
