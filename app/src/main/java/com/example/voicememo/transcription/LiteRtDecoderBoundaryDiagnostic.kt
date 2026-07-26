package com.example.voicememo.transcription

import android.content.res.AssetManager
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
 * A generated decoder boundary and the backend assigned to each side.
 *
 * These models and this runtime are diagnostic-only. They are not reachable
 * through [WhisperBackendSelector] and cannot persist a transcript.
 */
internal enum class WhisperDecoderBoundaryStage(
    val id: String,
    val sharedModelFileName: String,
    val boundaryInputName: String,
    val prefixBackend: WhisperSplitDecoderCoreBackend,
    val tailBackend: WhisperSplitDecoderCoreBackend,
) {
    AFTER_POSITION_ADD(
        id = "after-position-add",
        sharedModelFileName =
            "whisper_large_v3_turbo_decode_shared_after_position_add.tflite",
        boundaryInputName = "hidden_after_position_add",
        prefixBackend = WhisperSplitDecoderCoreBackend.GPU_COMPILED_MODEL,
        tailBackend = WhisperSplitDecoderCoreBackend.CPU_XNNPACK,
    ),
    AFTER_CROSS_ATTENTION(
        id = "after-cross-attention",
        sharedModelFileName =
            "whisper_large_v3_turbo_decode_shared_after_cross_attention.tflite",
        boundaryInputName = "hidden_after_cross_attention",
        prefixBackend = WhisperSplitDecoderCoreBackend.GPU_COMPILED_MODEL,
        tailBackend = WhisperSplitDecoderCoreBackend.CPU_XNNPACK,
    ),
    AFTER_1_BLOCK(
        id = "after-1-block",
        sharedModelFileName =
            "whisper_large_v3_turbo_decode_shared_after_1_block.tflite",
        boundaryInputName = "hidden_after_1_decoder_block",
        prefixBackend = WhisperSplitDecoderCoreBackend.GPU_COMPILED_MODEL,
        tailBackend = WhisperSplitDecoderCoreBackend.CPU_XNNPACK,
    ),
    AFTER_2_BLOCKS(
        id = "after-2-blocks",
        sharedModelFileName =
            "whisper_large_v3_turbo_decode_shared_after_2_blocks.tflite",
        boundaryInputName = "hidden_after_2_decoder_blocks",
        prefixBackend = WhisperSplitDecoderCoreBackend.GPU_COMPILED_MODEL,
        tailBackend = WhisperSplitDecoderCoreBackend.CPU_XNNPACK,
    ),
    AFTER_3_BLOCKS(
        id = "after-3-blocks",
        sharedModelFileName =
            "whisper_large_v3_turbo_decode_shared_after_3_blocks.tflite",
        boundaryInputName = "hidden_after_3_decoder_blocks",
        prefixBackend = WhisperSplitDecoderCoreBackend.GPU_COMPILED_MODEL,
        tailBackend = WhisperSplitDecoderCoreBackend.CPU_XNNPACK,
    ),
    ;

    val tailSignature: String
        get() = "decode_tail_${id.replace('-', '_')}"

    companion object {
        fun fromId(id: String): WhisperDecoderBoundaryStage? =
            entries.singleOrNull { stage -> stage.id == id }
    }
}

/**
 * [ALL_CPU_CONTROL] runs both signatures from the same generated shared model
 * through Interpreter/XNNPACK. It is the required model-patching control
 * before attributing a stage-routed mismatch to GPU execution.
 */
internal enum class WhisperDecoderBoundaryExecutionMode {
    STAGE_ROUTED,
    CPU_PREFIX_GPU_TAIL,
    ALL_CPU_CONTROL,
    ;

    fun prefixBackend(
        stage: WhisperDecoderBoundaryStage,
    ): WhisperSplitDecoderCoreBackend =
        when (this) {
            STAGE_ROUTED -> stage.prefixBackend
            CPU_PREFIX_GPU_TAIL -> WhisperSplitDecoderCoreBackend.CPU_XNNPACK
            ALL_CPU_CONTROL -> WhisperSplitDecoderCoreBackend.CPU_XNNPACK
        }

    fun tailBackend(
        stage: WhisperDecoderBoundaryStage,
    ): WhisperSplitDecoderCoreBackend =
        when (this) {
            STAGE_ROUTED -> stage.tailBackend
            CPU_PREFIX_GPU_TAIL ->
                WhisperSplitDecoderCoreBackend.GPU_COMPILED_MODEL
            ALL_CPU_CONTROL -> WhisperSplitDecoderCoreBackend.CPU_XNNPACK
        }
}

internal data class WhisperDecoderBoundaryAssets(
    val sharedDecoder: String,
    val activeRowNormalizeAndProject: String,
) {
    companion object {
        const val DEFAULT_DIRECTORY = WhisperSplitDecoderAssets.DEFAULT_DIRECTORY

        fun forStage(stage: WhisperDecoderBoundaryStage) =
            WhisperDecoderBoundaryAssets(
                sharedDecoder = "$DEFAULT_DIRECTORY/${stage.sharedModelFileName}",
                activeRowNormalizeAndProject =
                    "$DEFAULT_DIRECTORY/" +
                        "whisper_large_v3_turbo_active_row_layer_norm_projection_i8.tflite",
            )
    }
}

/**
 * Content-free teacher-forced result for one decoder position.
 *
 * Only token IDs/scores from the existing CPU ranking are exposed. No
 * tokenizer is opened and no transcript text is decoded or stored.
 */
internal data class WhisperDecoderBoundaryStep(
    val promptLength: Int,
    val activeBoundaryRow: FloatArray,
    val activePreLayerNormRow: FloatArray,
    val rankedLogits: WhisperTeacherForcedLogitsRow,
    val prefixMillis: Long,
    val tailMillis: Long,
    val projectionMillis: Long,
)

internal interface WhisperDecoderBoundaryPrefix : AutoCloseable {
    fun run(
        encoderState: FloatArray,
        tokens: IntArray,
        shouldCancel: () -> Boolean,
    ): FloatArray
}

internal interface WhisperDecoderBoundaryTail : AutoCloseable {
    fun run(
        encoderState: FloatArray,
        boundary: FloatArray,
        shouldCancel: () -> Boolean,
    ): FloatArray
}

internal interface WhisperDecoderBoundaryProjection : AutoCloseable {
    fun run(
        activePreLayerNorm: FloatArray,
        shouldCancel: () -> Boolean,
    ): FloatArray
}

/**
 * Exact CPU verifier backed by whichever side of the routed split already owns
 * a shared-model Interpreter. One call runs the shared model's prefix and tail
 * signatures with a causal token buffer and returns every pre-layer-norm row.
 */
internal interface WhisperDecoderBoundaryCpuVerifier {
    fun runFullDecoder(
        encoderState: FloatArray,
        tokens: IntArray,
        shouldCancel: () -> Boolean,
    ): FloatArray
}

/**
 * Runs a generated compact shared decoder split across CPU and GPU.
 *
 * The prefix and tail always use opposite backends according to
 * [executionMode]. The final active-row layer norm and INT8 vocabulary
 * projection always use Interpreter/XNNPACK, and ranking always uses the exact
 * CPU policy.
 */
internal class LiteRtDecoderBoundaryDiagnostic internal constructor(
    private val prefix: WhisperDecoderBoundaryPrefix,
    private val tail: WhisperDecoderBoundaryTail,
    private val projection: WhisperDecoderBoundaryProjection,
    private val cpuVerifier: WhisperDecoderBoundaryCpuVerifier? =
        (prefix as? WhisperDecoderBoundaryCpuVerifier)
            ?: (tail as? WhisperDecoderBoundaryCpuVerifier),
    private val generationConfig: WhisperGenerationConfig,
    val stage: WhisperDecoderBoundaryStage,
    val executionMode: WhisperDecoderBoundaryExecutionMode,
) : AutoCloseable {
    private val closed = AtomicBoolean(false)
    private val inferenceLock = Any()

    fun teacherForcedStep(
        encoderState: FloatArray,
        language: WhisperLanguage.Forced,
        referenceTokenPrefix: List<Int>,
        shouldCancel: () -> Boolean = { false },
    ): WhisperDecoderBoundaryStep = synchronized(inferenceLock) {
        check(!closed.get()) { "Decoder-boundary diagnostic is closed." }
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
        val tokens = WhisperSplitDecoderTensors.paddedTokens(
            prompt = prompt,
            paddingToken = generationConfig.eosTokenId,
        )

        val prefixStarted = System.nanoTime()
        val boundary = prefix.run(
            encoderState = encoderState,
            tokens = tokens,
            shouldCancel = shouldCancel,
        )
        val prefixMillis = elapsedMillis(prefixStarted)
        checkCancellation(shouldCancel)
        val activeBoundary = WhisperSplitDecoderTensors.activeHiddenRow(
            decoderHidden = boundary,
            promptLength = prompt.size,
        )

        val tailStarted = System.nanoTime()
        val preLayerNorm = tail.run(
            encoderState = encoderState,
            boundary = boundary,
            shouldCancel = shouldCancel,
        )
        val tailMillis = elapsedMillis(tailStarted)
        checkCancellation(shouldCancel)
        val activePreLayerNorm = WhisperSplitDecoderTensors.activeHiddenRow(
            decoderHidden = preLayerNorm,
            promptLength = prompt.size,
        )

        val projectionStarted = System.nanoTime()
        val logits = projection.run(
            activePreLayerNorm = activePreLayerNorm,
            shouldCancel = shouldCancel,
        )
        val projectionMillis = elapsedMillis(projectionStarted)
        checkCancellation(shouldCancel)

        WhisperDecoderBoundaryStep(
            promptLength = prompt.size,
            activeBoundaryRow = activeBoundary,
            activePreLayerNormRow = activePreLayerNorm,
            rankedLogits = WhisperTeacherForcedTokenRanking.forCpu(
                logits = logits,
                generationConfig = generationConfig,
                firstGeneratedToken = referenceTokenPrefix.isEmpty(),
            ),
            prefixMillis = prefixMillis,
            tailMillis = tailMillis,
            projectionMillis = projectionMillis,
        )
    }

    /**
     * Proposes a short autoregressive block using the routed CPU/GPU split.
     *
     * The returned token IDs are diagnostic-only and must be passed to
     * [verifyProposedTokenBlock] on this same instance before acceptance.
     */
    internal fun proposeTokenBlock(
        encoderState: FloatArray,
        language: WhisperLanguage.Forced,
        verifiedTokenPrefix: List<Int>,
        maximumTokenCount: Int,
        shouldCancel: () -> Boolean = { false },
    ): WhisperGpuProposedTokenBlock = synchronized(inferenceLock) {
        requireSpeculativeRuntime()
        check(!closed.get()) { "Decoder-boundary diagnostic is closed." }
        requireEncoderState(encoderState)
        require(maximumTokenCount > 0) {
            "Boundary speculative block size must be positive."
        }
        val prompt = WhisperSplitDecoderTensors.forcedPrompt(
            generationConfig = generationConfig,
            language = language,
            referenceTokenPrefix = verifiedTokenPrefix,
        ).toMutableList()
        val allowedTokenCount = minOf(
            maximumTokenCount,
            WhisperSplitDecoderTensors.TARGET_LENGTH - prompt.size,
        )
        require(allowedTokenCount > 0) {
            "The verified prefix already fills the decoder context."
        }

        checkCancellation(shouldCancel)
        val proposed = mutableListOf<Int>()
        val selectableMargins = mutableListOf<Float?>()
        val started = System.nanoTime()
        while (proposed.size < allowedTokenCount) {
            checkCancellation(shouldCancel)
            val tokens = WhisperSplitDecoderTensors.paddedTokens(
                prompt = prompt,
                paddingToken = generationConfig.eosTokenId,
            )
            val boundary = prefix.run(
                encoderState = encoderState,
                tokens = tokens,
                shouldCancel = shouldCancel,
            )
            checkCancellation(shouldCancel)
            val preLayerNorm = tail.run(
                encoderState = encoderState,
                boundary = boundary,
                shouldCancel = shouldCancel,
            )
            checkCancellation(shouldCancel)
            val activePreLayerNorm = WhisperSplitDecoderTensors.activeHiddenRow(
                decoderHidden = preLayerNorm,
                promptLength = prompt.size,
            )
            val logits = projection.run(
                activePreLayerNorm = activePreLayerNorm,
                shouldCancel = shouldCancel,
            )
            checkCancellation(shouldCancel)
            val firstGeneratedToken =
                verifiedTokenPrefix.isEmpty() && proposed.isEmpty()
            val nextToken = WhisperGpuGreedyTokenSelector.selectTextToken(
                logits = logits,
                position = 0,
                vocabularySize = WhisperSplitDecoderTensors.VOCABULARY_SIZE,
                decoderStartTokenId = generationConfig.decoderStartTokenId,
                eosTokenId = generationConfig.eosTokenId,
                suppressTokens = generationConfig.suppressTokens,
                beginSuppressTokens = generationConfig.beginSuppressTokens,
                firstGeneratedToken = firstGeneratedToken,
            )
            proposed += nextToken
            selectableMargins += WhisperTeacherForcedTokenRanking.forGpu(
                logits = logits,
                generationConfig = generationConfig,
                firstGeneratedToken = firstGeneratedToken,
            ).selectableMargin
            if (nextToken == generationConfig.eosTokenId) break
            prompt += nextToken
        }

        WhisperGpuProposedTokenBlock(
            tokens = proposed,
            decoderInvocations = proposed.size,
            inferenceMillis = elapsedMillis(started),
            gpuSelectableTop1Margins = selectableMargins,
        )
    }

    /**
     * Verifies all proposed causal rows with one CPU shared-decoder pass.
     *
     * The verifier is the Interpreter already used by this instance's CPU
     * side. It runs the prefix and tail signatures once each, then the compact
     * CPU projection selects each causal row. Rows after a mismatch remain in
     * the result for [WhisperSpeculativeTokenAcceptance] to discard.
     */
    internal fun verifyProposedTokenBlock(
        encoderState: FloatArray,
        language: WhisperLanguage.Forced,
        verifiedTokenPrefix: List<Int>,
        proposedTokens: List<Int>,
        shouldCancel: () -> Boolean = { false },
    ): WhisperCpuVerifiedTokenBlock = synchronized(inferenceLock) {
        requireSpeculativeRuntime()
        check(!closed.get()) { "Decoder-boundary diagnostic is closed." }
        requireEncoderState(encoderState)
        require(proposedTokens.isNotEmpty()) {
            "At least one speculative token is required."
        }
        require(
            proposedTokens.all {
                it in 0 until WhisperSplitDecoderTensors.VOCABULARY_SIZE
            },
        ) {
            "Speculative block contains an out-of-vocabulary token."
        }
        val basePrompt = WhisperSplitDecoderTensors.forcedPrompt(
            generationConfig = generationConfig,
            language = language,
            referenceTokenPrefix = verifiedTokenPrefix,
        )
        require(
            basePrompt.size + proposedTokens.size <=
                WhisperSplitDecoderTensors.TARGET_LENGTH,
        ) {
            "Speculative block exceeds the decoder's " +
                "${WhisperSplitDecoderTensors.TARGET_LENGTH}-token context."
        }
        val decoderInput = basePrompt + proposedTokens
        val tokens = WhisperSplitDecoderTensors.paddedTokens(
            prompt = decoderInput,
            paddingToken = generationConfig.eosTokenId,
        )
        val verifier = checkNotNull(cpuVerifier) {
            "The routed CPU side does not expose the shared-model verifier."
        }

        checkCancellation(shouldCancel)
        val started = System.nanoTime()
        val preLayerNorm = verifier.runFullDecoder(
            encoderState = encoderState,
            tokens = tokens,
            shouldCancel = shouldCancel,
        )
        checkCancellation(shouldCancel)
        val cpuRankings = proposedTokens.indices.map { offset ->
            checkCancellation(shouldCancel)
            val causalRow = basePrompt.lastIndex + offset
            val activePreLayerNorm = WhisperSplitDecoderTensors.activeHiddenRow(
                decoderHidden = preLayerNorm,
                promptLength = causalRow + 1,
            )
            val logits = projection.run(
                activePreLayerNorm = activePreLayerNorm,
                shouldCancel = shouldCancel,
            )
            checkCancellation(shouldCancel)
            WhisperTeacherForcedTokenRanking.forCpu(
                logits = logits,
                generationConfig = generationConfig,
                firstGeneratedToken =
                    verifiedTokenPrefix.isEmpty() && offset == 0,
            )
        }
        val cpuSelections = cpuRankings.map { ranking ->
            ranking.selectableTop1?.tokenId ?: generationConfig.eosTokenId
        }

        WhisperCpuVerifiedTokenBlock(
            proposedTokens = proposedTokens.toList(),
            cpuSelectedTokens = cpuSelections,
            inferenceMillis = elapsedMillis(started),
            cpuSelectableTop1Margins =
                cpuRankings.map { ranking -> ranking.selectableMargin },
            proposalMatchesCpuTop1 =
                proposedTokens.indices.map { index ->
                    proposedTokens[index] == cpuSelections[index]
                },
        )
    }

    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        synchronized(inferenceLock) {
            runCatching(projection::close)
            runCatching(tail::close)
            prefix.close()
        }
    }

    private fun checkCancellation(shouldCancel: () -> Boolean) {
        if (closed.get() || shouldCancel()) throw WhisperCancelledException()
    }

    private fun requireEncoderState(encoderState: FloatArray) {
        require(encoderState.size == WhisperSplitDecoderTensors.ENCODER_STATE_FLOATS) {
            "Whisper encoder state has ${encoderState.size} floats; expected " +
                "${WhisperSplitDecoderTensors.ENCODER_STATE_FLOATS}."
        }
    }

    private fun requireSpeculativeRuntime() {
        check(stage == WhisperDecoderBoundaryStage.AFTER_1_BLOCK) {
            "Boundary speculative decoding requires the after-1-block stage."
        }
        check(
            executionMode == WhisperDecoderBoundaryExecutionMode.STAGE_ROUTED ||
                executionMode ==
                WhisperDecoderBoundaryExecutionMode.CPU_PREFIX_GPU_TAIL,
        ) {
            "Boundary speculative decoding requires a routed CPU/GPU execution mode."
        }
    }

    companion object {
        private const val NANOS_PER_MILLISECOND = 1_000_000L

        fun open(
            configurationAssets: AssetManager,
            diagnosticAssets: AssetManager = configurationAssets,
            stage: WhisperDecoderBoundaryStage =
                WhisperDecoderBoundaryStage.AFTER_3_BLOCKS,
            assets: WhisperDecoderBoundaryAssets =
                WhisperDecoderBoundaryAssets.forStage(stage),
            executionMode: WhisperDecoderBoundaryExecutionMode =
                WhisperDecoderBoundaryExecutionMode.STAGE_ROUTED,
            cpuThreadCount: Int = LiteRtWhisperEngine.DEFAULT_THREAD_COUNT,
            allowSourceQuantizedFcConvOps: Boolean = true,
            constantTensorSharing: Boolean = true,
            gpuBackend: WhisperSplitDecoderGpuBackend =
                WhisperSplitDecoderGpuBackend.OPENCL,
            strictGpuOnly: Boolean = false,
        ): LiteRtDecoderBoundaryDiagnostic {
            require(cpuThreadCount > 0) {
                "Decoder-boundary CPU thread count must be positive."
            }
            val generationConfig =
                configurationAssets.open(WhisperAssets.GENERATION_CONFIG).use {
                    WhisperGenerationConfig.load(it)
                }
            validateGenerationConfig(generationConfig)

            val sharedCpuControl =
                if (executionMode == WhisperDecoderBoundaryExecutionMode.ALL_CPU_CONTROL) {
                    InterpreterBoundaryPrefixTail.open(
                        assetManager = diagnosticAssets,
                        assetPath = assets.sharedDecoder,
                        stage = stage,
                        threadCount = cpuThreadCount,
                    )
                } else {
                    null
                }
            val prefix = sharedCpuControl ?: openPrefix(
                diagnosticAssets = diagnosticAssets,
                assetPath = assets.sharedDecoder,
                stage = stage,
                backend = executionMode.prefixBackend(stage),
                threadCount = cpuThreadCount,
                allowSourceQuantizedFcConvOps = allowSourceQuantizedFcConvOps,
                constantTensorSharing = constantTensorSharing,
                gpuBackend = gpuBackend,
                strictGpuOnly = strictGpuOnly,
            )
            var tail: WhisperDecoderBoundaryTail? = null
            var projection: WhisperDecoderBoundaryProjection? = null
            return try {
                tail = sharedCpuControl ?: openTail(
                    diagnosticAssets = diagnosticAssets,
                    assetPath = assets.sharedDecoder,
                    stage = stage,
                    backend = executionMode.tailBackend(stage),
                    threadCount = cpuThreadCount,
                    allowSourceQuantizedFcConvOps = allowSourceQuantizedFcConvOps,
                    constantTensorSharing = constantTensorSharing,
                    gpuBackend = gpuBackend,
                    strictGpuOnly = strictGpuOnly,
                )
                projection = InterpreterBoundaryProjection.open(
                    assetManager = diagnosticAssets,
                    assetPath = assets.activeRowNormalizeAndProject,
                    threadCount = cpuThreadCount,
                )
                LiteRtDecoderBoundaryDiagnostic(
                    prefix = prefix,
                    tail = tail,
                    projection = projection,
                    generationConfig = generationConfig,
                    stage = stage,
                    executionMode = executionMode,
                )
            } catch (error: Throwable) {
                runCatching { projection?.close() }
                runCatching { tail?.close() }
                prefix.close()
                throw error
            }
        }

        fun openCpuControl(
            configurationAssets: AssetManager,
            diagnosticAssets: AssetManager = configurationAssets,
            stage: WhisperDecoderBoundaryStage =
                WhisperDecoderBoundaryStage.AFTER_3_BLOCKS,
            assets: WhisperDecoderBoundaryAssets =
                WhisperDecoderBoundaryAssets.forStage(stage),
            cpuThreadCount: Int = LiteRtWhisperEngine.DEFAULT_THREAD_COUNT,
        ): LiteRtDecoderBoundaryDiagnostic =
            open(
                configurationAssets = configurationAssets,
                diagnosticAssets = diagnosticAssets,
                stage = stage,
                assets = assets,
                executionMode = WhisperDecoderBoundaryExecutionMode.ALL_CPU_CONTROL,
                cpuThreadCount = cpuThreadCount,
            )

        private fun openPrefix(
            diagnosticAssets: AssetManager,
            assetPath: String,
            stage: WhisperDecoderBoundaryStage,
            backend: WhisperSplitDecoderCoreBackend,
            threadCount: Int,
            allowSourceQuantizedFcConvOps: Boolean,
            constantTensorSharing: Boolean,
            gpuBackend: WhisperSplitDecoderGpuBackend,
            strictGpuOnly: Boolean,
        ): WhisperDecoderBoundaryPrefix =
            when (backend) {
                WhisperSplitDecoderCoreBackend.CPU_XNNPACK ->
                    InterpreterBoundaryPrefix.open(
                        diagnosticAssets,
                        assetPath,
                        stage,
                        threadCount,
                    )

                WhisperSplitDecoderCoreBackend.GPU_COMPILED_MODEL ->
                    GpuBoundaryPrefix.open(
                        diagnosticAssets,
                        assetPath,
                        constantTensorSharing,
                        allowSourceQuantizedFcConvOps,
                        gpuBackend,
                        strictGpuOnly,
                    )
            }

        private fun openTail(
            diagnosticAssets: AssetManager,
            assetPath: String,
            stage: WhisperDecoderBoundaryStage,
            backend: WhisperSplitDecoderCoreBackend,
            threadCount: Int,
            allowSourceQuantizedFcConvOps: Boolean,
            constantTensorSharing: Boolean,
            gpuBackend: WhisperSplitDecoderGpuBackend,
            strictGpuOnly: Boolean,
        ): WhisperDecoderBoundaryTail =
            when (backend) {
                WhisperSplitDecoderCoreBackend.CPU_XNNPACK ->
                    InterpreterBoundaryTail.open(
                        diagnosticAssets,
                        assetPath,
                        stage,
                        threadCount,
                    )

                WhisperSplitDecoderCoreBackend.GPU_COMPILED_MODEL ->
                    GpuBoundaryTail.open(
                        diagnosticAssets,
                        assetPath,
                        stage,
                        constantTensorSharing,
                        allowSourceQuantizedFcConvOps,
                        gpuBackend,
                        strictGpuOnly,
                    )
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

        private fun elapsedMillis(started: Long): Long =
            (System.nanoTime() - started) / NANOS_PER_MILLISECOND
    }
}

private class InterpreterBoundaryPrefix private constructor(
    private val interpreter: Interpreter,
    private val stage: WhisperDecoderBoundaryStage,
    private val bindings: BoundaryPrefixBindings,
    private val tailBindings: BoundaryTailBindings,
) : WhisperDecoderBoundaryPrefix, WhisperDecoderBoundaryCpuVerifier {
    private val encoder =
        boundaryDirectFloatBuffer(WhisperSplitDecoderTensors.ENCODER_STATE_FLOATS)
    private val tokens =
        boundaryDirectIntBuffer(WhisperSplitDecoderTensors.TARGET_LENGTH)
    private val mask = boundaryMaskBuffer()
    private val boundary =
        boundaryDirectFloatBuffer(WhisperSplitDecoderTensors.DECODER_HIDDEN_FLOATS)
    private val preLayerNorm =
        boundaryDirectFloatBuffer(WhisperSplitDecoderTensors.DECODER_HIDDEN_FLOATS)

    override fun run(
        encoderState: FloatArray,
        tokens: IntArray,
        shouldCancel: () -> Boolean,
    ): FloatArray {
        if (shouldCancel()) throw WhisperCancelledException()
        interpreter.setCancelled(false)
        encoder.putFloats(encoderState)
        this.tokens.putInts(tokens)
        mask.rewind()
        boundary.clear()
        interpreter.runSignature(
            mapOf(
                bindings.encoder to encoder,
                bindings.tokens to this.tokens,
                bindings.mask to mask,
            ),
            mapOf(bindings.boundary to boundary),
            BOUNDARY_PREFIX_SIGNATURE,
        )
        if (shouldCancel()) throw WhisperCancelledException()
        return boundary.readFloats(WhisperSplitDecoderTensors.DECODER_HIDDEN_FLOATS)
    }

    override fun runFullDecoder(
        encoderState: FloatArray,
        tokens: IntArray,
        shouldCancel: () -> Boolean,
    ): FloatArray {
        if (shouldCancel()) throw WhisperCancelledException()
        interpreter.setCancelled(false)
        encoder.putFloats(encoderState)
        this.tokens.putInts(tokens)
        mask.rewind()
        boundary.clear()
        interpreter.runSignature(
            mapOf(
                bindings.encoder to encoder,
                bindings.tokens to this.tokens,
                bindings.mask to mask,
            ),
            mapOf(bindings.boundary to boundary),
            BOUNDARY_PREFIX_SIGNATURE,
        )
        if (shouldCancel()) throw WhisperCancelledException()

        encoder.rewind()
        boundary.rewind()
        mask.rewind()
        preLayerNorm.clear()
        interpreter.runSignature(
            mapOf(
                tailBindings.encoder to encoder,
                tailBindings.boundary to boundary,
                tailBindings.mask to mask,
            ),
            mapOf(tailBindings.preLayerNorm to preLayerNorm),
            stage.tailSignature,
        )
        if (shouldCancel()) throw WhisperCancelledException()
        return preLayerNorm.readFloats(
            WhisperSplitDecoderTensors.DECODER_HIDDEN_FLOATS,
        )
    }

    override fun close() {
        interpreter.setCancelled(true)
        interpreter.close()
    }

    companion object {
        fun open(
            assetManager: AssetManager,
            assetPath: String,
            stage: WhisperDecoderBoundaryStage,
            threadCount: Int,
        ): InterpreterBoundaryPrefix {
            val interpreter = boundaryInterpreter(
                assetManager,
                assetPath,
                "decoder-boundary shared model",
                threadCount,
            )
            return try {
                InterpreterBoundaryPrefix(
                    interpreter = interpreter,
                    stage = stage,
                    bindings = resolveBoundaryPrefixBindings(interpreter),
                    tailBindings = resolveBoundaryTailBindings(interpreter, stage),
                )
            } catch (error: Throwable) {
                interpreter.close()
                throw error
            }
        }
    }
}

private class InterpreterBoundaryTail private constructor(
    private val interpreter: Interpreter,
    private val stage: WhisperDecoderBoundaryStage,
    private val prefixBindings: BoundaryPrefixBindings,
    private val bindings: BoundaryTailBindings,
) : WhisperDecoderBoundaryTail, WhisperDecoderBoundaryCpuVerifier {
    private val encoder =
        boundaryDirectFloatBuffer(WhisperSplitDecoderTensors.ENCODER_STATE_FLOATS)
    private val tokens =
        boundaryDirectIntBuffer(WhisperSplitDecoderTensors.TARGET_LENGTH)
    private val boundary =
        boundaryDirectFloatBuffer(WhisperSplitDecoderTensors.DECODER_HIDDEN_FLOATS)
    private val mask = boundaryMaskBuffer()
    private val preLayerNorm =
        boundaryDirectFloatBuffer(WhisperSplitDecoderTensors.DECODER_HIDDEN_FLOATS)

    override fun run(
        encoderState: FloatArray,
        boundary: FloatArray,
        shouldCancel: () -> Boolean,
    ): FloatArray {
        if (shouldCancel()) throw WhisperCancelledException()
        interpreter.setCancelled(false)
        encoder.putFloats(encoderState)
        this.boundary.putFloats(boundary)
        mask.rewind()
        preLayerNorm.clear()
        interpreter.runSignature(
            mapOf(
                bindings.encoder to encoder,
                bindings.boundary to this.boundary,
                bindings.mask to mask,
            ),
            mapOf(bindings.preLayerNorm to preLayerNorm),
            stage.tailSignature,
        )
        if (shouldCancel()) throw WhisperCancelledException()
        return preLayerNorm.readFloats(
            WhisperSplitDecoderTensors.DECODER_HIDDEN_FLOATS,
        )
    }

    override fun runFullDecoder(
        encoderState: FloatArray,
        tokens: IntArray,
        shouldCancel: () -> Boolean,
    ): FloatArray {
        if (shouldCancel()) throw WhisperCancelledException()
        interpreter.setCancelled(false)
        encoder.putFloats(encoderState)
        this.tokens.putInts(tokens)
        mask.rewind()
        boundary.clear()
        interpreter.runSignature(
            mapOf(
                prefixBindings.encoder to encoder,
                prefixBindings.tokens to this.tokens,
                prefixBindings.mask to mask,
            ),
            mapOf(prefixBindings.boundary to boundary),
            BOUNDARY_PREFIX_SIGNATURE,
        )
        if (shouldCancel()) throw WhisperCancelledException()

        encoder.rewind()
        boundary.rewind()
        mask.rewind()
        preLayerNorm.clear()
        interpreter.runSignature(
            mapOf(
                bindings.encoder to encoder,
                bindings.boundary to boundary,
                bindings.mask to mask,
            ),
            mapOf(bindings.preLayerNorm to preLayerNorm),
            stage.tailSignature,
        )
        if (shouldCancel()) throw WhisperCancelledException()
        return preLayerNorm.readFloats(
            WhisperSplitDecoderTensors.DECODER_HIDDEN_FLOATS,
        )
    }

    override fun close() {
        interpreter.setCancelled(true)
        interpreter.close()
    }

    companion object {
        fun open(
            assetManager: AssetManager,
            assetPath: String,
            stage: WhisperDecoderBoundaryStage,
            threadCount: Int,
        ): InterpreterBoundaryTail {
            val interpreter = boundaryInterpreter(
                assetManager,
                assetPath,
                "decoder-boundary shared model",
                threadCount,
            )
            return try {
                InterpreterBoundaryTail(
                    interpreter,
                    stage,
                    resolveBoundaryPrefixBindings(interpreter),
                    resolveBoundaryTailBindings(interpreter, stage),
                )
            } catch (error: Throwable) {
                interpreter.close()
                throw error
            }
        }
    }
}

/**
 * All-CPU control owner. Both signatures run through one Interpreter so the
 * 755 MB shared model is mapped and XNNPACK-initialized only once.
 */
private class InterpreterBoundaryPrefixTail private constructor(
    private val interpreter: Interpreter,
    private val stage: WhisperDecoderBoundaryStage,
    private val prefixBindings: BoundaryPrefixBindings,
    private val tailBindings: BoundaryTailBindings,
) : WhisperDecoderBoundaryPrefix, WhisperDecoderBoundaryTail {
    private val closed = AtomicBoolean(false)
    private val encoder =
        boundaryDirectFloatBuffer(WhisperSplitDecoderTensors.ENCODER_STATE_FLOATS)
    private val tokens =
        boundaryDirectIntBuffer(WhisperSplitDecoderTensors.TARGET_LENGTH)
    private val mask = boundaryMaskBuffer()
    private val boundary =
        boundaryDirectFloatBuffer(WhisperSplitDecoderTensors.DECODER_HIDDEN_FLOATS)
    private val preLayerNorm =
        boundaryDirectFloatBuffer(WhisperSplitDecoderTensors.DECODER_HIDDEN_FLOATS)

    override fun run(
        encoderState: FloatArray,
        tokens: IntArray,
        shouldCancel: () -> Boolean,
    ): FloatArray {
        check(!closed.get()) { "CPU decoder-boundary control is closed." }
        if (shouldCancel()) throw WhisperCancelledException()
        interpreter.setCancelled(false)
        encoder.putFloats(encoderState)
        this.tokens.putInts(tokens)
        mask.rewind()
        boundary.clear()
        interpreter.runSignature(
            mapOf(
                prefixBindings.encoder to encoder,
                prefixBindings.tokens to this.tokens,
                prefixBindings.mask to mask,
            ),
            mapOf(prefixBindings.boundary to boundary),
            BOUNDARY_PREFIX_SIGNATURE,
        )
        if (shouldCancel()) throw WhisperCancelledException()
        return boundary.readFloats(WhisperSplitDecoderTensors.DECODER_HIDDEN_FLOATS)
    }

    override fun run(
        encoderState: FloatArray,
        boundary: FloatArray,
        shouldCancel: () -> Boolean,
    ): FloatArray {
        check(!closed.get()) { "CPU decoder-boundary control is closed." }
        if (shouldCancel()) throw WhisperCancelledException()
        interpreter.setCancelled(false)
        encoder.putFloats(encoderState)
        this.boundary.putFloats(boundary)
        mask.rewind()
        preLayerNorm.clear()
        interpreter.runSignature(
            mapOf(
                tailBindings.encoder to encoder,
                tailBindings.boundary to this.boundary,
                tailBindings.mask to mask,
            ),
            mapOf(tailBindings.preLayerNorm to preLayerNorm),
            stage.tailSignature,
        )
        if (shouldCancel()) throw WhisperCancelledException()
        return preLayerNorm.readFloats(
            WhisperSplitDecoderTensors.DECODER_HIDDEN_FLOATS,
        )
    }

    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        interpreter.setCancelled(true)
        interpreter.close()
    }

    companion object {
        fun open(
            assetManager: AssetManager,
            assetPath: String,
            stage: WhisperDecoderBoundaryStage,
            threadCount: Int,
        ): InterpreterBoundaryPrefixTail {
            val interpreter = boundaryInterpreter(
                assetManager,
                assetPath,
                "decoder-boundary shared CPU-control model",
                threadCount,
            )
            return try {
                InterpreterBoundaryPrefixTail(
                    interpreter = interpreter,
                    stage = stage,
                    prefixBindings = resolveBoundaryPrefixBindings(interpreter),
                    tailBindings = resolveBoundaryTailBindings(interpreter, stage),
                )
            } catch (error: Throwable) {
                interpreter.close()
                throw error
            }
        }
    }
}

private class InterpreterBoundaryProjection private constructor(
    private val interpreter: Interpreter,
    private val hiddenInput: String,
    private val logitsOutput: String,
) : WhisperDecoderBoundaryProjection {
    private val hidden =
        boundaryDirectFloatBuffer(WhisperSplitDecoderTensors.ENCODER_WIDTH)
    private val logits =
        boundaryDirectFloatBuffer(WhisperSplitDecoderTensors.VOCABULARY_SIZE)

    override fun run(
        activePreLayerNorm: FloatArray,
        shouldCancel: () -> Boolean,
    ): FloatArray {
        require(activePreLayerNorm.size == WhisperSplitDecoderTensors.ENCODER_WIDTH) {
            "Active pre-layer-norm row has ${activePreLayerNorm.size} floats; expected " +
                "${WhisperSplitDecoderTensors.ENCODER_WIDTH}."
        }
        if (shouldCancel()) throw WhisperCancelledException()
        interpreter.setCancelled(false)
        hidden.putFloats(activePreLayerNorm)
        logits.clear()
        interpreter.runSignature(
            mapOf(hiddenInput to hidden),
            mapOf(logitsOutput to logits),
            BOUNDARY_PROJECTION_SIGNATURE,
        )
        if (shouldCancel()) throw WhisperCancelledException()
        return logits.readFloats(WhisperSplitDecoderTensors.VOCABULARY_SIZE)
    }

    override fun close() {
        interpreter.setCancelled(true)
        interpreter.close()
    }

    companion object {
        fun open(
            assetManager: AssetManager,
            assetPath: String,
            threadCount: Int,
        ): InterpreterBoundaryProjection {
            val interpreter = boundaryInterpreter(
                assetManager,
                assetPath,
                "active-row normalize-and-project model",
                threadCount,
            )
            return try {
                requireBoundarySignature(
                    interpreter,
                    BOUNDARY_PROJECTION_SIGNATURE,
                )
                val input = requireBoundaryInterpreterTensor(
                    interpreter = interpreter,
                    signature = BOUNDARY_PROJECTION_SIGNATURE,
                    input = true,
                    name = BOUNDARY_PROJECTION_INPUT,
                    type = DataType.FLOAT32,
                    shape = intArrayOf(1, WhisperSplitDecoderTensors.ENCODER_WIDTH),
                )
                val output = requireBoundaryInterpreterTensor(
                    interpreter = interpreter,
                    signature = BOUNDARY_PROJECTION_SIGNATURE,
                    input = false,
                    name = BOUNDARY_PROJECTION_OUTPUT,
                    type = DataType.FLOAT32,
                    shape = intArrayOf(
                        1,
                        WhisperSplitDecoderTensors.VOCABULARY_SIZE,
                    ),
                )
                InterpreterBoundaryProjection(interpreter, input, output)
            } catch (error: Throwable) {
                interpreter.close()
                throw error
            }
        }
    }
}

private class GpuBoundaryPrefix private constructor(
    private val model: CompiledModel,
    private val encoder: TensorBuffer,
    private val tokens: TensorBuffer,
    private val mask: TensorBuffer,
    private val boundary: TensorBuffer,
) : WhisperDecoderBoundaryPrefix {
    override fun run(
        encoderState: FloatArray,
        tokens: IntArray,
        shouldCancel: () -> Boolean,
    ): FloatArray {
        if (shouldCancel()) throw WhisperCancelledException()
        boundaryGpuOperation("writing boundary-prefix encoder state") {
            encoder.writeFloat(encoderState)
        }
        boundaryGpuOperation("writing boundary-prefix tokens") {
            this.tokens.writeInt(tokens)
        }
        boundaryGpuOperation("running decoder-boundary prefix") {
            model.run(
                mapOf(
                    BOUNDARY_PREFIX_ENCODER to encoder,
                    BOUNDARY_PREFIX_TOKENS to this.tokens,
                    BOUNDARY_PREFIX_MASK to mask,
                ),
                mapOf(BOUNDARY_PREFIX_OUTPUT to boundary),
                BOUNDARY_PREFIX_SIGNATURE,
            )
        }
        if (shouldCancel()) throw WhisperCancelledException()
        return boundaryGpuOperation("reading decoder-boundary prefix") {
            boundary.readFloat()
        }.also(::requireBoundaryHiddenSize)
    }

    override fun close() {
        listOf(boundary, mask, tokens, encoder).forEach { runCatching(it::close) }
        model.close()
    }

    companion object {
        fun open(
            assetManager: AssetManager,
            assetPath: String,
            constantTensorSharing: Boolean,
            allowSourceQuantizedFcConvOps: Boolean,
            gpuBackend: WhisperSplitDecoderGpuBackend,
            strictGpuOnly: Boolean,
        ): GpuBoundaryPrefix {
            val model = openBoundaryGpuModel(
                assetManager,
                assetPath,
                constantTensorSharing,
                allowSourceQuantizedFcConvOps,
                gpuBackend,
                strictGpuOnly,
            )
            val allocated = mutableListOf<TensorBuffer>()
            return try {
                validateGpuBoundaryPrefix(model)
                fun input(name: String) =
                    model.createInputBuffer(name, BOUNDARY_PREFIX_SIGNATURE)
                        .also(allocated::add)
                fun output(name: String) =
                    model.createOutputBuffer(name, BOUNDARY_PREFIX_SIGNATURE)
                        .also(allocated::add)

                val mask = input(BOUNDARY_PREFIX_MASK)
                mask.writeFloat(WhisperSplitDecoderTensors.causalMask())
                GpuBoundaryPrefix(
                    model = model,
                    encoder = input(BOUNDARY_PREFIX_ENCODER),
                    tokens = input(BOUNDARY_PREFIX_TOKENS),
                    mask = mask,
                    boundary = output(BOUNDARY_PREFIX_OUTPUT),
                )
            } catch (error: Throwable) {
                allocated.asReversed().forEach { runCatching(it::close) }
                model.close()
                throw error
            }
        }
    }
}

private class GpuBoundaryTail private constructor(
    private val model: CompiledModel,
    private val stage: WhisperDecoderBoundaryStage,
    private val encoder: TensorBuffer,
    private val boundary: TensorBuffer,
    private val mask: TensorBuffer,
    private val preLayerNorm: TensorBuffer,
) : WhisperDecoderBoundaryTail {
    override fun run(
        encoderState: FloatArray,
        boundary: FloatArray,
        shouldCancel: () -> Boolean,
    ): FloatArray {
        if (shouldCancel()) throw WhisperCancelledException()
        boundaryGpuOperation("writing boundary-tail encoder state") {
            encoder.writeFloat(encoderState)
        }
        boundaryGpuOperation("writing decoder boundary") {
            this.boundary.writeFloat(boundary)
        }
        boundaryGpuOperation("running decoder-boundary tail") {
            model.run(
                mapOf(
                    BOUNDARY_TAIL_ENCODER to encoder,
                    stage.boundaryInputName to this.boundary,
                    BOUNDARY_TAIL_MASK to mask,
                ),
                mapOf(BOUNDARY_TAIL_OUTPUT to preLayerNorm),
                stage.tailSignature,
            )
        }
        if (shouldCancel()) throw WhisperCancelledException()
        return boundaryGpuOperation("reading decoder-boundary tail") {
            preLayerNorm.readFloat()
        }.also(::requireBoundaryHiddenSize)
    }

    override fun close() {
        listOf(preLayerNorm, mask, boundary, encoder).forEach {
            runCatching(it::close)
        }
        model.close()
    }

    companion object {
        fun open(
            assetManager: AssetManager,
            assetPath: String,
            stage: WhisperDecoderBoundaryStage,
            constantTensorSharing: Boolean,
            allowSourceQuantizedFcConvOps: Boolean,
            gpuBackend: WhisperSplitDecoderGpuBackend,
            strictGpuOnly: Boolean,
        ): GpuBoundaryTail {
            val model = openBoundaryGpuModel(
                assetManager,
                assetPath,
                constantTensorSharing,
                allowSourceQuantizedFcConvOps,
                gpuBackend,
                strictGpuOnly,
            )
            val allocated = mutableListOf<TensorBuffer>()
            return try {
                validateGpuBoundaryTail(model, stage)
                fun input(name: String) =
                    model.createInputBuffer(name, stage.tailSignature)
                        .also(allocated::add)
                fun output(name: String) =
                    model.createOutputBuffer(name, stage.tailSignature)
                        .also(allocated::add)

                val mask = input(BOUNDARY_TAIL_MASK)
                mask.writeFloat(WhisperSplitDecoderTensors.causalMask())
                GpuBoundaryTail(
                    model = model,
                    stage = stage,
                    encoder = input(BOUNDARY_TAIL_ENCODER),
                    boundary = input(stage.boundaryInputName),
                    mask = mask,
                    preLayerNorm = output(BOUNDARY_TAIL_OUTPUT),
                )
            } catch (error: Throwable) {
                allocated.asReversed().forEach { runCatching(it::close) }
                model.close()
                throw error
            }
        }
    }
}

private data class BoundaryPrefixBindings(
    val encoder: String,
    val tokens: String,
    val mask: String,
    val boundary: String,
)

private data class BoundaryTailBindings(
    val encoder: String,
    val boundary: String,
    val mask: String,
    val preLayerNorm: String,
)

private const val BOUNDARY_PREFIX_SIGNATURE = "decode"
private const val BOUNDARY_PREFIX_ENCODER = "args_0"
private const val BOUNDARY_PREFIX_TOKENS = "args_1"
private const val BOUNDARY_PREFIX_MASK = "args_2"
private const val BOUNDARY_PREFIX_OUTPUT = "output_0"
private const val BOUNDARY_TAIL_ENCODER = "encoder_hidden_states"
private const val BOUNDARY_TAIL_MASK = "causal_mask"
private const val BOUNDARY_TAIL_OUTPUT = "hidden_pre_layer_norm"
private const val BOUNDARY_PROJECTION_SIGNATURE = "normalize_and_project"
private const val BOUNDARY_PROJECTION_INPUT = "hidden_pre_layer_norm"
private const val BOUNDARY_PROJECTION_OUTPUT = "logits"

private fun resolveBoundaryPrefixBindings(
    interpreter: Interpreter,
): BoundaryPrefixBindings {
    requireBoundarySignature(interpreter, BOUNDARY_PREFIX_SIGNATURE)
    return BoundaryPrefixBindings(
        encoder = requireBoundaryInterpreterTensor(
            interpreter,
            BOUNDARY_PREFIX_SIGNATURE,
            true,
            BOUNDARY_PREFIX_ENCODER,
            DataType.FLOAT32,
            intArrayOf(
                1,
                WhisperSplitDecoderTensors.ENCODER_FRAMES,
                WhisperSplitDecoderTensors.ENCODER_WIDTH,
            ),
        ),
        tokens = requireBoundaryInterpreterTensor(
            interpreter,
            BOUNDARY_PREFIX_SIGNATURE,
            true,
            BOUNDARY_PREFIX_TOKENS,
            DataType.INT32,
            intArrayOf(1, WhisperSplitDecoderTensors.TARGET_LENGTH),
        ),
        mask = requireBoundaryInterpreterTensor(
            interpreter,
            BOUNDARY_PREFIX_SIGNATURE,
            true,
            BOUNDARY_PREFIX_MASK,
            DataType.FLOAT32,
            boundaryMaskShape(),
        ),
        boundary = requireBoundaryInterpreterTensor(
            interpreter,
            BOUNDARY_PREFIX_SIGNATURE,
            false,
            BOUNDARY_PREFIX_OUTPUT,
            DataType.FLOAT32,
            boundaryHiddenShape(),
        ),
    )
}

private fun resolveBoundaryTailBindings(
    interpreter: Interpreter,
    stage: WhisperDecoderBoundaryStage,
): BoundaryTailBindings {
    requireBoundarySignature(interpreter, stage.tailSignature)
    return BoundaryTailBindings(
        encoder = requireBoundaryInterpreterTensor(
            interpreter,
            stage.tailSignature,
            true,
            BOUNDARY_TAIL_ENCODER,
            DataType.FLOAT32,
            intArrayOf(
                1,
                WhisperSplitDecoderTensors.ENCODER_FRAMES,
                WhisperSplitDecoderTensors.ENCODER_WIDTH,
            ),
        ),
        boundary = requireBoundaryInterpreterTensor(
            interpreter,
            stage.tailSignature,
            true,
            stage.boundaryInputName,
            DataType.FLOAT32,
            boundaryHiddenShape(),
        ),
        mask = requireBoundaryInterpreterTensor(
            interpreter,
            stage.tailSignature,
            true,
            BOUNDARY_TAIL_MASK,
            DataType.FLOAT32,
            boundaryMaskShape(),
        ),
        preLayerNorm = requireBoundaryInterpreterTensor(
            interpreter,
            stage.tailSignature,
            false,
            BOUNDARY_TAIL_OUTPUT,
            DataType.FLOAT32,
            boundaryHiddenShape(),
        ),
    )
}

private fun requireBoundarySignature(
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

private fun requireBoundaryInterpreterTensor(
    interpreter: Interpreter,
    signature: String,
    input: Boolean,
    name: String,
    type: DataType,
    shape: IntArray,
): String {
    val names = if (input) {
        interpreter.getSignatureInputs(signature)
    } else {
        interpreter.getSignatureOutputs(signature)
    }
    if (name !in names) {
        throw WhisperModelContractException(
            "$signature requires tensor '$name'; available names are ${names.toList()}.",
        )
    }
    val tensor = if (input) {
        interpreter.getInputTensorFromSignature(name, signature)
    } else {
        interpreter.getOutputTensorFromSignature(name, signature)
    }
    validateBoundaryInterpreterTensor("$signature/$name", tensor, type, shape)
    return name
}

private fun validateBoundaryInterpreterTensor(
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

private fun validateGpuBoundaryPrefix(model: CompiledModel) {
    validateBoundaryGpuTensor(
        "$BOUNDARY_PREFIX_SIGNATURE/$BOUNDARY_PREFIX_ENCODER",
        model.getInputTensorType(
            BOUNDARY_PREFIX_ENCODER,
            BOUNDARY_PREFIX_SIGNATURE,
        ),
        TensorType.ElementType.FLOAT,
        intArrayOf(
            1,
            WhisperSplitDecoderTensors.ENCODER_FRAMES,
            WhisperSplitDecoderTensors.ENCODER_WIDTH,
        ),
    )
    validateBoundaryGpuTensor(
        "$BOUNDARY_PREFIX_SIGNATURE/$BOUNDARY_PREFIX_TOKENS",
        model.getInputTensorType(
            BOUNDARY_PREFIX_TOKENS,
            BOUNDARY_PREFIX_SIGNATURE,
        ),
        TensorType.ElementType.INT,
        intArrayOf(1, WhisperSplitDecoderTensors.TARGET_LENGTH),
    )
    validateBoundaryGpuTensor(
        "$BOUNDARY_PREFIX_SIGNATURE/$BOUNDARY_PREFIX_MASK",
        model.getInputTensorType(BOUNDARY_PREFIX_MASK, BOUNDARY_PREFIX_SIGNATURE),
        TensorType.ElementType.FLOAT,
        boundaryMaskShape(),
    )
    validateBoundaryGpuTensor(
        "$BOUNDARY_PREFIX_SIGNATURE/$BOUNDARY_PREFIX_OUTPUT",
        model.getOutputTensorType(
            BOUNDARY_PREFIX_OUTPUT,
            BOUNDARY_PREFIX_SIGNATURE,
        ),
        TensorType.ElementType.FLOAT,
        boundaryHiddenShape(),
    )
}

private fun validateGpuBoundaryTail(
    model: CompiledModel,
    stage: WhisperDecoderBoundaryStage,
) {
    validateBoundaryGpuTensor(
        "${stage.tailSignature}/$BOUNDARY_TAIL_ENCODER",
        model.getInputTensorType(BOUNDARY_TAIL_ENCODER, stage.tailSignature),
        TensorType.ElementType.FLOAT,
        intArrayOf(
            1,
            WhisperSplitDecoderTensors.ENCODER_FRAMES,
            WhisperSplitDecoderTensors.ENCODER_WIDTH,
        ),
    )
    validateBoundaryGpuTensor(
        "${stage.tailSignature}/${stage.boundaryInputName}",
        model.getInputTensorType(stage.boundaryInputName, stage.tailSignature),
        TensorType.ElementType.FLOAT,
        boundaryHiddenShape(),
    )
    validateBoundaryGpuTensor(
        "${stage.tailSignature}/$BOUNDARY_TAIL_MASK",
        model.getInputTensorType(BOUNDARY_TAIL_MASK, stage.tailSignature),
        TensorType.ElementType.FLOAT,
        boundaryMaskShape(),
    )
    validateBoundaryGpuTensor(
        "${stage.tailSignature}/$BOUNDARY_TAIL_OUTPUT",
        model.getOutputTensorType(BOUNDARY_TAIL_OUTPUT, stage.tailSignature),
        TensorType.ElementType.FLOAT,
        boundaryHiddenShape(),
    )
}

private fun validateBoundaryGpuTensor(
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

private fun openBoundaryGpuModel(
    assetManager: AssetManager,
    assetPath: String,
    constantTensorSharing: Boolean,
    allowSourceQuantizedFcConvOps: Boolean,
    gpuBackend: WhisperSplitDecoderGpuBackend,
    strictGpuOnly: Boolean,
): CompiledModel {
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
                precision = CompiledModel.GpuOptions.Precision.FP32,
                backend = selection.backend,
            )
        }
    return CompiledModel.create(assetManager, assetPath, options)
}

private inline fun <T> boundaryGpuOperation(
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

private fun requireBoundaryHiddenSize(values: FloatArray) {
    if (values.size != WhisperSplitDecoderTensors.DECODER_HIDDEN_FLOATS) {
        throw WhisperGpuBackendException(
            "Decoder boundary returned ${values.size} floats; expected " +
                "${WhisperSplitDecoderTensors.DECODER_HIDDEN_FLOATS}.",
        )
    }
}

private fun boundaryInterpreter(
    assetManager: AssetManager,
    assetPath: String,
    label: String,
    threadCount: Int,
): Interpreter =
    Interpreter(
        mapBoundaryAsset(assetManager, assetPath, label),
        Interpreter.Options()
            .setNumThreads(threadCount)
            .setUseXNNPACK(true)
            .setCancellable(true),
    )

private fun mapBoundaryAsset(
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

private fun boundaryHiddenShape() = intArrayOf(
    1,
    WhisperSplitDecoderTensors.TARGET_LENGTH,
    WhisperSplitDecoderTensors.ENCODER_WIDTH,
)

private fun boundaryMaskShape() = intArrayOf(
    1,
    1,
    WhisperSplitDecoderTensors.TARGET_LENGTH,
    WhisperSplitDecoderTensors.TARGET_LENGTH,
)

private fun boundaryMaskBuffer(): ByteBuffer =
    boundaryDirectFloatBuffer(
        WhisperSplitDecoderTensors.TARGET_LENGTH *
            WhisperSplitDecoderTensors.TARGET_LENGTH,
    ).also { buffer ->
        buffer.asFloatBuffer().put(WhisperSplitDecoderTensors.causalMask())
        buffer.rewind()
    }

private fun boundaryDirectFloatBuffer(elements: Int): ByteBuffer =
    ByteBuffer.allocateDirect(Math.multiplyExact(elements, Float.SIZE_BYTES))
        .order(ByteOrder.nativeOrder())

private fun boundaryDirectIntBuffer(elements: Int): ByteBuffer =
    ByteBuffer.allocateDirect(Math.multiplyExact(elements, Int.SIZE_BYTES))
        .order(ByteOrder.nativeOrder())

private fun ByteBuffer.putFloats(values: FloatArray) {
    clear()
    asFloatBuffer().put(values)
    rewind()
}

private fun ByteBuffer.putInts(values: IntArray) {
    clear()
    asIntBuffer().put(values)
    rewind()
}

private fun ByteBuffer.readFloats(count: Int): FloatArray {
    rewind()
    return FloatArray(count).also { values ->
        asFloatBuffer().get(values)
        rewind()
    }
}
