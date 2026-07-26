package com.example.voicememo.transcription

import android.content.Context
import android.content.res.AssetManager
import android.os.Trace
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import com.google.ai.edge.litert.TensorType
import java.util.concurrent.atomic.AtomicBoolean

/**
 * LiteRT GPU-dominant parity candidate for the pinned INT4/INT8 model.
 *
 * The decoder retains one CPU GATHER operation. This engine is diagnostic-only
 * until its exact token stream has passed the CPU reference suite. Normal app
 * transcription never calls it, so divergent probe output cannot be persisted.
 */
internal class LiteRtFullGpuWhisperEngine private constructor(
    private val model: CompiledModel,
    private val tokenizer: WhisperTokenizer,
    private val generationConfig: WhisperGenerationConfig,
    private val buffers: InferenceBuffers,
    private val featureExtractor: WhisperFeatureExtractor,
    private val timestampsEnabled: Boolean,
) : WhisperEngine {
    private val closed = AtomicBoolean(false)
    private val inferenceLock = Any()
    private val tokenInput = IntArray(TARGET_LENGTH)

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
        val encoderState = encodeAudio(
            audio = audio,
            shouldCancel = shouldCancel,
        )
        decodePreparedEncoderState(
            encoderState = encoderState,
            language = language,
            shouldCancel = shouldCancel,
            onTokenProgress = onTokenProgress,
            startedNanos = started,
            audioDurationMillis = audio.size * 1_000L / WhisperFeatureExtractor.SAMPLE_RATE,
        )
    }

    /**
     * Produces the GPU encoder state without decoding. Diagnostic tests use
     * this to quantify the first accelerator boundary independently.
     */
    internal fun encodeChunk(
        audio: FloatArray,
        shouldCancel: () -> Boolean = { false },
    ): FloatArray = synchronized(inferenceLock) {
        check(!closed.get()) { "WhisperEngine is closed." }
        require(audio.size <= WhisperFeatureExtractor.MAX_SAMPLES) {
            "encodeChunk accepts at most 30 seconds of 16 kHz PCM."
        }
        encodeAudio(
            audio = audio,
            shouldCancel = shouldCancel,
        )
    }

    /**
     * Runs only the GPU-dominant decoder against a caller-provided encoder
     * state. It exists solely to isolate decoder parity from encoder drift.
     */
    internal fun transcribeEncodedChunk(
        encoderState: FloatArray,
        language: WhisperLanguage,
        shouldCancel: () -> Boolean = { false },
        onTokenProgress: (generated: Int, limit: Int) -> Unit = { _, _ -> },
        startedNanos: Long = System.nanoTime(),
    ): WhisperChunkResult = synchronized(inferenceLock) {
        check(!closed.get()) { "WhisperEngine is closed." }
        decodePreparedEncoderState(
            encoderState = encoderState,
            language = language,
            shouldCancel = shouldCancel,
            onTokenProgress = onTokenProgress,
            startedNanos = startedNanos,
            audioDurationMillis = MAX_AUDIO_DURATION_MILLIS,
        )
    }

    /**
     * Computes only the active decoder-logit row for a known-good token
     * prefix. Diagnostic tests use this with the CPU encoder state to locate
     * the first numerical GPU decoder divergence without persisting text.
     */
    internal fun teacherForcedLogitsRow(
        encoderState: FloatArray,
        language: WhisperLanguage.Forced,
        referenceTokenPrefix: List<Int>,
        shouldCancel: () -> Boolean = { false },
    ): WhisperTeacherForcedLogitsRow = synchronized(inferenceLock) {
        check(!closed.get()) { "WhisperEngine is closed." }
        require(encoderState.size == ENCODER_STATE_FLOATS) {
            "Whisper encoder state has ${encoderState.size} floats; expected " +
                "$ENCODER_STATE_FLOATS."
        }
        val prompt = teacherForcedPrompt(language, referenceTokenPrefix)
        checkCancellation(shouldCancel)
        gpuOperation("writing teacher-forced encoder state") {
            buffers.decoderEncoderState.writeFloat(encoderState)
        }
        prepareTokens(prompt, generationConfig.eosTokenId)
        gpuOperation("writing teacher-forced decoder tokens") {
            buffers.tokens.writeInt(tokenInput)
        }
        tracedGpu("WhisperGpuTeacherForcedDecodeStep", "teacher-forced decoder invocation") {
            model.run(
                mapOf(
                    DECODE_ENCODER_STATE to buffers.decoderEncoderState,
                    DECODE_TOKENS to buffers.tokens,
                    DECODE_MASK to buffers.causalMask,
                ),
                mapOf(DECODE_OUTPUT to buffers.logits),
                DECODE_SIGNATURE,
            )
        }
        checkCancellation(shouldCancel)

        val logits = gpuOperation("reading teacher-forced decoder logits") {
            buffers.logits.readFloat()
        }
        if (logits.size != LOGITS_FLOATS) {
            throw WhisperGpuBackendException(
                "GPU decoder returned ${logits.size} floats; expected $LOGITS_FLOATS.",
            )
        }
        val rowStart = prompt.lastIndex * VOCAB_SIZE
        WhisperTeacherForcedTokenRanking.forGpu(
            logits = logits.copyOfRange(rowStart, rowStart + VOCAB_SIZE),
            generationConfig = generationConfig,
            firstGeneratedToken = referenceTokenPrefix.isEmpty(),
        )
    }

    /**
     * Proposes a short autoregressive block from a CPU-verified prefix.
     *
     * This method is diagnostic-only. Its output is not safe to persist until
     * [LiteRtWhisperEngine.verifyProposedTokenBlock] has approved it.
     */
    internal fun proposeTokenBlock(
        encoderState: FloatArray,
        language: WhisperLanguage.Forced,
        verifiedTokenPrefix: List<Int>,
        maximumTokenCount: Int,
        shouldCancel: () -> Boolean = { false },
    ): WhisperGpuProposedTokenBlock = synchronized(inferenceLock) {
        check(!closed.get()) { "WhisperEngine is closed." }
        require(encoderState.size == ENCODER_STATE_FLOATS) {
            "Whisper encoder state has ${encoderState.size} floats; expected " +
                "$ENCODER_STATE_FLOATS."
        }
        require(maximumTokenCount > 0) {
            "GPU speculative block size must be positive."
        }
        val prompt = teacherForcedPrompt(language, verifiedTokenPrefix).toMutableList()
        val allowedTokenCount = minOf(maximumTokenCount, TARGET_LENGTH - prompt.size)
        require(allowedTokenCount > 0) {
            "The verified prefix already fills the decoder context."
        }
        checkCancellation(shouldCancel)
        gpuOperation("writing speculative decoder encoder state") {
            buffers.decoderEncoderState.writeFloat(encoderState)
        }

        val proposed = mutableListOf<Int>()
        val startedNanos = System.nanoTime()
        while (proposed.size < allowedTokenCount) {
            checkCancellation(shouldCancel)
            prepareTokens(prompt, generationConfig.eosTokenId)
            gpuOperation("writing speculative decoder tokens") {
                buffers.tokens.writeInt(tokenInput)
            }
            tracedGpu("WhisperGpuSpeculativeDecodeStep", "speculative decoder invocation") {
                model.run(
                    mapOf(
                        DECODE_ENCODER_STATE to buffers.decoderEncoderState,
                        DECODE_TOKENS to buffers.tokens,
                        DECODE_MASK to buffers.causalMask,
                    ),
                    mapOf(DECODE_OUTPUT to buffers.logits),
                    DECODE_SIGNATURE,
                )
            }
            checkCancellation(shouldCancel)
            val logits = gpuOperation("reading speculative decoder logits") {
                buffers.logits.readFloat()
            }
            if (logits.size != LOGITS_FLOATS) {
                throw WhisperGpuBackendException(
                    "GPU decoder returned ${logits.size} floats; expected $LOGITS_FLOATS.",
                )
            }
            val generatedTokens = verifiedTokenPrefix + proposed
            val nextToken = selectToken(
                logits = logits,
                position = prompt.lastIndex,
                firstGeneratedToken = generatedTokens.isEmpty(),
                generatedTokens = generatedTokens,
            )
            proposed += nextToken
            if (nextToken == generationConfig.eosTokenId) break
            prompt += nextToken
        }
        WhisperGpuProposedTokenBlock(
            tokens = proposed,
            decoderInvocations = proposed.size,
            inferenceMillis = (System.nanoTime() - startedNanos) / 1_000_000L,
        )
    }

    private fun encodeAudio(
        audio: FloatArray,
        shouldCancel: () -> Boolean,
    ): FloatArray {
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

        // Signature-created buffers can have different accelerator storage
        // requirements even when their logical FP32 tensor shapes match.
        val encoderState = gpuOperation("reading encoder state") {
            buffers.encoderState.readFloat()
        }
        if (encoderState.size != ENCODER_STATE_FLOATS) {
            throw WhisperGpuBackendException(
                "GPU encoder returned ${encoderState.size} floats; expected " +
                    "$ENCODER_STATE_FLOATS.",
            )
        }
        return encoderState
    }

    private fun decodePreparedEncoderState(
        encoderState: FloatArray,
        language: WhisperLanguage,
        shouldCancel: () -> Boolean,
        onTokenProgress: (generated: Int, limit: Int) -> Unit,
        startedNanos: Long,
        audioDurationMillis: Long,
    ): WhisperChunkResult {
        require(encoderState.size == ENCODER_STATE_FLOATS) {
            "Whisper encoder state has ${encoderState.size} floats; expected " +
                "$ENCODER_STATE_FLOATS."
        }
        checkCancellation(shouldCancel)
        gpuOperation("writing decoder encoder state") {
            buffers.decoderEncoderState.writeFloat(encoderState)
        }

        val prompt = mutableListOf(generationConfig.decoderStartTokenId)
        var detectedLanguage: String? = null
        when (language) {
            WhisperLanguage.Auto -> Unit
            is WhisperLanguage.Forced -> {
                val code = language.code.lowercase()
                prompt += generationConfig.languageToken(code)
                prompt += generationConfig.transcribeTokenId
                if (!timestampsEnabled) prompt += generationConfig.noTimestampsTokenId
                detectedLanguage = code
            }
        }

        val generated = mutableListOf<Int>()
        var reachedEos = false
        val generationLimit = TARGET_LENGTH - standardPromptLength()
        onTokenProgress(0, generationLimit)

        while (prompt.size < TARGET_LENGTH) {
            checkCancellation(shouldCancel)
            prepareTokens(prompt, generationConfig.eosTokenId)
            gpuOperation("writing decoder tokens") {
                buffers.tokens.writeInt(tokenInput)
            }
            tracedGpu("WhisperGpuDecodeStep", "decoder invocation") {
                model.run(
                    mapOf(
                        DECODE_ENCODER_STATE to buffers.decoderEncoderState,
                        DECODE_TOKENS to buffers.tokens,
                        DECODE_MASK to buffers.causalMask,
                    ),
                    mapOf(DECODE_OUTPUT to buffers.logits),
                    DECODE_SIGNATURE,
                )
            }
            checkCancellation(shouldCancel)

            val logits = gpuOperation("reading decoder logits") {
                buffers.logits.readFloat()
            }
            if (logits.size != LOGITS_FLOATS) {
                throw WhisperGpuBackendException(
                    "GPU decoder returned ${logits.size} floats; expected $LOGITS_FLOATS.",
                )
            }
            val logitsPosition = prompt.lastIndex
            if (language is WhisperLanguage.Auto && detectedLanguage == null) {
                val languageToken = WhisperGpuGreedyTokenSelector.selectLanguageToken(
                    logits = logits,
                    position = logitsPosition,
                    vocabularySize = VOCAB_SIZE,
                    languageTokenIds = generationConfig.idToLanguage.keys,
                )
                detectedLanguage = generationConfig.idToLanguage[languageToken]
                    ?: throw WhisperModelContractException(
                        "Decoder selected unknown language token $languageToken.",
                    )
                prompt += languageToken
                prompt += generationConfig.transcribeTokenId
                if (!timestampsEnabled) prompt += generationConfig.noTimestampsTokenId
                continue
            }

            val nextToken = selectToken(
                logits = logits,
                position = logitsPosition,
                firstGeneratedToken = generated.isEmpty(),
                generatedTokens = generated,
            )
            if (nextToken == generationConfig.eosTokenId) {
                reachedEos = true
                break
            }
            prompt += nextToken
            generated += nextToken
            onTokenProgress(generated.size, generationLimit)
        }

        val truncated = !reachedEos && prompt.size >= TARGET_LENGTH
        return WhisperChunkResult(
            tokens = generated,
            text = decodeTokens(generated).trim(),
            detectedLanguage = detectedLanguage ?: "und",
            inferenceMillis = (System.nanoTime() - startedNanos) / 1_000_000L,
            truncated = truncated,
            warnings = if (truncated) {
                listOf("Decoder reached its 128-token limit before EOS.")
            } else {
                emptyList()
            },
            timedSegments = if (timestampsEnabled) {
                WhisperTimestampTokens.parseSegments(
                    tokens = generated,
                    timestampBeginTokenId = generationConfig.timestampBeginTokenId,
                    tokenizer = tokenizer,
                    audioDurationMillis = audioDurationMillis,
                )
            } else {
                emptyList()
            },
        )
    }

    override fun decodeTokens(tokens: List<Int>): String = tokenizer.decode(
        WhisperTimestampTokens.withoutTimestamps(
            tokens = tokens,
            timestampBeginTokenId = generationConfig.timestampBeginTokenId,
        ),
    )

    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        synchronized(inferenceLock) {
            buffers.close()
            model.close()
        }
    }

    private fun checkCancellation(shouldCancel: () -> Boolean) {
        if (closed.get() || shouldCancel()) throw WhisperCancelledException()
    }

    private fun prepareTokens(prompt: List<Int>, paddingToken: Int) {
        repeat(TARGET_LENGTH) { index ->
            tokenInput[index] = if (index < prompt.size) prompt[index] else paddingToken
        }
    }

    private fun teacherForcedPrompt(
        language: WhisperLanguage.Forced,
        referenceTokenPrefix: List<Int>,
    ): List<Int> {
        require(referenceTokenPrefix.size < TARGET_LENGTH - standardPromptLength()) {
            "Teacher-forced prefix has ${referenceTokenPrefix.size} tokens; expected fewer than " +
                "${TARGET_LENGTH - standardPromptLength()}."
        }
        require(referenceTokenPrefix.all { it in 0 until VOCAB_SIZE }) {
            "Teacher-forced prefix contains an out-of-vocabulary token."
        }
        return buildList(standardPromptLength() + referenceTokenPrefix.size) {
            add(generationConfig.decoderStartTokenId)
            add(generationConfig.languageToken(language.code))
            add(generationConfig.transcribeTokenId)
            if (!timestampsEnabled) add(generationConfig.noTimestampsTokenId)
            addAll(referenceTokenPrefix)
        }
    }

    private fun selectToken(
        logits: FloatArray,
        position: Int,
        firstGeneratedToken: Boolean,
        generatedTokens: List<Int>,
    ): Int {
        if (timestampsEnabled) {
            val rowStart = position * VOCAB_SIZE
            return WhisperTimestampTokenSelector.select(
                vocabularySize = VOCAB_SIZE,
                generatedTokens = generatedTokens,
                config = generationConfig,
                firstGeneratedToken = firstGeneratedToken,
                logitAt = { tokenId ->
                    val index = rowStart + tokenId
                    if (index !in logits.indices) {
                        throw WhisperGpuBackendException(
                            "GPU decoder returned ${logits.size} logits; expected at least " +
                                "${(position + 1) * VOCAB_SIZE}.",
                        )
                    }
                    logits[index]
                },
                failure = { message -> WhisperGpuBackendException(message) },
            )
        }
        return WhisperGpuGreedyTokenSelector.selectTextToken(
            logits = logits,
            position = position,
            vocabularySize = VOCAB_SIZE,
            decoderStartTokenId = generationConfig.decoderStartTokenId,
            eosTokenId = generationConfig.eosTokenId,
            suppressTokens = generationConfig.suppressTokens,
            beginSuppressTokens = generationConfig.beginSuppressTokens,
            firstGeneratedToken = firstGeneratedToken,
        )
    }

    private fun standardPromptLength(): Int =
        if (timestampsEnabled) TIMESTAMP_PROMPT_LENGTH else STANDARD_PROMPT_LENGTH

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
        val decoderEncoderState: TensorBuffer,
        val tokens: TensorBuffer,
        val causalMask: TensorBuffer,
        val logits: TensorBuffer,
    ) : AutoCloseable {
        override fun close() {
            listOf(
                logits,
                causalMask,
                tokens,
                decoderEncoderState,
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
                    val mask = FloatArray(TARGET_LENGTH * TARGET_LENGTH)
                    for (row in 0 until TARGET_LENGTH) {
                        for (column in 0 until TARGET_LENGTH) {
                            mask[row * TARGET_LENGTH + column] =
                                if (column <= row) 0.0f else -1.0e9f
                        }
                    }
                    val causalMask = input(DECODE_MASK, DECODE_SIGNATURE)
                    causalMask.writeFloat(mask)
                    InferenceBuffers(
                        features = input(ENCODE_INPUT, ENCODE_SIGNATURE),
                        encoderState = output(ENCODE_OUTPUT, ENCODE_SIGNATURE),
                        decoderEncoderState =
                            input(DECODE_ENCODER_STATE, DECODE_SIGNATURE),
                        tokens = input(DECODE_TOKENS, DECODE_SIGNATURE),
                        causalMask = causalMask,
                        logits = output(DECODE_OUTPUT, DECODE_SIGNATURE),
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
        private const val TARGET_LENGTH = 128
        private const val VOCAB_SIZE = 51_866
        private const val STANDARD_PROMPT_LENGTH = 4
        private const val TIMESTAMP_PROMPT_LENGTH = 3
        private const val MAX_AUDIO_DURATION_MILLIS = 30_000L
        private const val ENCODER_STATE_FLOATS = ENCODER_FRAMES * ENCODER_WIDTH
        private const val LOGITS_FLOATS = TARGET_LENGTH * VOCAB_SIZE

        fun open(
            context: Context,
            diagnosticHintWaitingForCompletion: Boolean = false,
            diagnosticKernelBatchSize: Int? = null,
            allowSourceQuantizedFcConvOps: Boolean = true,
            strictGpuOnly: Boolean = false,
            diagnosticBackend: CompiledModel.GpuOptions.Backend =
                CompiledModel.GpuOptions.Backend.OPENCL,
            diagnosticBufferStorageType: CompiledModel.GpuOptions.BufferStorageType =
                CompiledModel.GpuOptions.BufferStorageType.DEFAULT,
            diagnosticPreferTextureWeights: Boolean? = null,
            diagnosticBenchmarkMode: Boolean = false,
            diagnosticModelAssets: AssetManager? = null,
            diagnosticModelPath: String? = null,
            timestampsEnabled: Boolean = false,
        ): LiteRtFullGpuWhisperEngine {
            require(diagnosticKernelBatchSize == null || diagnosticKernelBatchSize > 0) {
                "GPU kernel batch size must be positive."
            }
            require((diagnosticModelAssets == null) == (diagnosticModelPath == null)) {
                "Diagnostic model assets and path must be supplied together."
            }
            val applicationContext = context.applicationContext
            val tokenizer = applicationContext.assets.open(WhisperAssets.TOKENIZER).use {
                WhisperTokenizer.load(it)
            }
            val config = applicationContext.assets.open(WhisperAssets.GENERATION_CONFIG).use {
                WhisperGenerationConfig.load(it)
            }
            validateConfig(config)

            val accelerators = if (strictGpuOnly) {
                arrayOf(Accelerator.GPU)
            } else {
                arrayOf(Accelerator.GPU, Accelerator.CPU)
            }
            val options = CompiledModel.Options(*accelerators).apply {
                gpuOptions = CompiledModel.GpuOptions(
                    constantTensorSharing = true,
                    allowSrcQuantizedFcConvOps = allowSourceQuantizedFcConvOps,
                    precision = CompiledModel.GpuOptions.Precision.FP32,
                    backend = diagnosticBackend,
                    bufferStorageType = diagnosticBufferStorageType,
                    preferTextureWeights = diagnosticPreferTextureWeights,
                    modelCacheKey = diagnosticHiddenRuntimeOptions(
                        hintWaitingForCompletion = diagnosticHintWaitingForCompletion,
                        kernelBatchSize = diagnosticKernelBatchSize,
                        benchmarkMode = diagnosticBenchmarkMode,
                    ),
                )
            }
            val modelAssets = diagnosticModelAssets ?: applicationContext.assets
            val modelPath = diagnosticModelPath ?: WhisperAssets.MODEL
            val model = CompiledModel.create(
                modelAssets,
                modelPath,
                options,
            )
            return try {
                validateModelContract(model)
                LiteRtFullGpuWhisperEngine(
                    model = model,
                    tokenizer = tokenizer,
                    generationConfig = config,
                    buffers = InferenceBuffers.allocate(model),
                    featureExtractor = WhisperFeatureExtractor(),
                    timestampsEnabled = timestampsEnabled,
                )
            } catch (error: Throwable) {
                model.close()
                throw error
            }
        }

        /**
         * LiteRT 2.1.6 exposes these runtime options in C but omits them from
         * its Kotlin API. Its diagnostic parser is line based and does not
         * escape string values, so a test-only model cache key can inject the
         * missing settings. Never use this path for production inference; a
         * passing experiment must be implemented through a proper API bridge.
         */
        private fun diagnosticHiddenRuntimeOptions(
            hintWaitingForCompletion: Boolean,
            kernelBatchSize: Int?,
            benchmarkMode: Boolean,
        ): String? {
            if (
                !hintWaitingForCompletion &&
                kernelBatchSize == null &&
                !benchmarkMode
            ) {
                return null
            }
            return buildString {
                append("whisper-diagnostic\"")
                if (hintWaitingForCompletion) {
                    append("\nhint_waiting_for_completion = true")
                }
                if (kernelBatchSize != null) {
                    append("\nkernel_batch_size = ")
                    append(kernelBatchSize)
                }
                if (benchmarkMode) {
                    append("\nbenchmark_mode = true")
                }
                append("\nmodel_cache_key = \"")
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
                "$ENCODE_SIGNATURE/$ENCODE_INPUT",
                model.getInputTensorType(ENCODE_INPUT, ENCODE_SIGNATURE),
                TensorType.ElementType.FLOAT,
                intArrayOf(1, 128, 3_000),
            )
            validateTensor(
                "$ENCODE_SIGNATURE/$ENCODE_OUTPUT",
                model.getOutputTensorType(ENCODE_OUTPUT, ENCODE_SIGNATURE),
                TensorType.ElementType.FLOAT,
                intArrayOf(1, ENCODER_FRAMES, ENCODER_WIDTH),
            )
            validateTensor(
                "$DECODE_SIGNATURE/$DECODE_ENCODER_STATE",
                model.getInputTensorType(DECODE_ENCODER_STATE, DECODE_SIGNATURE),
                TensorType.ElementType.FLOAT,
                intArrayOf(1, ENCODER_FRAMES, ENCODER_WIDTH),
            )
            validateTensor(
                "$DECODE_SIGNATURE/$DECODE_TOKENS",
                model.getInputTensorType(DECODE_TOKENS, DECODE_SIGNATURE),
                TensorType.ElementType.INT,
                intArrayOf(1, TARGET_LENGTH),
            )
            validateTensor(
                "$DECODE_SIGNATURE/$DECODE_MASK",
                model.getInputTensorType(DECODE_MASK, DECODE_SIGNATURE),
                TensorType.ElementType.FLOAT,
                intArrayOf(1, 1, TARGET_LENGTH, TARGET_LENGTH),
            )
            validateTensor(
                "$DECODE_SIGNATURE/$DECODE_OUTPUT",
                model.getOutputTensorType(DECODE_OUTPUT, DECODE_SIGNATURE),
                TensorType.ElementType.FLOAT,
                intArrayOf(1, TARGET_LENGTH, VOCAB_SIZE),
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
