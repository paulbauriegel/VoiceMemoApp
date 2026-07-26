package com.example.voicememo.transcription

import android.content.Context
import android.os.Trace
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.atomic.AtomicBoolean

class LiteRtWhisperEngine private constructor(
    private val interpreter: Interpreter,
    private val tokenizer: WhisperTokenizer,
    private val generationConfig: WhisperGenerationConfig,
    private val bindings: Bindings,
    private val featureExtractor: WhisperFeatureExtractor,
    private val timestampsEnabled: Boolean,
) : WhisperEngine {
    private val closed = AtomicBoolean(false)
    private val inferenceLock = Any()
    private var buffers: InferenceBuffers? = null

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
        interpreter.setCancelled(false)
        val started = System.nanoTime()
        val reusable = buffers ?: InferenceBuffers.allocate().also { buffers = it }

        encodeIntoReusable(
            audio = audio,
            reusable = reusable,
            shouldCancel = shouldCancel,
        )

        decodePreparedEncoderState(
            reusable = reusable,
            language = language,
            shouldCancel = shouldCancel,
            onTokenProgress = onTokenProgress,
            startedNanos = started,
            audioDurationMillis = audio.size * 1_000L / WhisperFeatureExtractor.SAMPLE_RATE,
        )
    }

    /**
     * Produces the reference Interpreter/XNNPACK encoder state without running
     * the decoder. This is exposed only for non-persisting accelerator parity
     * diagnostics.
     */
    internal fun encodeChunk(
        audio: FloatArray,
        shouldCancel: () -> Boolean = { false },
    ): FloatArray = synchronized(inferenceLock) {
        check(!closed.get()) { "WhisperEngine is closed." }
        require(audio.size <= WhisperFeatureExtractor.MAX_SAMPLES) {
            "encodeChunk accepts at most 30 seconds of 16 kHz PCM."
        }
        interpreter.setCancelled(false)
        val reusable = buffers ?: InferenceBuffers.allocate().also { buffers = it }
        encodeIntoReusable(
            audio = audio,
            reusable = reusable,
            shouldCancel = shouldCancel,
        )
        FloatArray(ENCODER_FRAMES * ENCODER_WIDTH).also { state ->
            reusable.encoderState.asFloatBuffer().get(state)
            reusable.encoderState.rewind()
        }
    }

    /**
     * Runs the reference XNNPACK decoder against an encoder state produced by
     * another backend. The experimental GPU engine uses this to keep the
     * encoder accelerated without accepting a GPU decoder's divergent
     * autoregressive token stream.
     */
    internal fun transcribeEncodedChunk(
        encoderState: FloatArray,
        language: WhisperLanguage,
        shouldCancel: () -> Boolean = { false },
        onTokenProgress: (generated: Int, limit: Int) -> Unit = { _, _ -> },
        startedNanos: Long = System.nanoTime(),
    ): WhisperChunkResult = synchronized(inferenceLock) {
        check(!closed.get()) { "WhisperEngine is closed." }
        require(encoderState.size == ENCODER_FRAMES * ENCODER_WIDTH) {
            "Whisper encoder state has ${encoderState.size} floats; expected " +
                "${ENCODER_FRAMES * ENCODER_WIDTH}."
        }
        interpreter.setCancelled(false)
        checkCancellation(shouldCancel)
        val reusable = buffers ?: InferenceBuffers.allocate().also { buffers = it }
        reusable.encoderState.clear()
        reusable.encoderState.asFloatBuffer().put(encoderState)
        reusable.encoderState.rewind()

        decodePreparedEncoderState(
            reusable = reusable,
            language = language,
            shouldCancel = shouldCancel,
            onTokenProgress = onTokenProgress,
            startedNanos = startedNanos,
            audioDurationMillis = MAX_AUDIO_DURATION_MILLIS,
        )
    }

    /**
     * Computes only the active decoder-logit row for a known-good token
     * prefix. This is a non-persisting diagnostic boundary: it shares the
     * normal decoder buffers and signature invocation, but cannot affect
     * autoregressive production transcription.
     */
    internal fun teacherForcedLogitsRow(
        encoderState: FloatArray,
        language: WhisperLanguage.Forced,
        referenceTokenPrefix: List<Int>,
        shouldCancel: () -> Boolean = { false },
    ): WhisperTeacherForcedLogitsRow = synchronized(inferenceLock) {
        check(!closed.get()) { "WhisperEngine is closed." }
        require(encoderState.size == ENCODER_FRAMES * ENCODER_WIDTH) {
            "Whisper encoder state has ${encoderState.size} floats; expected " +
                "${ENCODER_FRAMES * ENCODER_WIDTH}."
        }
        val prompt = teacherForcedPrompt(language, referenceTokenPrefix)
        interpreter.setCancelled(false)
        checkCancellation(shouldCancel)
        val reusable = buffers ?: InferenceBuffers.allocate().also { buffers = it }
        reusable.encoderState.clear()
        reusable.encoderState.asFloatBuffer().put(encoderState)
        reusable.encoderState.rewind()
        prepareTokens(reusable.tokens, prompt, generationConfig.eosTokenId)
        reusable.encoderState.rewind()
        reusable.tokens.rewind()
        reusable.causalMask.rewind()
        reusable.logits.clear()

        traced("WhisperTeacherForcedDecodeStep") {
            interpreter.runSignature(
                mapOf(
                    bindings.decodeEncoderState to reusable.encoderState,
                    bindings.decodeTokens to reusable.tokens,
                    bindings.decodeMask to reusable.causalMask,
                ),
                mapOf(bindings.decodeOutput to reusable.logits),
                DECODE_SIGNATURE,
            )
        }
        checkCancellation(shouldCancel)

        reusable.logits.rewind()
        val row = FloatArray(VOCAB_SIZE).also { values ->
            reusable.logits.asFloatBuffer().apply {
                position(prompt.lastIndex * VOCAB_SIZE)
                get(values)
            }
            reusable.logits.rewind()
        }
        WhisperTeacherForcedTokenRanking.forCpu(
            logits = row,
            generationConfig = generationConfig,
            firstGeneratedToken = referenceTokenPrefix.isEmpty(),
        )
    }

    /**
     * Verifies several GPU-proposed tokens with one causal CPU decoder call.
     *
     * Output row `basePrompt.lastIndex + offset` predicts proposal `offset`.
     * Because the supplied attention mask is causal, each such row can see
     * only the verified prefix and earlier proposals. Callers must discard all
     * rows after the first mismatch.
     */
    internal fun verifyProposedTokenBlock(
        encoderState: FloatArray,
        language: WhisperLanguage.Forced,
        verifiedTokenPrefix: List<Int>,
        proposedTokens: List<Int>,
        shouldCancel: () -> Boolean = { false },
    ): WhisperCpuVerifiedTokenBlock = synchronized(inferenceLock) {
        check(!closed.get()) { "WhisperEngine is closed." }
        require(encoderState.size == ENCODER_FRAMES * ENCODER_WIDTH) {
            "Whisper encoder state has ${encoderState.size} floats; expected " +
                "${ENCODER_FRAMES * ENCODER_WIDTH}."
        }
        require(proposedTokens.isNotEmpty()) {
            "At least one speculative token is required."
        }
        require(proposedTokens.all { it in 0 until VOCAB_SIZE }) {
            "Speculative block contains an out-of-vocabulary token."
        }
        val basePrompt = teacherForcedPrompt(language, verifiedTokenPrefix)
        require(basePrompt.size + proposedTokens.size <= TARGET_LENGTH) {
            "Speculative block exceeds the decoder's $TARGET_LENGTH-token context."
        }
        val decoderInput = basePrompt + proposedTokens
        interpreter.setCancelled(false)
        checkCancellation(shouldCancel)
        val reusable = buffers ?: InferenceBuffers.allocate().also { buffers = it }
        reusable.encoderState.clear()
        reusable.encoderState.asFloatBuffer().put(encoderState)
        reusable.encoderState.rewind()
        prepareTokens(reusable.tokens, decoderInput, generationConfig.eosTokenId)
        reusable.encoderState.rewind()
        reusable.tokens.rewind()
        reusable.causalMask.rewind()
        reusable.logits.clear()

        val startedNanos = System.nanoTime()
        traced("WhisperCpuSpeculativeVerifyBlock") {
            interpreter.runSignature(
                mapOf(
                    bindings.decodeEncoderState to reusable.encoderState,
                    bindings.decodeTokens to reusable.tokens,
                    bindings.decodeMask to reusable.causalMask,
                ),
                mapOf(bindings.decodeOutput to reusable.logits),
                DECODE_SIGNATURE,
            )
        }
        checkCancellation(shouldCancel)
        val cpuSelections = proposedTokens.indices.map { offset ->
            selectTextToken(
                logits = reusable.logits,
                position = basePrompt.lastIndex + offset,
                firstGeneratedToken = verifiedTokenPrefix.isEmpty() && offset == 0,
                generatedTokens = verifiedTokenPrefix + proposedTokens.take(offset),
            )
        }
        WhisperCpuVerifiedTokenBlock(
            proposedTokens = proposedTokens.toList(),
            cpuSelectedTokens = cpuSelections,
            inferenceMillis = (System.nanoTime() - startedNanos) / 1_000_000L,
        )
    }

    private fun decodePreparedEncoderState(
        reusable: InferenceBuffers,
        language: WhisperLanguage,
        shouldCancel: () -> Boolean,
        onTokenProgress: (generated: Int, limit: Int) -> Unit,
        startedNanos: Long,
        audioDurationMillis: Long,
    ): WhisperChunkResult {
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
            prepareTokens(reusable.tokens, prompt, generationConfig.eosTokenId)
            reusable.encoderState.rewind()
            reusable.tokens.rewind()
            reusable.causalMask.rewind()
            reusable.logits.clear()

            traced("WhisperDecodeStep") {
                interpreter.runSignature(
                    mapOf(
                        bindings.decodeEncoderState to reusable.encoderState,
                        bindings.decodeTokens to reusable.tokens,
                        bindings.decodeMask to reusable.causalMask,
                    ),
                    mapOf(bindings.decodeOutput to reusable.logits),
                    DECODE_SIGNATURE,
                )
            }

            val logitsPosition = prompt.lastIndex
            if (language is WhisperLanguage.Auto && detectedLanguage == null) {
                val languageToken = selectLanguageToken(reusable.logits, logitsPosition)
                detectedLanguage = generationConfig.idToLanguage[languageToken]
                    ?: throw WhisperModelContractException(
                        "Decoder selected unknown language token $languageToken.",
                    )
                prompt += languageToken
                prompt += generationConfig.transcribeTokenId
                if (!timestampsEnabled) prompt += generationConfig.noTimestampsTokenId
                continue
            }

            val nextToken = selectTextToken(
                logits = reusable.logits,
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
        val warnings = if (truncated) {
            listOf("Decoder reached its 128-token limit before EOS.")
        } else {
            emptyList()
        }
        return WhisperChunkResult(
            tokens = generated,
            text = decodeTokens(generated).trim(),
            detectedLanguage = detectedLanguage ?: "und",
            inferenceMillis = (System.nanoTime() - startedNanos) / 1_000_000L,
            truncated = truncated,
            warnings = warnings,
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

    private fun encodeIntoReusable(
        audio: FloatArray,
        reusable: InferenceBuffers,
        shouldCancel: () -> Boolean,
    ) {
        checkCancellation(shouldCancel)
        val features = traced("WhisperFeatureExtraction") {
            featureExtractor.extract(audio)
        }
        reusable.features.clear()
        reusable.features.asFloatBuffer().put(features)
        reusable.features.rewind()
        reusable.encoderState.clear()
        traced("WhisperEncode") {
            interpreter.runSignature(
                mapOf(bindings.encodeInput to reusable.features),
                mapOf(bindings.encodeOutput to reusable.encoderState),
                ENCODE_SIGNATURE,
            )
        }
        reusable.encoderState.rewind()
        checkCancellation(shouldCancel)
    }

    override fun decodeTokens(tokens: List<Int>): String = tokenizer.decode(
        WhisperTimestampTokens.withoutTimestamps(
            tokens = tokens,
            timestampBeginTokenId = generationConfig.timestampBeginTokenId,
        ),
    )

    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        // This can interrupt a currently executing cancellable LiteRT call.
        interpreter.setCancelled(true)
        synchronized(inferenceLock) {
            interpreter.close()
            buffers = null
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

    private fun selectLanguageToken(logits: ByteBuffer, position: Int): Int {
        var selected = -1
        var selectedScore = Float.NEGATIVE_INFINITY
        for (tokenId in generationConfig.idToLanguage.keys) {
            val score = logit(logits, position, tokenId)
            if (!score.isNaN() && score > selectedScore) {
                selected = tokenId
                selectedScore = score
            }
        }
        if (selected < 0) {
            throw WhisperModelContractException("Decoder produced no valid language logits.")
        }
        return selected
    }

    private fun selectTextToken(
        logits: ByteBuffer,
        position: Int,
        firstGeneratedToken: Boolean,
        generatedTokens: List<Int>,
    ): Int {
        if (timestampsEnabled) {
            return WhisperTimestampTokenSelector.select(
                vocabularySize = VOCAB_SIZE,
                generatedTokens = generatedTokens,
                config = generationConfig,
                firstGeneratedToken = firstGeneratedToken,
                logitAt = { tokenId -> logit(logits, position, tokenId) },
            )
        }
        var selected = generationConfig.eosTokenId
        var selectedScore = Float.NEGATIVE_INFINITY
        for (tokenId in 0 until VOCAB_SIZE) {
            val isSpecial = tokenId >= generationConfig.decoderStartTokenId &&
                tokenId != generationConfig.eosTokenId
            val isConfiguredSuppressed = tokenId in generationConfig.suppressTokens
            val isBeginningSuppressed =
                firstGeneratedToken && tokenId in generationConfig.beginSuppressTokens
            if (isSpecial || isConfiguredSuppressed || isBeginningSuppressed) continue

            val score = logit(logits, position, tokenId)
            if (!score.isNaN() && score > selectedScore) {
                selected = tokenId
                selectedScore = score
            }
        }
        return selected
    }

    private fun logit(buffer: ByteBuffer, position: Int, token: Int): Float {
        val floatIndex = position * VOCAB_SIZE + token
        return buffer.getFloat(floatIndex * Float.SIZE_BYTES)
    }

    private fun prepareTokens(buffer: ByteBuffer, prompt: List<Int>, paddingToken: Int) {
        buffer.clear()
        val integers = buffer.asIntBuffer()
        repeat(TARGET_LENGTH) { index ->
            integers.put(if (index < prompt.size) prompt[index] else paddingToken)
        }
        buffer.rewind()
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

    private fun standardPromptLength(): Int =
        if (timestampsEnabled) TIMESTAMP_PROMPT_LENGTH else STANDARD_PROMPT_LENGTH

    private data class InferenceBuffers(
        val features: ByteBuffer,
        val encoderState: ByteBuffer,
        val tokens: ByteBuffer,
        val causalMask: ByteBuffer,
        val logits: ByteBuffer,
    ) {
        companion object {
            fun allocate(): InferenceBuffers {
                val mask = directFloatBuffer(TARGET_LENGTH * TARGET_LENGTH)
                val floats = mask.asFloatBuffer()
                for (row in 0 until TARGET_LENGTH) {
                    for (column in 0 until TARGET_LENGTH) {
                        floats.put(if (column <= row) 0.0f else -1.0e9f)
                    }
                }
                mask.rewind()
                return InferenceBuffers(
                    features = directFloatBuffer(
                        WhisperFeatureExtractor.MEL_BINS * WhisperFeatureExtractor.FRAMES,
                    ),
                    encoderState = directFloatBuffer(ENCODER_FRAMES * ENCODER_WIDTH),
                    tokens = ByteBuffer.allocateDirect(TARGET_LENGTH * Int.SIZE_BYTES)
                        .order(ByteOrder.nativeOrder()),
                    causalMask = mask,
                    logits = directFloatBuffer(TARGET_LENGTH * VOCAB_SIZE),
                )
            }

            private fun directFloatBuffer(elements: Int): ByteBuffer =
                ByteBuffer.allocateDirect(Math.multiplyExact(elements, Float.SIZE_BYTES))
                    .order(ByteOrder.nativeOrder())
        }
    }

    private data class Bindings(
        val encodeInput: String,
        val encodeOutput: String,
        val decodeEncoderState: String,
        val decodeTokens: String,
        val decodeMask: String,
        val decodeOutput: String,
    )

    companion object {
        private const val ENCODE_SIGNATURE = "encode"
        private const val DECODE_SIGNATURE = "decode"
        private const val ENCODER_FRAMES = 1_500
        private const val ENCODER_WIDTH = 1_280
        private const val TARGET_LENGTH = 128
        private const val VOCAB_SIZE = 51_866
        private const val STANDARD_PROMPT_LENGTH = 4
        private const val TIMESTAMP_PROMPT_LENGTH = 3
        private const val MAX_AUDIO_DURATION_MILLIS = 30_000L

        fun open(
            context: Context,
            threadCount: Int = DEFAULT_THREAD_COUNT,
            timestampsEnabled: Boolean = false,
        ): LiteRtWhisperEngine {
            require(threadCount > 0) { "LiteRT thread count must be positive." }
            val applicationContext = context.applicationContext
            val tokenizer = applicationContext.assets.open(WhisperAssets.TOKENIZER).use {
                WhisperTokenizer.load(it)
            }
            val config = applicationContext.assets.open(WhisperAssets.GENERATION_CONFIG).use {
                WhisperGenerationConfig.load(it)
            }
            validateConfig(config)

            val mappedModel = mapModelAsset(applicationContext)
            val options = Interpreter.Options()
                .setNumThreads(threadCount)
                .setUseXNNPACK(true)
                .setCancellable(true)
            val interpreter = Interpreter(mappedModel, options)
            return try {
                val bindings = validateAndResolveBindings(interpreter)
                LiteRtWhisperEngine(
                    interpreter = interpreter,
                    tokenizer = tokenizer,
                    generationConfig = config,
                    bindings = bindings,
                    featureExtractor = WhisperFeatureExtractor(),
                    timestampsEnabled = timestampsEnabled,
                )
            } catch (error: Throwable) {
                interpreter.close()
                throw error
            }
        }

        const val DEFAULT_THREAD_COUNT = 3

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

        private fun mapModelAsset(context: Context): MappedByteBuffer {
            return context.assets.openFd(WhisperAssets.MODEL).use { descriptor ->
                if (descriptor.declaredLength != WhisperAssets.MODEL_SIZE_BYTES) {
                    throw WhisperModelContractException(
                        "Bundled model size is ${descriptor.declaredLength}; " +
                            "expected ${WhisperAssets.MODEL_SIZE_BYTES}.",
                    )
                }
                FileInputStream(descriptor.fileDescriptor).channel.use { channel ->
                    channel.map(
                        FileChannel.MapMode.READ_ONLY,
                        descriptor.startOffset,
                        descriptor.declaredLength,
                    )
                }
            }
        }

        private fun validateAndResolveBindings(interpreter: Interpreter): Bindings {
            val signatureKeys = interpreter.signatureKeys.toSet()
            if (!signatureKeys.containsAll(setOf(ENCODE_SIGNATURE, DECODE_SIGNATURE))) {
                throw WhisperModelContractException(
                    "Model signatures are $signatureKeys; encode and decode are required.",
                )
            }

            val encodeInput = findTensor(
                interpreter = interpreter,
                signature = ENCODE_SIGNATURE,
                input = true,
                preferredName = "args_0",
                type = DataType.FLOAT32,
                shape = intArrayOf(1, 128, 3_000),
            )
            val encodeOutput = findTensor(
                interpreter = interpreter,
                signature = ENCODE_SIGNATURE,
                input = false,
                preferredName = "output_0",
                type = DataType.FLOAT32,
                shape = intArrayOf(1, 1_500, 1_280),
            )
            val decodeEncoderState = findTensor(
                interpreter = interpreter,
                signature = DECODE_SIGNATURE,
                input = true,
                preferredName = "args_0",
                type = DataType.FLOAT32,
                shape = intArrayOf(1, 1_500, 1_280),
            )
            val decodeTokens = findTensor(
                interpreter = interpreter,
                signature = DECODE_SIGNATURE,
                input = true,
                preferredName = "args_1",
                type = DataType.INT32,
                shape = intArrayOf(1, 128),
            )
            val decodeMask = findTensor(
                interpreter = interpreter,
                signature = DECODE_SIGNATURE,
                input = true,
                preferredName = "args_2",
                type = DataType.FLOAT32,
                shape = intArrayOf(1, 1, 128, 128),
            )
            val decodeOutput = findTensor(
                interpreter = interpreter,
                signature = DECODE_SIGNATURE,
                input = false,
                preferredName = "output_0",
                type = DataType.FLOAT32,
                shape = intArrayOf(1, 128, 51_866),
            )
            return Bindings(
                encodeInput,
                encodeOutput,
                decodeEncoderState,
                decodeTokens,
                decodeMask,
                decodeOutput,
            )
        }

        private fun findTensor(
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
                validateTensor("$signature/$preferredName", tensor(preferredName), type, shape)
                return preferredName
            }
            val matching = names.filter { name ->
                val candidate = tensor(name)
                candidate.dataType() == type && candidate.shape().contentEquals(shape)
            }
            if (matching.size != 1) {
                throw WhisperModelContractException(
                    "Cannot uniquely bind $signature $preferredName with " +
                        "$type ${shape.contentToString()}; inputs/outputs are ${names.toList()}.",
                )
            }
            return matching.single()
        }

        private fun validateTensor(
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
    }
}
